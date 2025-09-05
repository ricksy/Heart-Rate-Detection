import cv2
import mediapipe as mp
import numpy as np
import time
from collections import deque
from scipy.signal import butter, filtfilt, welch, coherence as sig_coherence
from sklearn.decomposition import PCA
import math

mp_face_mesh = mp.solutions.face_mesh

# ========================== Configuration Parameters ==========================
CONFIG = {
    # Image / timing
    'process_width': 640,
    'target_fs': 30.0,
    'window_sec': 20.0,
    'hop_sec': 0.5,

    # Spectral band (Hz) for heart rate
    'bandpass_range': [0.7, 3.0],
    'butter_order': 4,
    'adaptive_filtering': True,

    # Acceptance gates / smoothing
    'snr_db_min': 6.0,
    'quality_min': 60.0,
    'quality_smoothing_alpha': 0.1,
    'low_signal_grace_hops': 3,
    'clamp_bpm': 2.5,
    'roc_max': 6.0,

    # ROI requirements
    'min_roi_pixels': 2000,

    # Landmark smoothing (prevents ROI jitter without resizing the frame)
    'landmark_smooth_alpha': 0.4,  # 0..1 (higher = faster follow, lower = smoother)

    # Cheeks
    'use_multiple_rois': True,
    'cheek_circle_frac': 0.15,
    'cheek_min_px_frac': 0.35,
    'cheek_color': (0, 255, 0),
    'cheek_thickness': 1,
    'cheek_inward_frac': 0.05,

    # Forehead shape adjustments
    'forehead_shrink_px_big': 15,
    'forehead_pad_frac': 0.06,

    # Motion / lighting artifact gates
    'motion_norm_thresh': 0.045,
    'illum_jump_thresh': 0.06,

    # Coherence gating
    'coherence_min': 0.30,

    # Measurement session
    'measurement_sec': 60,
    'min_accepted_points': 10,

    # Head-turn gate
    'yaw_gate_frac': 0.28
}

# ========================== Helpers: geometry & masks ==========================

def unique_indices_from_connections(conns):
    # robust to tuples/Lists of tuples
    idxs = set()
    for c in conns:
        a, b = int(c[0]), int(c[1])
        idxs.add(a); idxs.add(b)
    return sorted(list(idxs))

def as_xy(landmarks, w, h, indices):
    """
    Accepts either:
      - mediapipe landmark object with .landmark
      - numpy array of shape (468, 2) with pixel coords
    """
    if hasattr(landmarks, 'landmark'):
        return np.array([(int(landmarks.landmark[i].x * w),
                          int(landmarks.landmark[i].y * h)) for i in indices], dtype=np.int32)
    else:
        # assume ndarray pixels
        return landmarks[np.array(indices, dtype=int)].astype(np.int32)

def all_landmarks_xy(landmarks, w, h):
    """Return (468,2) array of pixel coords from mediapipe landmarks."""
    return np.array([(landmarks.landmark[i].x * w,
                      landmarks.landmark[i].y * h) for i in range(468)], dtype=np.float32)

def poly_mask(h, w, polygon_pts):
    m = np.zeros((h, w), dtype=np.uint8)
    if polygon_pts is not None and len(polygon_pts) >= 3:
        cv2.fillPoly(m, [polygon_pts.astype(np.int32)], 255)
    return m

def hull_mask(h, w, pts):
    if pts is None or len(pts) < 3:
        return np.zeros((h, w), dtype=np.uint8)
    hull = cv2.convexHull(pts.astype(np.int32))
    return poly_mask(h, w, hull)

def smooth_forehead_outline(face_poly, mask_eyes_dil, eye_dist, pad_frac=0.06, shrink_px=0):
    if face_poly is None or len(face_poly) < 5:
        return None

    top_y = int(np.min(face_poly[:, 1]))
    left_top  = face_poly[np.argmin(face_poly[:, 0])]
    right_top = face_poly[np.argmax(face_poly[:, 0])]

    ys, _ = np.where(mask_eyes_dil > 0)
    if ys.size == 0:
        return None
    y_cut = max(0, int(ys.min()) - int(pad_frac * eye_dist))

    sL = int(max(0, shrink_px) // 2)
    sR = int(max(0, shrink_px) - sL)
    x_left  = int(left_top[0]  + sL)
    x_right = int(right_top[0] - sR)
    if x_right <= x_left:
        x_left, x_right = int(left_top[0]), int(right_top[0])

    return np.array([[x_left, top_y],
                     [x_right, top_y],
                     [x_right, y_cut],
                     [x_left,  y_cut]], dtype=np.int32)

def get_cheek_centers_and_radius(lm, w, h, eye_dist):
    left_indices  = [123, 50, 36, 137, 93]
    right_indices = [352, 280, 266, 366, 323]
    l = np.mean(as_xy(lm, w, h, left_indices),  axis=0).astype(np.float32)
    r = np.mean(as_xy(lm, w, h, right_indices), axis=0).astype(np.float32)

    le = np.mean(as_xy(lm, w, h, [33,133]), axis=0).astype(np.float32)
    re = np.mean(as_xy(lm, w, h, [362,263]), axis=0).astype(np.float32)
    v  = re - le
    nv = v / (np.linalg.norm(v) + 1e-6)

    da = float(CONFIG['cheek_inward_frac']) * eye_dist
    l = l + da * nv
    r = r - da * nv

    radius = max(2, int(CONFIG['cheek_circle_frac'] * eye_dist))
    return l.astype(int), r.astype(int), radius

def head_yaw_fraction(lm, w, h):
    le = np.mean(as_xy(lm, w, h, [33,133]), axis=0).astype(np.float32)
    re = np.mean(as_xy(lm, w, h, [362,263]), axis=0).astype(np.float32)
    eye_mid = 0.5*(le + re)
    nose = as_xy(lm, w, h, [1])[0].astype(np.float32)
    eye_dist = float(np.linalg.norm(re - le) + 1e-6)
    yaw = (nose[0] - eye_mid[0]) / eye_dist
    return abs(float(yaw)), eye_dist

def build_face_masks(frame, lm, face_mesh_module, eye_scale=0.30):
    h, w, _ = frame.shape
    face_idx = unique_indices_from_connections(face_mesh_module.FACEMESH_FACE_OVAL)
    leye_idx = unique_indices_from_connections(face_mesh_module.FACEMESH_LEFT_EYE)
    reye_idx = unique_indices_from_connections(face_mesh_module.FACEMESH_RIGHT_EYE)

    face_poly = as_xy(lm, w, h, face_idx)
    leye_pts  = as_xy(lm, w, h, leye_idx)
    reye_pts  = as_xy(lm, w, h, reye_idx)

    mask_face = poly_mask(h, w, face_poly)
    mask_leye = hull_mask(h, w, leye_pts)
    mask_reye = hull_mask(h, w, reye_pts)
    mask_eyes = cv2.bitwise_or(mask_leye, mask_reye)

    eye_center_left  = np.mean(leye_pts, axis=0)
    eye_center_right = np.mean(reye_pts, axis=0)
    eye_dist = float(np.linalg.norm(eye_center_left - eye_center_right) + 1e-6)

    eye_ks = max(3, int(2 * int(max(1, eye_scale * eye_dist)) + 1))
    k_eye = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (eye_ks, eye_ks))
    mask_eyes_dil = cv2.dilate(mask_eyes, k_eye, iterations=1)

    l_center, r_center, radius = get_cheek_centers_and_radius(lm, w, h, eye_dist)

    return {
        "mask_face": mask_face,
        "mask_eyes_dil": mask_eyes_dil,
        "face_poly": face_poly,
        "eye_dist": eye_dist,
        "cheek_centers": (l_center, r_center),
        "cheek_radius": radius,
    }

# ============================ Signal processing helpers ========================

def butter_bandpass(lowcut, highcut, fs, order=3):
    nyq = max(1e-6, 0.5 * fs)
    low = max(1e-6, lowcut / nyq)
    high = min(0.999, highcut / nyq)
    if high <= low:
        high = min(0.999, low + 0.05)
    b, a = butter(order, [low, high], btype='band')
    return b, a

def bandpass_filter(data, lowcut, highcut, fs, order=3):
    x = np.asarray(data, dtype=float)
    if len(x) < max(8*order, 32) or fs <= 0:
        return x - np.mean(x)
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    try:
        y = filtfilt(b, a, x)
    except Exception:
        y = x - np.mean(x)
    return y

def adaptive_bandpass_filter(data, fs, prev_hr=None):
    if prev_hr is None:
        lowcut, highcut = CONFIG['bandpass_range']
    else:
        hr_hz = max(0.8, min(3.0, float(prev_hr) / 60.0))
        lowcut = max(0.7, hr_hz - 0.35)
        highcut = min(3.0, hr_hz + 0.45)
    return bandpass_filter(data, lowcut, highcut, fs, order=CONFIG['butter_order'])

def moving_average_detrend(x, win_len):
    x = np.asarray(x, dtype=float)
    if win_len <= 1 or win_len >= len(x):
        return x - np.mean(x)
    c = np.cumsum(np.insert(x, 0, 0.0))
    ma = (c[win_len:] - c[:-win_len]) / float(win_len)
    pad_left = win_len // 2
    pad_right = len(x) - len(ma) - pad_left
    ma_full = np.pad(ma, (pad_left, pad_right), mode='edge')
    return x - ma_full

def pca_method(r, g, b):
    r_norm = (r - np.mean(r)) / (np.std(r) + 1e-8)
    g_norm = (g - np.mean(g)) / (np.std(g) + 1e-8)
    b_norm = (b - np.mean(b)) / (np.std(b) + 1e-8)
    X = np.vstack([r_norm, g_norm, b_norm]).T
    pca = PCA(n_components=2)
    Y = pca.fit_transform(X)
    return Y[:, 1] if Y.shape[1] >= 2 else Y[:, 0]

def chrom_method(r, g, b):
    r = np.asarray(r); g = np.asarray(g); b = np.asarray(b)
    def znorm(x):
        x = x - np.mean(x); s = np.std(x) + 1e-8; return x / s
    r_n, g_n, b_n = znorm(r), znorm(g), znorm(b)
    x1 = 3.0 * r_n - 2.0 * g_n
    x2 = 1.5 * r_n + g_n - 1.5 * b_n
    alpha = (np.std(x1) + 1e-8) / (np.std(x2) + 1e-8)
    return x1 - alpha * x2

def pos_method(r, g, b):
    r = np.asarray(r); g = np.asarray(g); b = np.asarray(b)
    r_n = (r - np.mean(r)) / (np.std(r) + 1e-8)
    g_n = (g - np.mean(g)) / (np.std(g) + 1e-8)
    b_n = (b - np.mean(b)) / (np.std(b) + 1e-8)
    return r_n + g_n - b_n

def resample_to_uniform(t, x, target_fs=30.0):
    t = np.asarray(t, dtype=float)
    x = np.asarray(x, dtype=float)
    if len(t) < 3 or len(x) < 3 or len(t) != len(x):
        return None, None
    t0, t1 = t[0], t[-1]
    dur = max(1e-6, t1 - t0)
    n = int(max(16, round(dur * target_fs)))
    if n < 16:
        return None, None
    t_uniform = np.linspace(t0, t1, n, endpoint=True)
    x_uniform = np.interp(t_uniform, t, x)
    return x_uniform, float(target_fs)

def refine_peak_parabolic(freqs, psd, idx):
    if idx <= 0 or idx >= len(psd) - 1:
        return float(freqs[idx])
    a, b, c = psd[idx-1], psd[idx], psd[idx+1]
    denom = (a - 2*b + c)
    if abs(denom) < 1e-12:
        return float(freqs[idx])
    delta = 0.5 * (a - c) / denom
    return float(freqs[idx] + delta * (freqs[1] - freqs[0]))

def psd_peak_metrics(x, fs, fmin, fmax):
    x = np.asarray(x, dtype=float)
    if len(x) < 32 or fs <= 0:
        return None
    nper = min(512, max(64, len(x)//2))
    freqs, psd = welch(x, fs=fs, window='hann', nperseg=nper, noverlap=nper//2, detrend='constant')
    band = (freqs >= fmin) & (freqs <= fmax)
    if not np.any(band):
        return None
    f = freqs[band]; p = psd[band]
    ipk = int(np.argmax(p))
    fpk = refine_peak_parabolic(f, p, ipk)
    notch = (np.abs(f - f[ipk]) <= 0.10)
    noise = float(np.median(p[~notch])) if np.any(~notch) else float(np.median(p))
    snr_db = 10.0 * math.log10((p[ipk] + 1e-12) / (noise + 1e-12))
    quality = max(0.0, min(100.0, 20.0 * (snr_db - 1.0)))
    return f, p, fpk, snr_db, quality

def ema(prev, new, alpha=0.20):
    return new if prev is None else (1 - alpha) * prev + alpha * new

def pick_peak_with_harmonics(freqs, mag, prev_bpm):
    if len(freqs) == 0 or len(mag) == 0:
        return None
    order = np.argsort(mag)[::-1][:5]
    cand = [(freqs[i]*60.0, mag[i]) for i in order]
    if prev_bpm is not None:
        lo, hi = 0.70*prev_bpm, 1.30*prev_bpm
        near = [c for c in cand if lo <= c[0] <= hi]
        if near:
            cand = sorted(near, key=lambda x: x[1], reverse=True)
    if not cand:
        return None
    bpm, p = cand[0]
    for (b2, p2) in cand[1:]:
        if 40 <= b2 <= 120 and abs(bpm - 2*b2) <= 8 and p2 > 0.35*p:
            return b2
        if 40 <= bpm <= 90 and abs(2*bpm - b2) <= 8 and p2 > 1.2*p:
            return b2
    return bpm

def weighted_median(values, weights):
    v = np.asarray(values, dtype=float)
    w = np.asarray(weights, dtype=float)
    if len(v) == 0 or np.sum(w) <= 0:
        return None
    order = np.argsort(v)
    v = v[order]; w = w[order]
    cdf = np.cumsum(w)
    cutoff = 0.5 * np.sum(w)
    idx = int(np.searchsorted(cdf, cutoff))
    idx = min(max(idx, 0), len(v)-1)
    return float(v[idx])

def srgb_to_linear(bgr):
    x = bgr.astype(np.float32) / 255.0
    a = 0.055
    lin = np.where(x <= 0.04045, x/12.92, ((x + a)/(1+a))**2.4)
    return lin

def skin_mask(img_bgr):
    ycrcb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YCrCb)
    m = cv2.inRange(ycrcb, (0, 133, 77), (255, 173, 127))
    m = cv2.medianBlur(m, 5)
    return m

def band_coherence(x, y, fs, fmin, fmax):
    if x is None or y is None:
        return 0.0
    x = np.asarray(x, dtype=float); y = np.asarray(y, dtype=float)
    if len(x) < 32 or len(y) < 32:
        return 0.0
    f, Cxy = sig_coherence(x, y, fs=fs, nperseg=min(256, len(x)))
    band = (f >= fmin) & (f <= fmax)
    if not np.any(band):
        return 0.0
    return float(np.nanmean(Cxy[band]))

# ================================ Main =========================================

def main():
    cap = cv2.VideoCapture(0)
    try:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
        cap.set(cv2.CAP_PROP_EXPOSURE, -6)
        cap.set(cv2.CAP_PROP_AUTO_WB, 0)
        cap.set(cv2.CAP_PROP_WB_TEMPERATURE, 4500)
    except Exception:
        pass

    if not cap.isOpened():
        print("Could not open camera.")
        return

    actual_fps = cap.get(cv2.CAP_PROP_FPS)
    if actual_fps and actual_fps > 0:
        CONFIG['target_fs'] = float(actual_fps)

    max_samples = int(CONFIG['target_fs'] * (CONFIG['window_sec'] + 2))
    r_sig, g_sig, b_sig, t_sig = deque(maxlen=max_samples), deque(maxlen=max_samples), deque(maxlen=max_samples), deque(maxlen=max_samples)
    r_cheek1, g_cheek1, b_cheek1 = deque(maxlen=max_samples), deque(maxlen=max_samples), deque(maxlen=max_samples)
    r_cheek2, g_cheek2, b_cheek2 = deque(maxlen=max_samples), deque(maxlen=max_samples), deque(maxlen=max_samples)

    last_analysis_time = 0.0
    prev_bpm = None
    last_bpm_time = time.time()
    bpm_display = None
    signal_quality = 0.0
    low_signal_streak = 0
    prev_face_center = None
    prev_forehead_mean = None

    # Kalman filter
    KF = {
        'x': np.array([[70.0],[0.0]], dtype=float),
        'P': np.eye(2)*50.0,
        'F': np.array([[1.0, CONFIG['hop_sec']],
                       [0.0, 1.0]], dtype=float),
        'Q': np.diag([0.5, 2.0]),
        'H': np.array([[1.0, 0.0]], dtype=float),
        'R': np.array([[4.0]], dtype=float)
    }

    # Measurement session
    measure_started = False
    measure_start_time = None
    measure_done = False
    log_bpm, log_quality, log_snrdb, log_time = [], [], [], []

    last_fused = None
    last_fs = None
    MIN_CHEEK_PX = int(CONFIG['min_roi_pixels'] * CONFIG['cheek_min_px_frac'])

    # Landmark smoothing state
    smoothed_lm_xy = None
    alpha_lm = float(CONFIG['landmark_smooth_alpha'])

    with mp_face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7
    ) as face_mesh:

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            H0, W0, _ = frame.shape
            scale = 1.0
            if W0 > CONFIG['process_width']:
                scale = CONFIG['process_width'] / W0
                proc = cv2.resize(frame, (int(W0*scale), int(H0*scale)), interpolation=cv2.INTER_LINEAR)
            else:
                proc = frame

            h, w, _ = proc.shape
            rgb = cv2.cvtColor(proc, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb)

            draw_img = proc.copy()

            if results.multi_face_landmarks:
                lm_raw = results.multi_face_landmarks[0]

                # ---- Landmark EMA smoothing (no frame warp) ----
                cur_xy = all_landmarks_xy(lm_raw, w, h)      # (468,2) pixels
                if smoothed_lm_xy is None:
                    smoothed_lm_xy = cur_xy
                else:
                    smoothed_lm_xy = (1 - alpha_lm) * smoothed_lm_xy + alpha_lm * cur_xy

                lm_for_masks = smoothed_lm_xy  # downstream as_xy() accepts ndarray

                # Head-turn gate
                yaw_abs, eye_dist_for_gate = head_yaw_fraction(lm_for_masks, w, h)
                if eye_dist_for_gate < 10 or yaw_abs > CONFIG['yaw_gate_frac']:
                    cv2.putText(draw_img, "Face turned too much. Look forward.", (30, 36),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)
                    cv2.imshow("rPPG (1-min Measurement)", draw_img)
                    if cv2.waitKey(1) & 0xFF == 27:
                        break
                    continue

                masks = build_face_masks(draw_img, lm_for_masks, mp_face_mesh, eye_scale=0.30)
                mask_eyes_dil_sm  = masks["mask_eyes_dil"]
                eye_dist          = masks["eye_dist"]
                face_poly_small   = masks["face_poly"]
                cheek_centers_sm  = masks["cheek_centers"]
                cheek_radius_sm   = masks["cheek_radius"]

                # Forehead polygon
                shrink_small = max(0, int(CONFIG['forehead_shrink_px_big'] * scale))
                forehead_poly_small = smooth_forehead_outline(
                    face_poly_small, mask_eyes_dil_sm, eye_dist,
                    pad_frac=CONFIG['forehead_pad_frac'], shrink_px=shrink_small
                )

                roi_pixels = 0
                if forehead_poly_small is not None:
                    forehead_poly = forehead_poly_small.astype(np.int32)
                    cv2.polylines(draw_img, [forehead_poly], isClosed=True, color=(0, 255, 255), thickness=2)
                    forehead_mask = np.zeros((h, w), dtype=np.uint8)
                    cv2.fillPoly(forehead_mask, [forehead_poly], 255)
                    roi_pixels = int(np.count_nonzero(forehead_mask))
                else:
                    forehead_mask = None

                # Precompute once per frame
                lin_frame = srgb_to_linear(draw_img)
                skin = skin_mask(draw_img)

                # Forehead means
                if forehead_mask is not None and roi_pixels > CONFIG['min_roi_pixels']:
                    forehead_skin = cv2.bitwise_and(forehead_mask, skin)
                    b_mean = cv2.mean(lin_frame[:, :, 0], mask=forehead_skin)[0]
                    g_mean = cv2.mean(lin_frame[:, :, 1], mask=forehead_skin)[0]
                    r_mean = cv2.mean(lin_frame[:, :, 2], mask=forehead_skin)[0]
                else:
                    b_mean = g_mean = r_mean = None

                # Cheeks
                cheek_means = []
                (lc_sm, rc_sm) = cheek_centers_sm
                radius = max(2, int(cheek_radius_sm))
                lc = (int(lc_sm[0]), int(lc_sm[1]))
                rc = (int(rc_sm[0]), int(rc_sm[1]))

                cv2.circle(draw_img, lc, radius, CONFIG['cheek_color'], CONFIG['cheek_thickness'])
                cv2.circle(draw_img, rc, radius, CONFIG['cheek_color'], CONFIG['cheek_thickness'])

                left_mask  = np.zeros((h, w), dtype=np.uint8);  cv2.circle(left_mask,  lc, radius, 255, -1)
                right_mask = np.zeros((h, w), dtype=np.uint8);  cv2.circle(right_mask, rc, radius, 255, -1)

                left_skin  = cv2.bitwise_and(left_mask, skin)
                right_skin = cv2.bitwise_and(right_mask, skin)

                cheek_roi_pixels = [int(np.count_nonzero(left_skin)), int(np.count_nonzero(right_skin))]

                if cheek_roi_pixels[0] > int(CONFIG['min_roi_pixels'] * CONFIG['cheek_min_px_frac']):
                    b_c = cv2.mean(lin_frame[:, :, 0], mask=left_skin)[0]
                    g_c = cv2.mean(lin_frame[:, :, 1], mask=left_skin)[0]
                    r_c = cv2.mean(lin_frame[:, :, 2], mask=left_skin)[0]
                    cheek_means.append((r_c, g_c, b_c))
                else:
                    cheek_means.append((None, None, None))

                if cheek_roi_pixels[1] > int(CONFIG['min_roi_pixels'] * CONFIG['cheek_min_px_frac']):
                    b_c = cv2.mean(lin_frame[:, :, 0], mask=right_skin)[0]
                    g_c = cv2.mean(lin_frame[:, :, 1], mask=right_skin)[0]
                    r_c = cv2.mean(lin_frame[:, :, 2], mask=right_skin)[0]
                    cheek_means.append((r_c, g_c, b_c))
                else:
                    cheek_means.append((None, None, None))

                # Motion & illumination penalties
                face_center = np.mean(face_poly_small, axis=0)
                motion_penalty = 0.0
                if prev_face_center is not None and eye_dist > 1.0:
                    dt_est = max(1e-3, 1.0/CONFIG['target_fs'])
                    disp = float(np.linalg.norm(face_center - prev_face_center))
                    motion_rate = (disp / eye_dist) / dt_est
                    if motion_rate > CONFIG['motion_norm_thresh']:
                        motion_penalty = min(30.0, 200.0 * (motion_rate - CONFIG['motion_norm_thresh']))
                prev_face_center = face_center

                illum_penalty = 0.0
                if prev_forehead_mean is not None and g_mean is not None and prev_forehead_mean > 1e-6:
                    rel_jump = abs(g_mean - prev_forehead_mean) / (prev_forehead_mean + 1e-6)
                    if rel_jump > CONFIG['illum_jump_thresh']:
                        illum_penalty = min(30.0, 300.0 * (rel_jump - CONFIG['illum_jump_thresh']))
                if g_mean is not None:
                    prev_forehead_mean = g_mean

                # Buffer signals
                now_ts = time.time()
                if b_mean is not None and np.isfinite(b_mean) and np.isfinite(g_mean) and np.isfinite(r_mean):
                    r_sig.append(r_mean); g_sig.append(g_mean); b_sig.append(b_mean); t_sig.append(now_ts)
                    (r_cL, g_cL, b_cL), (r_cR, g_cR, b_cR) = cheek_means
                    if b_cL is not None and np.isfinite(b_cL):
                        r_cheek1.append(r_cL); g_cheek1.append(g_cL); b_cheek1.append(b_cL)
                    if b_cR is not None and np.isfinite(b_cR):
                        r_cheek2.append(r_cR); g_cheek2.append(g_cR); b_cheek2.append(b_cR)

                # HR estimation
                if len(t_sig) >= 3 and roi_pixels > CONFIG['min_roi_pixels'] and (t_sig[-1] - t_sig[0]) >= 5.0:
                    t_arr = np.array(t_sig, dtype=float)
                    r_arr = np.array(r_sig, dtype=float)
                    g_arr = np.array(g_sig, dtype=float)
                    b_arr = np.array(b_sig, dtype=float)

                    has_cheek1 = (len(r_cheek1) == len(r_sig)) and (len(r_cheek1) > 0)
                    has_cheek2 = (len(r_cheek2) == len(r_sig)) and (len(r_cheek2) > 0)

                    r_u, fs_u = resample_to_uniform(t_arr, r_arr, target_fs=CONFIG['target_fs'])
                    g_u, _    = resample_to_uniform(t_arr, g_arr, target_fs=CONFIG['target_fs'])
                    b_u, _    = resample_to_uniform(t_arr, b_arr, target_fs=CONFIG['target_fs'])

                    if r_u is not None and fs_u is not None:
                        now_wall = time.time()
                        if (now_wall - last_analysis_time) >= CONFIG['hop_sec']:
                            last_analysis_time = now_wall

                            ma_len = max(5, int(fs_u * 1.2))
                            r_dt = moving_average_detrend(r_u, ma_len)
                            g_dt = moving_average_detrend(g_u, ma_len)
                            b_dt = moving_average_detrend(b_u, ma_len)

                            def detrend_cheek(arr):
                                if len(arr) == 0: return None
                                x_u, _ = resample_to_uniform(t_arr, np.array(arr, dtype=float), target_fs=CONFIG['target_fs'])
                                return moving_average_detrend(x_u, ma_len) if x_u is not None else None

                            r_c1_dt = detrend_cheek(r_cheek1) if has_cheek1 else None
                            g_c1_dt = detrend_cheek(g_cheek1) if has_cheek1 else None
                            b_c1_dt = detrend_cheek(b_cheek1) if has_cheek1 else None
                            r_c2_dt = detrend_cheek(r_cheek2) if has_cheek2 else None
                            g_c2_dt = detrend_cheek(g_cheek2) if has_cheek2 else None
                            b_c2_dt = detrend_cheek(b_cheek2) if has_cheek2 else None

                            x_chrom_fh = chrom_method(r_dt, g_dt, b_dt)
                            x_green    = g_dt - np.mean(g_dt)
                            x_pca      = pca_method(r_dt, g_dt, b_dt)
                            x_pos      = pos_method(r_dt, g_dt, b_dt)

                            fmin, fmax = CONFIG['bandpass_range']
                            signals = [(x_chrom_fh, 'FH')]
                            c1_bp = c2_bp = None
                            if has_cheek1 and r_c1_dt is not None and g_c1_dt is not None and b_c1_dt is not None:
                                c1_raw = chrom_method(r_c1_dt, g_c1_dt, b_c1_dt)
                                c1_bp = adaptive_bandpass_filter(c1_raw, fs_u, prev_bpm) if CONFIG['adaptive_filtering'] \
                                        else bandpass_filter(c1_raw, fmin, fmax, fs_u, order=CONFIG['butter_order'])
                                signals.append((c1_bp, 'C1'))
                            if has_cheek2 and r_c2_dt is not None and g_c2_dt is not None and b_c2_dt is not None:
                                c2_raw = chrom_method(r_c2_dt, g_c2_dt, b_c2_dt)
                                c2_bp = adaptive_bandpass_filter(c2_raw, fs_u, prev_bpm) if CONFIG['adaptive_filtering'] \
                                        else bandpass_filter(c2_raw, fmin, fmax, fs_u, order=CONFIG['butter_order'])
                                signals.append((c2_bp, 'C2'))

                            fused = None
                            weights = []
                            parts = []
                            for sig, tag in signals:
                                x_bp = adaptive_bandpass_filter(sig, fs_u, prev_bpm) if (CONFIG['adaptive_filtering'] and tag=='FH') else \
                                       bandpass_filter(sig, fmin, fmax, fs_u, order=CONFIG['butter_order'])
                                m = psd_peak_metrics(x_bp, fs_u, fmin, fmax)
                                if m is None:
                                    continue
                                _, _, _, snr_db_tmp, _ = m
                                wgt = max(0.0, snr_db_tmp - 1.0)
                                parts.append((x_bp, wgt)); weights.append(wgt)

                            if parts and sum(weights) > 0:
                                fused = np.sum([wgt * x for (x, wgt) in parts], axis=0) / (sum(weights) + 1e-9)
                            else:
                                fused = adaptive_bandpass_filter(x_chrom_fh, fs_u, prev_bpm) if CONFIG['adaptive_filtering'] \
                                        else bandpass_filter(x_chrom_fh, fmin, fmax, fs_u, order=CONFIG['butter_order'])

                            last_fused = fused.copy()
                            last_fs = fs_u

                            methods = [
                                ('CHROM', fused),
                                ('GREEN', bandpass_filter(x_green, fmin, fmax, fs_u, order=CONFIG['butter_order'])),
                                ('PCA',   bandpass_filter(x_pca,   fmin, fmax, fs_u, order=CONFIG['butter_order'])),
                                ('POS',   bandpass_filter(x_pos,   fmin, fmax, fs_u, order=CONFIG['butter_order'])),
                            ]

                            cand = []
                            for name, sig in methods:
                                m = psd_peak_metrics(sig, fs_u, fmin, fmax)
                                if m is None:
                                    continue
                                f, p, fpk, snr_db, q = m
                                bpm_pick = pick_peak_with_harmonics(f, p, prev_bpm)
                                if bpm_pick is not None:
                                    q_eff = max(0.0, q - motion_penalty - illum_penalty)
                                    cand.append((name, bpm_pick, snr_db, q_eff, sig))

                            accepted = False
                            bpm_new = None
                            snr_db_used = -np.inf
                            qual_used = 0.0

                            if cand:
                                cand.sort(key=lambda z: (z[2] * 0.7 + z[3] * 0.3), reverse=True)
                                method_used, bpm_new, snr_db_used, qual_used, sig_used = cand[0]

                                coh_vals = []
                                if c1_bp is not None:
                                    coh_vals.append(band_coherence(sig_used, c1_bp, fs_u, fmin, fmax))
                                if c2_bp is not None:
                                    coh_vals.append(band_coherence(sig_used, c2_bp, fs_u, fmin, fmax))
                                coh = float(np.nanmean(coh_vals)) if len(coh_vals) > 0 else 1.0

                                if (bpm_new is not None and snr_db_used >= CONFIG['snr_db_min']
                                    and 40 <= bpm_new <= 190 and qual_used >= CONFIG['quality_min']
                                    and coh >= CONFIG['coherence_min']):
                                    accepted = True

                                signal_quality = ema(signal_quality, qual_used if accepted else 0.0, CONFIG['quality_smoothing_alpha'])
                            else:
                                bpm_new = None
                                snr_db_used = -np.inf
                                signal_quality = ema(signal_quality, 0.0, CONFIG['quality_smoothing_alpha'])

                            if accepted:
                                if not measure_started:
                                    measure_started = True
                                    measure_start_time = now_wall

                                low_signal_streak = 0

                                # Kalman filter update
                                KF['x'] = KF['F'] @ KF['x']
                                KF['P'] = KF['F'] @ KF['P'] @ KF['F'].T + KF['Q']
                                y_res = np.array([[bpm_new]]) - KF['H'] @ KF['x']
                                S = KF['H'] @ KF['P'] @ KF['H'].T + KF['R']
                                K = KF['P'] @ KF['H'].T @ np.linalg.inv(S)
                                KF['x'] = KF['x'] + K @ y_res
                                KF['P'] = (np.eye(2) - K @ KF['H']) @ KF['P']

                                bpm_display = float(KF['x'][0,0])

                                prev_disp = prev_bpm if prev_bpm is not None else bpm_display
                                bpm_tmp = float(np.clip(bpm_display, prev_disp - CONFIG['clamp_bpm'], prev_disp + CONFIG['clamp_bpm']))
                                dt_disp = max(1e-3, now_wall - last_bpm_time)
                                max_delta = CONFIG['roc_max'] * dt_disp
                                if prev_bpm is not None:
                                    bpm_tmp = float(np.clip(bpm_tmp, prev_bpm - max_delta, prev_bpm + max_delta))
                                bpm_display = bpm_tmp

                                last_bpm_time = now_wall
                                prev_bpm = bpm_display

                                weight = max(0.1, (snr_db_used - CONFIG['snr_db_min'] + 1.0)) * (qual_used/100.0 + 1e-6)
                                log_bpm.append(float(bpm_new))
                                log_quality.append(float(qual_used))
                                log_snrdb.append(float(snr_db_used))
                                log_time.append(now_wall)
                            else:
                                low_signal_streak += 1
                                if low_signal_streak >= CONFIG['low_signal_grace_hops']:
                                    bpm_display = None

                # ----- UI -----
                hr_text = "Heart Rate: -- bpm" if bpm_display is None else f"Heart Rate: {int(round(bpm_display))} bpm"
                cv2.putText(draw_img, hr_text, (30, 36), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 2)

                if measure_started and not measure_done:
                    elapsed = time.time() - measure_start_time
                    remaining = max(0, int(CONFIG['measurement_sec'] - elapsed))
                    cv2.putText(draw_img, f"Final heart rate reading in: {remaining:02d}s",
                                (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 2)
                    if 10 <= elapsed < 15:
                        cv2.putText(draw_img, f"Signal quality ~ {int(round(signal_quality))}/100",
                                    (30, 86), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,128,0), 2)
                    if elapsed >= CONFIG['measurement_sec']:
                        measure_done = True

                if measure_done:
                    final_bpm = None
                    final_conf = None
                    if len(log_bpm) >= CONFIG['min_accepted_points']:
                        w = np.array([max(0.1, (s - CONFIG['snr_db_min'] + 1.0)) * (q/100.0 + 1e-6)
                                      for s, q in zip(log_snrdb, log_quality)], dtype=float)
                        final_bpm = weighted_median(log_bpm, w)
                        med = np.median(log_bpm)
                        mad = np.median(np.abs(np.array(log_bpm) - med))
                        spread = 1.4826 * mad
                        final_conf = max(0.0, 100.0 - min(50.0, spread))
                    elif last_fused is not None and last_fs is not None:
                        fmin, fmax = CONFIG['bandpass_range']
                        m = psd_peak_metrics(last_fused, last_fs, fmin, fmax)
                        if m is not None:
                            f, p, fpk, snr_db, q = m
                            bpm_pick = pick_peak_with_harmonics(f, p, prev_bpm)
                            if bpm_pick is not None:
                                final_bpm = float(bpm_pick)
                                final_conf = q

                    if final_bpm is not None:
                        out = f"FINAL HR: {int(round(final_bpm))} bpm"
                        if final_conf is not None:
                            out += f"  (conf ~{int(round(final_conf))}%)"
                        print(out)
                        cv2.putText(draw_img, out, (140, 140), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255,0,0), 3)
                    else:
                        print("FINAL HR: unavailable (insufficient signal)")
                        cv2.putText(draw_img, "FINAL HR: unavailable", (30, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,0), 2)

                    cv2.imshow("rPPG (1-min Result)", draw_img)
                    cv2.waitKey(5500)
                    break

                cv2.imshow("rPPG (1-min Measurement)", draw_img)

            else:
                cv2.imshow("rPPG (1-min Measurement)", proc)

            if cv2.waitKey(1) & 0xFF == 27:  # ESC
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # ⚠️ Research use only. Validate against clinical references before any medical use.
    main()
