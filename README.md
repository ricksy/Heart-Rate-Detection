# Heart Rate Detection

This project estimates heart rate from a webcam video stream using remote photoplethysmography (rPPG) and facial landmark detection.

## Features

- Real-time heart rate estimation using a standard webcam
- Uses Mediapipe Face Mesh for robust facial landmark detection
- Multiple region-of-interest (ROI) strategies (forehead and cheeks)
- Signal processing pipeline with detrending, bandpass filtering, and PCA/CHROM/POS methods
- Quality gating and confidence estimation

## Requirements

- Python 3.12+
- OpenCV
- Mediapipe
- NumPy
- SciPy
- scikit-learn

Install dependencies with:

```sh
pip install -r requirements.txt
```