# Camera-Based Heart Rate Estimation

A research prototype that estimates heart rate from facial video using imaging photoplethysmography (iPPG). The project extracts changes in facial skin color, produces a pulse signal, and compares the results with contact photoplethysmography (PPG) recordings.

## Motivation

Camera-based pulse measurement could make contactless heart-rate estimation more accessible. This project investigates how signal processing choices and short measurement windows affect the estimated pulse signal and heart rate.

## Processing Pipeline

```
Facial video → face tracking and skin regions → RGB signals → POS → wavelet denoising → band-pass filtering → FFT → estimated heart rate
```

The current video-file workflow uses facial regions from the forehead and cheeks. Experimental scripts also compare POS and CHROM, template-based BPM estimation, and measurements over 5-, 10-, and 15-second windows.

## Project Structure

```
├── ippg/
│   ├── run_ippg.py                  # Entry point for video processing
│   ├── ippg_v2_refactored.py        # Core pipeline: face detection, skin mask, POS, DWT, bandpass, FFT
│   ├── analyze_ground_truth.py      # Read and analyze reference PPG from ground_truth.txt
│   ├── replay_realtime.py           # Frame-by-frame replay from recorded video
│   ├── signal_recording.py          # Record signals, plots, and metadata
│   ├── compare_dev_methods.py       # Experimental: POS vs CHROM comparison
│   ├── trial_short_templates.py     # Experimental: template matching in short windows
│   ├── trial_previous_bpm_real.py   # Experimental: template matching with previous BPM
│   ├── audit_waveform_morphology.py # Waveform inspection before/after bandpass
│   ├── inspect_pre_fft.py           # Pre-FFT signal plots
│   ├── tests/                       # Unit tests
│   └── ...                          # Additional experimental scripts
├── requirements-ippg.txt            # Python dependencies
└── README.md
```

## Installation

### Prerequisites

- Python 3.10+
- A webcam or recorded facial video file

### Setup

```bash
# Clone the repository
git clone https://github.com/tamingakaraponz/Camera-Based-Heart-Rate-Estimation.git
cd Camera-Based-Heart-Rate-Estimation

# Create and activate a virtual environment
python -m venv .venv

# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

# Install dependencies
pip install -r requirements-ippg.txt
```

### Face Landmark Model

This project uses the [MediaPipe Face Landmarker](https://developers.google.com/mediapipe/solutions/vision/face_landmarker) model. Download the model file before running:

1. Download `face_landmarker.task` from the [MediaPipe models page](https://developers.google.com/mediapipe/solutions/vision/face_landmarker#models)
2. Place it in a `models/` directory at the project root:
   ```
   models/face_landmarker.task
   ```

## Usage

```bash
# Run iPPG processing on a video file
python ippg/run_ippg.py
```

> **Note:** You will need to provide your own facial video and, optionally, a `ground_truth.txt` file with reference PPG data. See the script for configuration details.

## Evaluation

Experiments compare camera-derived and reference PPG signals using waveform correlation, frequency-domain comparisons, and heart-rate differences. Results are exploratory and should be interpreted alongside reference timing, signal quality, and available data.

## Current Status

- Local processing of recorded facial video is implemented.
- Signal plots and measurements can be saved for later analysis.
- Frame-by-frame replay has been used to explore short-window estimation.
- A live webcam website is a future goal and has not yet been implemented in the main workflow.

## Limitations

The current results come from a small development dataset. Reference timing and the processing behind device-provided heart-rate values have not been fully verified. **This research prototype is not a medical device and should not be used for diagnosis or treatment.**

## Future Work

- Compare short-window BPM estimators under the same evaluation protocol.
- Improve signal-quality checks and handling of missing data.
- Measure latency and evaluate live webcam performance.
- Develop a browser-based demonstration after validating the processing approach.

## Dependencies

| Package | Version |
|---|---|
| mediapipe | 1.0.1 |
| opencv-contrib-python | 5.0.0.93 |
| matplotlib | 3.11.1 |
| numpy | 2.5.2 |
| PyWavelets | 1.9.0 |
| scipy | 1.18.1 |
| scikit-learn | 1.9.0 |

## License

This project is shared for educational and research purposes. Please check the licenses of individual dependencies before reuse.
