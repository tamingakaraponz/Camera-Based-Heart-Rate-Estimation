# Camera-Based Heart Rate Estimation

A research prototype that estimates heart rate from facial video using imaging photoplethysmography (iPPG). The project extracts changes in facial skin color, produces a pulse signal, and compares the results with contact photoplethysmography (PPG) recordings.

## Motivation

Camera-based pulse measurement could make contactless heart-rate estimation more accessible. This project investigates how signal processing choices and short measurement windows affect the estimated pulse signal and heart rate.

## Processing Pipeline

```
Facial video → face tracking and skin regions → RGB signals → POS → band-pass filtering → FFT → estimated heart rate
```

The current video-file workflow uses facial regions from the forehead and cheeks. Experimental scripts also compare POS and CHROM, template-based BPM estimation, and measurements over 5-, 10-, and 15-second windows.

The main workflow filters and searches for heart rate in **0.6–3.0 Hz (36–180 BPM)**. Historical experiments may explicitly use the earlier 0.6–2.0 Hz range.

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

- Python compatible with the pinned packages in `requirements-ippg.txt`. Local experiments used Python 3.14; a clean installation on other versions has not been verified.
- A recorded facial video file

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
python ippg/run_ippg.py "path/to/video.avi" --no-show
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

## System Overview

The [iPPG component](ippg/README.md) owns recorded-video extraction, pulse analysis, reference comparison and local experiment runners. [Pipeline notes](ippg/PIPELINE.md) describe the processing stages. The main CLI is `ippg/run_ippg.py:main`; optional experiments have separate inputs and output policies.

## Interface

- [ippg/README.md](ippg/README.md) documents the component interface, invariants and side effects.
- [requirements-ippg.txt](requirements-ippg.txt) specifies the pinned Python dependencies.
- [ippg/assets/haarcascade_frontalface_default.xml](ippg/assets/haarcascade_frontalface_default.xml) supplies a bundled Haar face-detection asset.
- `ippg/tests/` holds verification code. Data, generated results, model downloads, environments and third-party folders are excluded from this documentation bootstrap according to `.gitignore`.

## Invariants

- `ippg/run_ippg.py:main` selects POS, legacy ROI, no DWT and no background compensation. DWT, CHROM and template estimators remain available in separate experimental workflows.
- `ippg/ippg_v2_refactored.py:resample_to_fps` requires strictly increasing finite timestamps, matching finite RGB arrays and positive FPS. RGB uses R/G/B channel order.
- `ippg/replay_realtime.py:ReplayEngine.push` evaluates each update before accepting a later frame. Replay is unpaced and has not established live-camera throughput.
- `ippg/analyze_ground_truth.py:analyze_file` interprets three rows as waveform, supplied BPM and time. It estimates BPM from waveform and time; waveform/clock validity remains important when interpreting comparisons.

## Side Effects

- `ippg/run_ippg.py:main` reads video/model files and saves successful results through `ippg/signal_recording.py:save_ippg_run` under `outputs/signal_runs` by default.
- `ippg/run_ippg.py:PROJECT_PYTHON` and `ippg/replay_realtime.py:PROJECT_PYTHON` may relaunch with a Windows `.venv` in the checkout's parent directory when run as scripts. This existing local convention can override the active interpreter.
- Experimental modules may write outputs during import. Consult the component README before using an experiment as a library.

## Verification

From the repository root, in a compatible environment:

```powershell
python -m unittest discover -s ippg/tests -p "test_*.py"
```

This command is documented from the existing project workflow. The documentation bootstrap does not constitute a new passing test run. Historical data-dependent experiments require their local input files.
