r"""Read a UBFC-style ground_truth.txt and plot PPG -> band-pass -> FFT.

Run without arguments to use FILE_PATH below, or pass another file:
    python ippg/analyze_ground_truth.py "C:\path\to\ground_truth.txt"

Dependencies: numpy scipy matplotlib
Rows: 1 = PPG samples, 2 = supplied HR (BPM), 3 = timestamps (seconds).
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import sys

import numpy as np
from scipy.signal import butter, sosfiltfilt


# Change this path when running with the Run button in your editor.
FILE_PATH = Path(r"C:\Users\Akkarapon\Downloads\subject1-20261104T091024Z-1-001\subject1\ground_truth.txt")
LOWCUT_HZ = 0.6   # 36 BPM
HIGHCUT_HZ = 3.0  # 180 BPM; change with --high if needed


@dataclass
class Result:
    original_time: np.ndarray
    original_ppg: np.ndarray
    provided_hr: np.ndarray
    time: np.ndarray
    uniform_ppg: np.ndarray
    filtered_ppg: np.ndarray
    frequency: np.ndarray
    magnitude: np.ndarray
    fps: float
    duplicate_count: int
    largest_gap_s: float
    heart_rate_bpm: float
    peak_hz: float
    low_hz: float
    high_hz: float
    source_time: np.ndarray | None = None


def analyze_file(path: Path, low=LOWCUT_HZ, high=HIGHCUT_HZ, order=3, fs=None) -> Result:
    """Uses timestamps as stored; never applies a fitted HR multiplier."""
    data = np.loadtxt(path, dtype=float, ndmin=2)
    if data.shape[0] != 3 or data.shape[1] < 2:
        raise ValueError("Expected exactly 3 numeric rows: PPG, provided BPM, time in seconds.")
    if not np.all(np.isfinite(data)):
        raise ValueError("The file contains NaN or infinite values.")
    ppg, provided_hr, times = data
    times = times - times[0]
    if np.any(np.diff(times) < 0):
        raise ValueError("Timestamps go backwards; verify the file before analysis.")

    # Combine only samples with exactly equal stored timestamps.
    unique_time, inverse, counts = np.unique(times, return_inverse=True, return_counts=True)
    if len(unique_time) < 2:
        raise ValueError("At least two distinct timestamps are required.")
    unique_ppg = np.bincount(inverse, weights=ppg) / counts
    duplicate_count = len(times) - len(unique_time)
    inferred_fs = (len(unique_time) - 1) / (unique_time[-1] - unique_time[0])
    sampling_rate = float(inferred_fs if fs is None else fs)
    if not np.isfinite(sampling_rate) or sampling_rate <= 0:
        raise ValueError("Sampling rate must be finite and positive.")
    if not np.isfinite(low) or not np.isfinite(high) or not 0 < low < high < sampling_rate / 2:
        raise ValueError("Filter must satisfy 0 < low < high < sampling_rate/2.")
    if order < 1:
        raise ValueError("Filter order must be a positive integer.")

    if fs is None:
        uniform_time = np.linspace(unique_time[0], unique_time[-1], len(unique_time))
    else:
        size = int(np.floor((unique_time[-1] - unique_time[0]) * sampling_rate)) + 1
        uniform_time = unique_time[0] + np.arange(size) / sampling_rate
    # Linear interpolation fills irregular timing; it does not add measurements.
    uniform_ppg = np.interp(uniform_time, unique_time, unique_ppg)
    if np.ptp(uniform_ppg) <= 1e-12:
        raise ValueError("The PPG signal is flat; no usable pulse was found.")

    # 1) Butterworth band-pass, forward/backward for zero phase shift.
    sos = butter(order, [low, high], btype="bandpass", fs=sampling_rate, output="sos")
    padlen = 3 * (2 * len(sos) + 1)
    if len(uniform_ppg) <= padlen:
        raise ValueError(f"Signal too short: need more than {padlen} samples for this filter.")
    filtered = sosfiltfilt(sos, uniform_ppg - np.mean(uniform_ppg), padlen=padlen)

    # 2) FFT of the full filtered signal, no DWT, time fitting or HR multiplier.
    magnitude = np.abs(np.fft.rfft(filtered - np.mean(filtered)))
    frequency = np.fft.rfftfreq(len(filtered), d=1.0 / sampling_rate)
    valid = np.flatnonzero((frequency >= low) & (frequency <= high))
    if len(valid) == 0 or np.max(magnitude[valid]) <= 1e-12:
        raise ValueError("No usable FFT peak was found inside the selected band.")
    peak = int(valid[np.argmax(magnitude[valid])])
    peak_hz = float(frequency[peak])
    return Result(times, ppg, provided_hr, uniform_time, uniform_ppg, filtered,
                  frequency, magnitude, sampling_rate, duplicate_count,
                  float(np.max(np.diff(unique_time))), peak_hz * 60, peak_hz, low, high,
                  source_time=data[2].copy())


def plot_results(result: Result, title: str):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(3, 1, figsize=(12, 10), layout="constrained")
    fig.suptitle(f"{title}\nBand-pass + FFT estimate: {result.heart_rate_bpm:.2f} BPM", fontsize=14)
    axes[0].plot(result.original_time, result.original_ppg, color="#2878b5", lw=0.9)
    axes[0].set(title="1. Original PPG (first row of file)", xlabel="Stored relative time (s)", ylabel="Amplitude (a.u.)")
    axes[1].plot(result.time, result.filtered_ppg, color="#30905a", lw=0.9)
    axes[1].set(title=f"2. Band-pass filtered PPG ({result.low_hz:g}-{result.high_hz:g} Hz)", xlabel="Time (s)", ylabel="Amplitude (a.u.)")
    axes[2].plot(result.frequency, result.magnitude, color="#c77822")
    axes[2].axvspan(result.low_hz, result.high_hz, color="#c77822", alpha=0.1, label="HR search band")
    axes[2].axvline(result.peak_hz, color="black", ls="--", label=f"Peak: {result.peak_hz:.3f} Hz = {result.heart_rate_bpm:.2f} BPM")
    axes[2].set(title="3. FFT of the filtered PPG", xlabel="Frequency (Hz)", ylabel="FFT magnitude (a.u.)", xlim=(0, min(result.fps/2, result.high_hz+1)))
    axes[2].legend()
    for ax in axes:
        ax.grid(alpha=0.25)
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("file", nargs="?", type=Path, default=FILE_PATH)
    parser.add_argument("--low", type=float, default=LOWCUT_HZ, help="Lower band-pass cutoff in Hz")
    parser.add_argument("--high", type=float, default=HIGHCUT_HZ, help="Upper band-pass cutoff in Hz")
    parser.add_argument("--order", type=int, default=3, help="Butterworth order (default: 3)")
    parser.add_argument("--fs", type=float, help="Optional uniform sampling rate; default inferred from timestamp span")
    parser.add_argument("--output", type=Path, help="Optional saved graph (.png or .pdf)")
    parser.add_argument("--no-show", action="store_true", help="Do not open graph window")
    parser.add_argument("--output-root", type=Path, help="Optional run archive root; a new folder is created automatically")
    args = parser.parse_args()
    try:
        path = args.file.expanduser().resolve()
        if args.output:
            target = args.output.expanduser().resolve()
            if target == path or target.exists():
                raise ValueError(f"Refusing to overwrite an existing file: {target}")
            if target.suffix.lower() not in (".png", ".pdf"):
                raise ValueError("Graph output must be .png or .pdf.")
        result = analyze_file(path, args.low, args.high, args.order, args.fs)
        print(f"File: {path}")
        print(f"Original samples: {len(result.original_ppg)}")
        print(f"Duration: {result.original_time[-1]:.3f} s")
        print(f"Duplicate timestamps combined: {result.duplicate_count}")
        print(f"Uniform sampling rate: {result.fps:.4f} Hz")
        print(f"Largest stored timestamp gap: {result.largest_gap_s:.3f} s")
        print(f"Band-pass: {result.low_hz:g}-{result.high_hz:g} Hz, order {args.order}")
        print(f"FFT peak: {result.peak_hz:.4f} Hz")
        print(f"Estimated Heart Rate: {result.heart_rate_bpm:.2f} BPM")
        print(f"FFT bin spacing: {result.fps / len(result.filtered_ppg) * 60:.3f} BPM (not measurement accuracy)")
        print(f"Provided BPM row mean: {np.mean(result.provided_hr):.2f} BPM (not used in FFT)")
        if args.no_show:
            import matplotlib
            matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig = plot_results(result, path.name)
        from signal_recording import DEFAULT_OUTPUT_ROOT, save_reference_run
        folder = save_reference_run(result, path, fig, args.order, args.fs,
                                    args.output_root if args.output_root else DEFAULT_OUTPUT_ROOT)
        print(f"Saved results: {folder}")
        if args.output:
            target.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(target, dpi=150)
            print(f"Saved graph: {target}")
        if not args.no_show:
            plt.show()
        plt.close(fig)
    except (OSError, ValueError, RuntimeError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
