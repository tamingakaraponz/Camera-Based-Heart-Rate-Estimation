"""Plot iPPG versus contact PPG with five exact sample comparisons.

Run from the project directory::

    python "ippg/compare_ppg.py" "DATA/temp/11.avi" "DATAcppg/11.xlsx"

Omit both paths to enter them interactively. Keep this file beside
ippg_v2_refactored.py and benchmark_background_methods.py. The contact workbook
must have amplitude in column A and timestamps in seconds in column B of its
first sheet. Source files are read only; saving a PNG/JSON is optional.

Method: preserve the existing iPPG filter (order 6) and cPPG benchmark filter
(order 3), both 0.6-2.0 Hz by default. Interpolate cPPG onto the common video
grid and independently Z-score over the full overlap, then sample five times.
No automatic lag correction, polarity flip, or best-matching segment search.
Matching relative start times does NOT verify hardware synchronization.

Chart contract: full-overlap and zoomed line plots with blue solid iPPG and
orange dashed cPPG, followed by an exact five-row table. Z units, not BPM/%.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import textwrap

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import numpy as np

from benchmark_background_methods import analyze_contact_samples, load_contact_ppg
from ippg_v2_refactored import (
    IPPGConfig,
    process_pulse, add_wavelet_arguments, wavelet_config_from_arguments,
    extract_video_signals,
)


DEFAULT_SAMPLE_TIMES = (5.0, 5.2, 5.4, 5.6, 5.8)


@dataclass(frozen=True)
class PulseComparison:
    """Common-grid Z-scored waveforms and five interpolated sample pairs."""

    time_seconds: np.ndarray
    ippg_z: np.ndarray
    cppg_z: np.ndarray
    sample_times: np.ndarray
    sample_ippg_z: np.ndarray
    sample_cppg_z: np.ndarray
    pearson_r: float
    sample_mae_z: float

    def metrics(self) -> dict:
        return {
            "common_start_s": float(self.time_seconds[0]),
            "common_end_s": float(self.time_seconds[-1]),
            "common_samples": int(self.time_seconds.size),
            "full_overlap_pearson_r": self.pearson_r,
            "five_sample_mae_z": self.sample_mae_z,
            "samples": [
                {"time_s": float(t), "ippg_z": float(i), "cppg_z": float(c),
                 "absolute_difference_z": float(abs(i - c))}
                for t, i, c in zip(
                    self.sample_times, self.sample_ippg_z, self.sample_cppg_z
                )
            ],
        }


def zscore(values: np.ndarray) -> np.ndarray:
    """Standardize using population SD; reject undefined/invalid signals."""

    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or values.size < 2 or not np.all(np.isfinite(values)):
        raise ValueError("A pulse must contain at least two finite 1-D samples.")
    deviation = float(np.std(values))
    if deviation <= 1e-12:
        raise ValueError("Cannot Z-score a constant pulse.")
    return (values - np.mean(values)) / deviation


def validate_signal(time: np.ndarray, pulse: np.ndarray, name: str) -> None:
    if time.ndim != 1 or pulse.shape != time.shape or time.size < 2:
        raise ValueError(f"{name}: time and pulse need matching 1-D arrays.")
    if not np.all(np.isfinite(time)) or not np.all(np.isfinite(pulse)):
        raise ValueError(f"{name}: time and pulse must be finite.")
    if np.any(np.diff(time) <= 0):
        raise ValueError(f"{name}: timestamps must be strictly increasing.")


def compare_filtered_signals(
    ippg_time: np.ndarray,
    ippg_pulse: np.ndarray,
    cppg_time: np.ndarray,
    cppg_pulse: np.ndarray,
    sample_times=DEFAULT_SAMPLE_TIMES,
) -> PulseComparison:
    """Compare existing filtered pulses without changing lag or polarity.

    Only the overlapping interval is retained. Interpolation does not add
    independent cPPG measurements; all five requested times must lie inside it.
    """

    it, ip, ct, cp = [np.asarray(v, dtype=float) for v in
                      (ippg_time, ippg_pulse, cppg_time, cppg_pulse)]
    validate_signal(it, ip, "iPPG")
    validate_signal(ct, cp, "cPPG")
    samples = np.asarray(sample_times, dtype=float)
    if samples.shape != (5,) or not np.all(np.isfinite(samples)):
        raise ValueError("Provide exactly five finite sample times.")
    if np.any(np.diff(samples) <= 0):
        raise ValueError("Sample times must be strictly increasing.")

    common = (it >= ct[0]) & (it <= ct[-1])
    time = it[common]
    if time.size < 3:
        raise ValueError("The signals have fewer than three common samples.")
    if samples[0] < time[0] or samples[-1] > time[-1]:
        raise ValueError(
            f"All sample times must be within {time[0]:.3f}-{time[-1]:.3f} s. "
            "Use --sample-times to choose five valid times after calibration."
        )

    ippg_z = zscore(ip[common])
    cppg_z = zscore(np.interp(time, ct, cp))
    sample_i = np.interp(samples, time, ippg_z)
    sample_c = np.interp(samples, time, cppg_z)
    return PulseComparison(
        time, ippg_z, cppg_z, samples, sample_i, sample_c,
        float(np.corrcoef(ippg_z, cppg_z)[0, 1]),
        float(np.mean(np.abs(sample_i - sample_c))),
    )


def analyze_pair(
    video_path: Path,
    contact_path: Path,
    config: IPPGConfig,
    sample_times=DEFAULT_SAMPLE_TIMES,
) -> tuple[PulseComparison, dict]:
    """Extract video pulse and read/filter the contact waveform once."""

    # Validate the workbook before spending time extracting the video.
    contact_time, amplitude, contact_fps = load_contact_ppg(contact_path)
    contact_analysis = analyze_contact_samples(
        contact_time,
        amplitude,
        contact_fps,
        config.forehead_low_hz,
        config.forehead_high_hz,
        filter_order=3,
    )
    signals = extract_video_signals(video_path, config)
    processed = process_pulse(signals.pulse_signal, signals.fps, signals.pulse_method, config)
    ippg_filtered = processed.filtered
    comparison = compare_filtered_signals(
        signals.time_seconds,
        ippg_filtered,
        contact_analysis.time_seconds,
        contact_analysis.filtered_amplitude,
        sample_times,
    )
    metadata = {
        "video": str(video_path.resolve()),
        "contact": str(contact_path.resolve()),
        "pulse_method": signals.pulse_method,
        "denoise": processed.denoise_metadata,
        "background_method": "none",
        "video_fps": signals.fps,
        "contact_fps": contact_fps,
        "cppg_fft_bpm": contact_analysis.heart_rate_bpm,
        "cppg_bpm_scope": "Full XLSX series after resample and band-pass",
        "bandpass_hz": [config.forehead_low_hz, config.forehead_high_hz],
        "filter_orders": {"ippg": config.forehead_filter_order, "cppg": 3},
        "normalization": "Each waveform Z-scored over the full common interval",
        "time_assumption": "Common recording start; synchronization unverified",
        "lag_correction": False,
        "polarity_flip": False,
        "contact_format": "First XLSX sheet: A=amplitude, B=timestamp in seconds",
    }
    return comparison, metadata


def plot_comparison(
    comparison: PulseComparison, metadata: dict, zoom: tuple[float, float] | None = None,
) -> Figure:
    """Build the overview, zoom, and table without saving or showing them."""

    data = comparison
    t = data.time_seconds
    if zoom is None:
        zoom = (max(float(t[0]), float(data.sample_times[0]) - 0.2),
                min(float(t[-1]), float(data.sample_times[-1]) + 0.4))
    left, right = zoom
    if (not np.all(np.isfinite(zoom)) or left >= right or left < t[0] or right > t[-1]
            or left > data.sample_times[0] or right < data.sample_times[-1]):
        raise ValueError("Zoom must be inside the common interval and contain all five sample times.")

    blue, orange, ink, grid = "#2878B5", "#D77824", "#24272B", "#E0E3E6"
    with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 11,
                         "text.color": ink, "axes.labelcolor": ink}):
        figure = plt.figure(figsize=(11, 9.6), facecolor="white", layout="constrained")
        gs = figure.add_gridspec(3, 1, height_ratios=[1, 1.35, 0.95])
        overview, detail, table_axis = [figure.add_subplot(gs[i]) for i in range(3)]
        band = metadata["bandpass_hz"]
        figure.suptitle(
            f"iPPG vs contact PPG | {Path(metadata['video']).stem[:45]}\n"
            f"{metadata['pulse_method'].upper()} + {metadata.get('denoise', {}).get('method', 'none').upper()}; band-pass {band[0]:g}-{band[1]:g} Hz; Z-scored",
            fontsize=16,
        )
        for axis in (overview, detail):
            axis.plot(t, data.ippg_z, color=blue, lw=1.6, label="iPPG")
            axis.plot(t, data.cppg_z, color=orange, lw=1.6, ls="--", label="cPPG")
            axis.axhline(0, color=grid, lw=0.8)
            axis.grid(axis="y", color=grid, linewidth=0.6)
            axis.set_ylabel("Pulse (Z-score)")
            axis.set_xlabel("Time from assumed recording start (s)")
            axis.spines[["top", "right"]].set_visible(False)
            axis.spines[["left", "bottom"]].set_color("#AEB3B8")
        overview.set_xlim(t[0], t[-1])
        overview.set_title(
            f"Full overlap: {t[0]:.2f}-{t[-1]:.2f} s | Pearson r = {data.pearson_r:+.3f}", loc="left"
        )
        overview.legend(loc="upper right", ncol=2, frameon=False)
        overview.axvspan(left, right, color=grid, alpha=0.4)

        detail.set_xlim(left, right)
        selected = (t >= left) & (t <= right)
        local_values = np.concatenate((data.ippg_z[selected], data.cppg_z[selected],
                                       data.sample_ippg_z, data.sample_cppg_z))
        padding = max(0.3, float(np.ptp(local_values)) * 0.12)
        detail.set_ylim(local_values.min() - padding, local_values.max() + padding)
        detail.set_title("Five preselected timestamps (zoom)", loc="left")
        detail.scatter(data.sample_times, data.sample_ippg_z, color=blue, marker="o", s=45, zorder=4)
        detail.scatter(data.sample_times, data.sample_cppg_z, color=orange, marker="s", s=45, zorder=4)
        for number, (time, i, c) in enumerate(zip(
                data.sample_times, data.sample_ippg_z, data.sample_cppg_z), 1):
            detail.plot([time, time], [i, c], color="#777777", lw=1, zorder=1)
            detail.annotate(str(number), (time, max(i, c)), xytext=(0, 10),
                            textcoords="offset points", ha="center")

        table_axis.axis("off")
        rows = [[str(n), f"{s:.3f}", f"{i:+.3f}", f"{c:+.3f}", f"{abs(i-c):.3f}"]
                for n, (s, i, c) in enumerate(zip(
                    data.sample_times, data.sample_ippg_z, data.sample_cppg_z), 1)]
        table = table_axis.table(
            cellText=rows, colLabels=["Point", "Time (s)", "iPPG (Z)", "cPPG (Z)", "Absolute gap (Z)"],
            cellLoc="center", bbox=[0, 0.43, 1, 0.54],
        )
        table.auto_set_font_size(False)
        table.set_fontsize(11)
        for (row, col), cell in table.get_celld().items():
            cell.set_edgecolor(grid)
            cell.set_linewidth(0.5)
            if row == 0:
                cell.set_facecolor("#F1F3F5")
            elif col in (2, 3):
                cell.get_text().set_color(blue if col == 2 else orange)
        table_axis.text(0, 0.30, f"Five-point mean absolute gap = {data.sample_mae_z:.3f} Z units (not BPM or %).",
                        transform=table_axis.transAxes)
        names = f"Sources: {Path(metadata['video']).name} + {Path(metadata['contact']).name}"
        notes = ("No lag correction or sign flip. Synchronization unverified; five points are illustrative only.\n"
                 "cPPG interpolated to video grid. Filter orders: iPPG 6, cPPG 3.\n" +
                 textwrap.fill(names, width=110))
        table_axis.text(0, 0.20, notes, fontsize=9, transform=table_axis.transAxes, va="top")
    return figure


def print_comparison(data: PulseComparison) -> None:
    print(f"\nCommon interval: {data.time_seconds[0]:.3f}-{data.time_seconds[-1]:.3f} s")
    print(f"Pearson r (full overlap, no lag correction): {data.pearson_r:+.6f}")
    print(" Point   Time(s)      iPPG(Z)      cPPG(Z)      Absolute gap(Z)")
    for n, row in enumerate(data.metrics()["samples"], 1):
        print(f" {n:>3}     {row['time_s']:>7.3f}   {row['ippg_z']:>+10.3f}   "
              f"{row['cppg_z']:>+10.3f}     {row['absolute_difference_z']:>10.3f}")
    print(f"Five-point MAE: {data.sample_mae_z:.6f} Z units (not BPM or %)")
    print("WARNING: Common recording start is assumed, not verified. No lag correction or sign flip.")


def input_path(value: str) -> Path:
    return Path(value.strip().strip('"\'')).expanduser()


def parse_arguments(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_wavelet_arguments(parser)
    parser.add_argument("video", nargs="?", type=input_path, help="Input video path (prompted if omitted)")
    parser.add_argument("cppg", nargs="?", type=input_path, help="cPPG XLSX path (prompted if omitted)")
    parser.add_argument("--sample-times", type=float, nargs=5, default=DEFAULT_SAMPLE_TIMES,
                        metavar="SECONDS", help="Five increasing times; default: 5.0 5.2 5.4 5.6 5.8")
    parser.add_argument("--zoom", type=float, nargs=2, metavar=("START", "END"),
                        help="Zoom interval containing all five samples; otherwise calculated automatically")
    parser.add_argument("--duration", type=float, help="Maximum video duration in seconds (default: whole video)")
    parser.add_argument("--pulse-method", choices=("pos", "g-r"), default="pos")
    parser.add_argument("--no-mediapipe", action="store_true", help="Use Haar instead of MediaPipe")
    parser.add_argument("--landmark-model", type=input_path)
    parser.add_argument("--cascade", type=input_path)
    parser.add_argument("--preview", action="store_true", help="Also show face/skin extraction preview")
    parser.add_argument("--output", type=input_path, help="Optional PNG path")
    parser.add_argument("--metrics-json", type=input_path, help="Optional JSON path for metrics and sample values")
    parser.add_argument("--no-show", action="store_true", help="Do not open the Matplotlib window")
    parser.add_argument("--overwrite", action="store_true", help="Allow replacing existing output PNG/JSON")
    args = parser.parse_args(argv)
    times = np.asarray(args.sample_times)
    if not np.all(np.isfinite(times)) or np.any(np.diff(times) <= 0):
        parser.error("--sample-times must be five finite, strictly increasing times")
    if args.duration is not None and (not np.isfinite(args.duration) or args.duration <= 0):
        parser.error("--duration must be finite and positive")
    if args.zoom is not None and (not np.all(np.isfinite(args.zoom)) or args.zoom[0] >= args.zoom[1]):
        parser.error("--zoom requires finite START < END")
    return args


def validate_output(path: Path | None, suffix: str, inputs: list[Path], overwrite: bool) -> None:
    if path is None:
        return
    if path.suffix.lower() != suffix:
        raise ValueError(f"Output must have a {suffix} extension: {path}")
    if path.resolve() in [source.resolve() for source in inputs]:
        raise ValueError("An output path cannot replace an input file.")
    if path.exists() and not overwrite:
        raise FileExistsError(f"Output already exists: {path}. Use --overwrite to replace it.")


def main() -> None:
    args = parse_arguments()
    try:
        video = args.video or input_path(input("Enter video filepath: "))
        contact = args.cppg or input_path(input("Enter cPPG .xlsx filepath: "))
        for path in (video, contact):
            if not path.is_file():
                raise FileNotFoundError(f"File not found: {path}")
        validate_output(args.output, ".png", [video, contact], args.overwrite)
        validate_output(args.metrics_json, ".json", [video, contact], args.overwrite)
        if args.no_show:
            plt.switch_backend("Agg")
        config = IPPGConfig(
            max_duration_seconds=args.duration, pulse_method=args.pulse_method,
            **wavelet_config_from_arguments(args),
            background_method="none", prefer_mediapipe=not args.no_mediapipe,
            face_landmarker_model_path=args.landmark_model, face_cascade_path=args.cascade,
            show_preview=args.preview,
        )
        data, metadata = analyze_pair(video, contact, config, args.sample_times)
        figure = plot_comparison(data, metadata, args.zoom)
        print_comparison(data)
        try:
            if args.output is not None:
                args.output.parent.mkdir(parents=True, exist_ok=True)
                figure.savefig(args.output, dpi=160, facecolor="white")
                print(f"Saved image: {args.output.resolve()}")
            if args.metrics_json is not None:
                args.metrics_json.parent.mkdir(parents=True, exist_ok=True)
                args.metrics_json.write_text(
                    json.dumps({**metadata, **data.metrics()}, indent=2, allow_nan=False), encoding="utf-8"
                )
                print(f"Saved metrics: {args.metrics_json.resolve()}")
            if not args.no_show:
                plt.show()
        finally:
            plt.close(figure)
    except (ValueError, RuntimeError, OSError, EOFError) as error:
        raise SystemExit(f"Comparison failed: {error}") from error


if __name__ == "__main__":
    main()
