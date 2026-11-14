"""Show three pulse plots: G-R, baseline POS and POS+DWT versus cPPG.

Example (from SCiUS-Project):
    python "ippg/compare_gr_pos_cppg.py" 21
    python "ippg/compare_gr_pos_cppg.py" "DATA/temp/11.avi" "DATAcppg/11.xlsx"

Enter a DATA number to resolve DATA/temp/<number>.avi and DATAcppg/<number>.xlsx
relative to this script's project directory, independent of the working directory.
Omit arguments to enter a DATA number or video filepath interactively.
Keep this file next to compare_ppg.py,
ippg_v2_refactored.py and benchmark_background_methods.py. Contact XLSX:
first sheet, column A = amplitude, column B = timestamps in seconds.

RGB is extracted once, shared by all three methods. All use the same order-6
band-pass filter; cPPG uses the existing benchmark's order-3 filter. All use
0.6-2.0 Hz by default. Each pulse is Z-scored over the SAME full overlap.
There is no background correction, time-lag fitting, or automatic sign flip.

Chart contract: three stacked line panels, shared time and amplitude scales;
blue solid = video-derived pulse, orange dashed = the same cPPG in every panel.
Zoom changes only the visible interval, not normalization or full-clip metrics.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import numpy as np

from benchmark_background_methods import analyze_contact_samples, load_contact_ppg
from compare_ppg import input_path, validate_output, validate_signal, zscore
from ippg_v2_refactored import (
    ExtractedSignals, IPPGConfig, bandpass_filter, calculate_pos,
    extract_video_signals,
    process_pulse, ProcessedPulse, calculate_spectrum, estimate_heart_rate,
    add_wavelet_arguments, wavelet_config_from_arguments,
)


@dataclass(frozen=True)
class MethodComparison:
    time_seconds: np.ndarray
    cppg_z: np.ndarray
    gr_z: np.ndarray
    pos_z: np.ndarray
    pos_dwt_z: np.ndarray
    bpm: dict
    pos_dwt_processing: ProcessedPulse

    def metrics(self) -> dict:
        result = {
            "common_start_s": float(self.time_seconds[0]),
            "common_end_s": float(self.time_seconds[-1]),
            "common_samples": int(self.time_seconds.size),
            "metric_scope": "Full common interval, independent of plot zoom",
            "methods": {},
            "cppg_fft_bpm": self.bpm["cppg"],
            "fft_scope": "iPPG uses full common video interval; cPPG uses the full workbook",
            "denoise": self.pos_dwt_processing.denoise_metadata,
        }
        for method, values in (("g-r", self.gr_z), ("pos", self.pos_z), ("pos-dwt", self.pos_dwt_z)):
            difference = values - self.cppg_z
            result["methods"][method] = {
                "pearson_r": float(np.corrcoef(values, self.cppg_z)[0, 1]),
                "mae_z": float(np.mean(np.abs(difference))),
                "rmse_z": float(np.sqrt(np.mean(difference**2))),
                "fft_bpm": self.bpm[method],
                "error_bpm": self.bpm[method] - self.bpm["cppg"],
                "absolute_error_bpm": abs(self.bpm[method] - self.bpm["cppg"]),
            }
        return result


def prepare_comparison(
    signals: ExtractedSignals,
    contact_time: np.ndarray,
    contact_amplitude: np.ndarray,
    contact_fps: float,
    config: IPPGConfig,
) -> MethodComparison:
    """Process full signals before cropping, then normalize on one common grid."""

    contact_time = np.asarray(contact_time, dtype=float)
    contact_amplitude = np.asarray(contact_amplitude, dtype=float)
    validate_signal(contact_time, contact_amplitude, "cPPG")
    contact_analysis = analyze_contact_samples(
        contact_time,
        contact_amplitude,
        contact_fps,
        config.forehead_low_hz,
        config.forehead_high_hz,
        filter_order=3,
    )
    video_time = signals.time_seconds
    rgb = signals.combined_rgb
    if rgb.shape != (video_time.size, 3) or not np.all(np.isfinite(rgb)):
        raise ValueError("Combined RGB must be a finite (N, 3) array.")
    gr = rgb[:, 1] - rgb[:, 0]
    validate_signal(video_time, gr, "iPPG")
    # Video extraction normally already calculated POS. Do not run it again.
    pos = signals.pulse_signal if signals.pulse_method == "pos" else calculate_pos(
        rgb, signals.fps, config.pos_window_seconds, config.pos_epsilon
    )
    validate_signal(video_time, pos, "POS")
    ct = contact_analysis.time_seconds
    cppg = contact_analysis.filtered_amplitude
    common = (video_time >= ct[0]) & (video_time <= ct[-1])
    time = video_time[common]
    if time.size < 3:
        raise ValueError("Fewer than three common waveform samples.")

    baseline = replace(config, denoise="none")
    gr_processed = process_pulse(gr, signals.fps, "g-r", baseline)
    pos_processed = process_pulse(pos, signals.fps, "pos", baseline)
    dwt_processed = process_pulse(pos, signals.fps, "pos", replace(config, denoise="dwt"))
    common_pulses = {
        "cppg": np.interp(time, ct, cppg),
        "g-r": gr_processed.filtered[common],
        "pos": pos_processed.filtered[common],
        "pos-dwt": dwt_processed.filtered[common],
    }
    bpm = {"cppg": contact_analysis.heart_rate_bpm}
    for name, pulse in common_pulses.items():
        if name == "cppg":
            continue
        frequencies, spectrum = calculate_spectrum(time, pulse, signals.fps)
        bpm[name] = estimate_heart_rate(frequencies, spectrum, config.forehead_low_hz, config.forehead_high_hz)

    return MethodComparison(
        time_seconds=time,
        cppg_z=zscore(common_pulses["cppg"]),
        gr_z=zscore(common_pulses["g-r"]),
        pos_z=zscore(common_pulses["pos"]),
        pos_dwt_z=zscore(common_pulses["pos-dwt"]),
        bpm=bpm, pos_dwt_processing=dwt_processed,
    )


def analyze_pair(video: Path, contact: Path, config: IPPGConfig) -> tuple[MethodComparison, dict]:
    contact_time, amplitude, contact_fps = load_contact_ppg(contact)
    config = replace(config, pulse_method="pos", background_method="none")
    signals = extract_video_signals(video, config)
    data = prepare_comparison(signals, contact_time, amplitude, contact_fps, config)
    metadata = {
        "video": str(video.resolve()), "contact": str(contact.resolve()),
        "video_fps": signals.fps, "contact_fps": contact_fps,
        "bandpass_hz": [config.forehead_low_hz, config.forehead_high_hz],
        "filter_orders": {"g-r": config.forehead_filter_order,
                          "pos": config.forehead_filter_order,
                          "pos-dwt": config.forehead_filter_order, "cppg": 3},
        "background_method": "none",
        "roi_method": config.roi_method,
        "normalization": "Each pulse Z-scored over the full common interval",
        "time_assumption": "Common recording start; synchronization unverified",
        "lag_correction": False, "polarity_flip": False,
        "contact_interpolated_to_video_grid": True,
        "cppg_bpm_scope": "Full XLSX series after resample and band-pass",
    }
    return data, metadata


def plot_comparison(data: MethodComparison, metadata: dict, zoom=None) -> Figure:
    """Draw three panels; zoom never changes normalization, metrics or FFT."""

    time = data.time_seconds
    left, right = (float(time[0]), float(time[-1])) if zoom is None else zoom
    if (not np.all(np.isfinite([left, right])) or left >= right or
            left < time[0] or right > time[-1]):
        raise ValueError(f"Zoom must lie within {time[0]:.3f}-{time[-1]:.3f} s.")
    visible = (time >= left) & (time <= right)
    if np.count_nonzero(visible) < 2:
        raise ValueError("Zoom needs at least two common samples.")
    # Include interpolated boundary values so lines cannot clip at zoom edges.
    visible_values = np.concatenate([
        np.concatenate((pulse[visible], np.interp([left, right], time, pulse)))
        for pulse in (data.cppg_z, data.gr_z, data.pos_z, data.pos_dwt_z)
    ])
    padding = max(0.2, float(np.ptp(visible_values)) * 0.08)
    blue, orange, ink, grid = "#2878B5", "#D77824", "#24272B", "#E0E3E6"
    metrics = data.metrics()["methods"]
    with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 11,
                         "text.color": ink, "axes.labelcolor": ink}):
        figure, axes = plt.subplots(3, 1, figsize=(12, 10.5), sharex=True, sharey=True)
        figure.set_facecolor("white")
        figure.subplots_adjust(left=0.08, right=0.97, top=0.80, bottom=0.16, hspace=0.45)
        band = metadata["bandpass_hz"]
        figure.suptitle("G-R / POS / POS+DWT compared with cPPG", fontsize=17, y=0.98)
        figure.text(0.08, 0.925,
                    f"Band-pass {band[0]:g}-{band[1]:g} Hz | Z-score over full overlap | "
                    f"shown: {left:.2f}-{right:.2f} s", fontsize=11)
        figure.text(0.08, 0.89,
                    f"Full overlap: {time[0]:.2f}-{time[-1]:.2f} s; {time.size} samples | cPPG FFT: {data.bpm['cppg']:.2f} BPM", fontsize=10)
        for axis, (method, label, pulse) in zip(axes, (
                ("g-r", "G-R", data.gr_z), ("pos", "POS baseline", data.pos_z),
                ("pos-dwt", "POS + DWT", data.pos_dwt_z))):
            axis.plot(time, pulse, color=blue, lw=1.5, label=f"iPPG ({label})")
            axis.plot(time, data.cppg_z, color=orange, ls="--", lw=1.5, label="cPPG")
            axis.set_title(f"{label} vs cPPG | r = {metrics[method]['pearson_r']:+.3f} | FFT: {data.bpm[method]:.2f} BPM", loc="left")
            axis.set_xlim(left, right)
            axis.set_ylim(visible_values.min() - padding, visible_values.max() + padding)
            axis.set_ylabel("Pulse (Z-score)")
            axis.set_xlabel("Time from assumed recording start (s)")
            axis.tick_params(labelbottom=True)
            axis.grid(axis="y", color=grid, lw=0.6)
            axis.axhline(0, color=grid, lw=0.8)
            axis.spines[["top", "right"]].set_visible(False)
            axis.spines[["left", "bottom"]].set_color("#AEB3B8")
            axis.legend(loc="upper right", ncol=2, frameon=False)
        figure.text(0.08, 0.07,
                    "No lag correction or sign flip. Synchronization unverified; r is not an accuracy percentage.\n"
                    "cPPG interpolated to video grid. Filter orders: G-R/POS 6, cPPG 3.", fontsize=9)
        figure.text(0.08, 0.025,
                    f"Sources: {Path(metadata['video']).name[:50]} + {Path(metadata['contact']).name[:50]}", fontsize=9)
    return figure


def dataset_paths(dataset_id: str) -> tuple[Path, Path]:
    """Map a numeric recording ID to its project-local video/workbook pair."""

    dataset_id = dataset_id.strip()
    if not dataset_id.isascii() or not dataset_id.isdigit():
        raise ValueError("DATA number must contain only digits, for example 21 or 62.")
    project_root = Path(__file__).resolve().parent.parent
    return (project_root / "DATA" / "temp" / f"{dataset_id}.avi",
            project_root / "DATAcppg" / f"{dataset_id}.xlsx")


def resolve_input_paths(video: Path | None, contact: Path | None) -> tuple[Path, Path]:
    """Accept one DATA number or preserve the original two-filepath interface."""

    if video is None:
        video = input_path(input("Enter DATA number (e.g. 21) or video filepath: "))
    entered = str(video)
    if entered.isascii() and entered.isdigit():
        if contact is not None:
            raise ValueError("Use a DATA number alone, or provide both video and cPPG filepaths.")
        return dataset_paths(entered)
    if contact is None:
        contact = input_path(input("Enter cPPG .xlsx filepath: "))
    return video.resolve(), contact.resolve()


def parse_arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_wavelet_arguments(parser, include_denoise=False)
    parser.add_argument("video", nargs="?", type=input_path, metavar="DATA_OR_VIDEO",
                        help="DATA number (e.g. 21) or video filepath; prompted if omitted")
    parser.add_argument("cppg", nargs="?", type=input_path,
                        help="cPPG XLSX filepath; omit when using a DATA number")
    parser.add_argument("--zoom", type=float, nargs=2, metavar=("START", "END"),
                        help="Display this interval in all three graphs; metrics remain full-overlap")
    parser.add_argument("--duration", type=float, help="Maximum video duration; default is whole clip")
    parser.add_argument("--output", type=input_path, help="Optional PNG output")
    parser.add_argument("--metrics-json", type=input_path, help="Optional JSON metrics")
    parser.add_argument("--no-show", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no-mediapipe", action="store_true")
    parser.add_argument("--landmark-model", type=input_path)
    parser.add_argument("--cascade", type=input_path)
    parser.add_argument("--preview", action="store_true", help="Show face/skin preview during extraction")
    parser.add_argument("--roi-method", choices=("quality", "legacy"), default="legacy",
                        help="ROI aggregation: legacy pixel weighting (default) or experimental quality-selected patches")
    parser.add_argument("--roi-trim-fraction", type=float, default=0.10)
    parser.add_argument("--roi-min-quality", type=float, default=0.05)
    args = parser.parse_args(argv)
    if args.duration is not None and (not np.isfinite(args.duration) or args.duration <= 0):
        parser.error("--duration must be finite and positive")
    if args.zoom is not None and (not np.all(np.isfinite(args.zoom)) or args.zoom[0] >= args.zoom[1]):
        parser.error("--zoom requires finite START < END")
    return args


def main():
    args = parse_arguments()
    try:
        video, contact = resolve_input_paths(args.video, args.cppg)
        print(f"Video: {video}")
        print(f"cPPG:  {contact}")
        missing = [str(path) for path in (video, contact) if not path.is_file()]
        if missing:
            raise FileNotFoundError("File(s) not found:\n" + "\n".join(missing))
        validate_output(args.output, ".png", [video, contact], args.overwrite)
        validate_output(args.metrics_json, ".json", [video, contact], args.overwrite)
        if args.no_show:
            plt.switch_backend("Agg")
        config = IPPGConfig(
            max_duration_seconds=args.duration, show_preview=args.preview,
            roi_method=args.roi_method,
            roi_trim_fraction=args.roi_trim_fraction,
            roi_min_quality=args.roi_min_quality,
            **wavelet_config_from_arguments(args),
            prefer_mediapipe=not args.no_mediapipe,
            face_landmarker_model_path=args.landmark_model, face_cascade_path=args.cascade,
        )
        data, metadata = analyze_pair(video, contact, config)
        figure = plot_comparison(data, metadata, args.zoom)
        metrics = {**metadata, **data.metrics(), "plot_zoom_s": args.zoom}
        print(f"\nFull common interval: {data.time_seconds[0]:.3f}-{data.time_seconds[-1]:.3f} s")
        for method, values in metrics["methods"].items():
            print(f"{method.upper():>3} vs cPPG: r={values['pearson_r']:+.6f}, "
                  f"MAE={values['mae_z']:.6f} Z, RMSE={values['rmse_z']:.6f} Z, "
                  f"FFT={values['fft_bpm']:.2f} BPM, error={values['error_bpm']:+.2f} BPM")
        print(f"cPPG FFT: {metrics['cppg_fft_bpm']:.2f} BPM; DWT: {metrics['denoise']}")
        print("WARNING: Start times assumed equal; no lag correction or sign flip. Waveform Z errors are not BPM.")
        try:
            if args.output:
                args.output.parent.mkdir(parents=True, exist_ok=True)
                figure.savefig(args.output, dpi=160, facecolor="white")
                print(f"Saved: {args.output.resolve()}")
            if args.metrics_json:
                args.metrics_json.parent.mkdir(parents=True, exist_ok=True)
                args.metrics_json.write_text(json.dumps(metrics, indent=2, allow_nan=False), encoding="utf-8")
            if not args.no_show:
                plt.show()
        finally:
            plt.close(figure)
    except (ValueError, RuntimeError, OSError, EOFError) as error:
        raise SystemExit(f"Comparison failed: {error}") from error


if __name__ == "__main__":
    main()
