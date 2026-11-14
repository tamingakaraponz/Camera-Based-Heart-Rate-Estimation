"""Compare POS+DWT iPPG with full-workbook cPPG in time and frequency domains.

Examples (run from the SCiUS-Project folder):
    python ippg/compare_ippg_cppg.py 21
    python ippg/compare_ippg_cppg.py 21 --no-show --output outputs/21-ippg-cppg.png
    python ippg/compare_ippg_cppg.py DATA/temp/21.avi DATAcppg/21.xlsx

cPPG BPM is always calculated from the entire XLSX series:
resample -> Butterworth band-pass -> FFT -> strongest peak in 0.6-2.0 Hz.
Cropping/interpolation is used only for the waveform comparison.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import numpy as np

from benchmark_background_methods import analyze_contact_ppg
from compare_gr_pos_cppg import analyze_pair, resolve_input_paths
from compare_ppg import input_path, validate_output
from ippg_v2_refactored import (
    IPPGConfig,
    add_wavelet_arguments,
    calculate_spectrum,
    wavelet_config_from_arguments,
)


def normalized_fft(
    time_seconds: np.ndarray,
    signal: np.ndarray,
    fps: float,
    low_hz: float,
    high_hz: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return a peak-normalized FFT magnitude inside the heart-rate band."""

    frequencies, spectrum = calculate_spectrum(time_seconds, signal, fps)
    selected = (frequencies >= low_hz) & (frequencies <= high_hz)
    frequencies = frequencies[selected]
    magnitude = np.abs(spectrum[selected])
    if frequencies.size == 0 or magnitude.size == 0 or np.max(magnitude) <= 1e-12:
        raise ValueError("No usable FFT peak inside the heart-rate band.")
    return frequencies, magnitude / np.max(magnitude)


def analyze_ippg_cppg(video: Path, contact: Path, config: IPPGConfig) -> tuple[dict, dict]:
    """Run POS+DWT iPPG and full-series cPPG through their canonical paths."""

    comparison, metadata = analyze_pair(video, contact, config)
    contact_analysis = analyze_contact_ppg(
        contact,
        config.forehead_low_hz,
        config.forehead_high_hz,
        filter_order=3,
    )
    ippg_frequency, ippg_fft = normalized_fft(
        comparison.time_seconds,
        comparison.pos_dwt_z,
        metadata["video_fps"],
        config.forehead_low_hz,
        config.forehead_high_hz,
    )
    cppg_frequency = contact_analysis.frequencies_hz
    cppg_fft = np.abs(contact_analysis.spectrum)
    selected = (
        (cppg_frequency >= config.forehead_low_hz)
        & (cppg_frequency <= config.forehead_high_hz)
    )
    cppg_frequency = cppg_frequency[selected]
    cppg_fft = cppg_fft[selected]
    if cppg_fft.size == 0 or np.max(cppg_fft) <= 1e-12:
        raise ValueError("No usable cPPG FFT peak inside the heart-rate band.")
    cppg_fft = cppg_fft / np.max(cppg_fft)

    method_metrics = comparison.metrics()["methods"]["pos-dwt"]
    result = {
        "time_seconds": comparison.time_seconds,
        "ippg_z": comparison.pos_dwt_z,
        "cppg_z": comparison.cppg_z,
        "ippg_frequency_hz": ippg_frequency,
        "ippg_fft_normalized": ippg_fft,
        "cppg_frequency_hz": cppg_frequency,
        "cppg_fft_normalized": cppg_fft,
        "ippg_bpm": float(method_metrics["fft_bpm"]),
        "cppg_bpm": float(contact_analysis.heart_rate_bpm),
        "bpm_error": float(method_metrics["fft_bpm"] - contact_analysis.heart_rate_bpm),
        "absolute_bpm_error": float(abs(method_metrics["fft_bpm"] - contact_analysis.heart_rate_bpm)),
        "pearson_r": float(method_metrics["pearson_r"]),
        "mae_z": float(method_metrics["mae_z"]),
        "rmse_z": float(method_metrics["rmse_z"]),
    }
    metadata = {
        **metadata,
        "ippg_method": "POS + DWT + band-pass + FFT",
        "cppg_method": "Full XLSX + resample + band-pass + FFT",
        "cppg_fft_scope": "Entire XLSX workbook",
    }
    return result, metadata


def plot_ippg_cppg(result: dict, metadata: dict, zoom=None) -> Figure:
    """Draw waveform and FFT comparisons without changing computed metrics."""

    time = result["time_seconds"]
    left, right = (float(time[0]), float(time[-1])) if zoom is None else zoom
    if left < time[0] or right > time[-1] or left >= right:
        raise ValueError(f"Zoom must lie within {time[0]:.3f}-{time[-1]:.3f} seconds.")

    blue, orange = "#2878B5", "#D77824"
    figure, axes = plt.subplots(2, 1, figsize=(12, 8))
    figure.subplots_adjust(left=0.09, right=0.97, top=0.87, bottom=0.10, hspace=0.38)
    figure.suptitle("POS+DWT iPPG compared with cPPG", fontsize=17)
    figure.text(
        0.09,
        0.905,
        f"iPPG: {result['ippg_bpm']:.2f} BPM | cPPG: {result['cppg_bpm']:.2f} BPM | "
        f"error: {result['bpm_error']:+.2f} BPM",
        fontsize=11,
    )

    axes[0].plot(time, result["ippg_z"], color=blue, lw=1.5, label="iPPG POS+DWT")
    axes[0].plot(time, result["cppg_z"], color=orange, ls="--", lw=1.5, label="cPPG")
    axes[0].set_xlim(left, right)
    axes[0].set_title(
        f"Waveform | Pearson r={result['pearson_r']:+.3f} | "
        f"MAE={result['mae_z']:.3f} Z | RMSE={result['rmse_z']:.3f} Z",
        loc="left",
    )
    axes[0].set_xlabel("Time from assumed recording start (s)")
    axes[0].set_ylabel("Pulse (Z-score)")
    axes[0].legend(frameon=False)
    axes[0].grid(axis="y", alpha=0.25)

    axes[1].plot(
        result["ippg_frequency_hz"] * 60.0,
        result["ippg_fft_normalized"],
        color=blue,
        lw=1.8,
        label=f"iPPG peak {result['ippg_bpm']:.2f} BPM",
    )
    axes[1].plot(
        result["cppg_frequency_hz"] * 60.0,
        result["cppg_fft_normalized"],
        color=orange,
        ls="--",
        lw=1.8,
        label=f"cPPG peak {result['cppg_bpm']:.2f} BPM",
    )
    band = metadata["bandpass_hz"]
    axes[1].set_xlim(band[0] * 60.0, band[1] * 60.0)
    axes[1].set_ylim(0.0, 1.08)
    axes[1].set_title("FFT spectrum in the heart-rate band", loc="left")
    axes[1].set_xlabel("Heart rate (BPM)")
    axes[1].set_ylabel("Normalized magnitude")
    axes[1].legend(frameon=False)
    axes[1].grid(axis="y", alpha=0.25)

    for axis in axes:
        axis.spines[["top", "right"]].set_visible(False)
    figure.text(
        0.09,
        0.025,
        "cPPG BPM uses the full XLSX. Waveform alignment assumes the same recording start; no lag correction or sign flip.",
        fontsize=9,
    )
    return figure


def serializable_metrics(result: dict, metadata: dict, zoom) -> dict:
    """Remove plotting arrays and return an inspectable JSON summary."""

    keys = (
        "ippg_bpm", "cppg_bpm", "bpm_error", "absolute_bpm_error",
        "pearson_r", "mae_z", "rmse_z",
    )
    return {
        **metadata,
        **{key: result[key] for key in keys},
        "common_start_s": float(result["time_seconds"][0]),
        "common_end_s": float(result["time_seconds"][-1]),
        "common_samples": int(result["time_seconds"].size),
        "plot_zoom_s": zoom,
    }


def parse_arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_wavelet_arguments(parser, include_denoise=False)
    parser.add_argument("video", nargs="?", type=input_path, metavar="DATA_OR_VIDEO")
    parser.add_argument("cppg", nargs="?", type=input_path)
    parser.add_argument("--duration", type=float, help="Maximum video duration; default is the whole clip")
    parser.add_argument("--zoom", nargs=2, type=float, metavar=("START", "END"))
    parser.add_argument("--output", type=input_path, help="PNG output path")
    parser.add_argument("--metrics-json", type=input_path, help="JSON metrics output path")
    parser.add_argument("--no-show", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no-mediapipe", action="store_true")
    parser.add_argument("--landmark-model", type=input_path)
    parser.add_argument("--cascade", type=input_path)
    parser.add_argument("--preview", action="store_true")
    parser.add_argument("--roi-method", choices=("legacy", "quality"), default="legacy")
    parser.add_argument("--roi-trim-fraction", type=float, default=0.10)
    parser.add_argument("--roi-min-quality", type=float, default=0.05)
    arguments = parser.parse_args(argv)
    if arguments.duration is not None and (
        not np.isfinite(arguments.duration) or arguments.duration <= 0
    ):
        parser.error("--duration must be finite and positive")
    if arguments.zoom is not None and (
        not np.all(np.isfinite(arguments.zoom)) or arguments.zoom[0] >= arguments.zoom[1]
    ):
        parser.error("--zoom requires finite START < END")
    return arguments


def main() -> None:
    arguments = parse_arguments()
    try:
        video, contact = resolve_input_paths(arguments.video, arguments.cppg)
        missing = [str(path) for path in (video, contact) if not path.is_file()]
        if missing:
            raise FileNotFoundError("File(s) not found:\n" + "\n".join(missing))
        validate_output(arguments.output, ".png", [video, contact], arguments.overwrite)
        validate_output(arguments.metrics_json, ".json", [video, contact], arguments.overwrite)
        if arguments.no_show:
            plt.switch_backend("Agg")

        config = IPPGConfig(
            max_duration_seconds=arguments.duration,
            show_preview=arguments.preview,
            pulse_method="pos",
            background_method="none",
            roi_method=arguments.roi_method,
            roi_trim_fraction=arguments.roi_trim_fraction,
            roi_min_quality=arguments.roi_min_quality,
            prefer_mediapipe=not arguments.no_mediapipe,
            face_landmarker_model_path=arguments.landmark_model,
            face_cascade_path=arguments.cascade,
            **wavelet_config_from_arguments(arguments),
        )
        result, metadata = analyze_ippg_cppg(video, contact, replace(config, denoise="dwt"))
        figure = plot_ippg_cppg(result, metadata, arguments.zoom)
        metrics = serializable_metrics(result, metadata, arguments.zoom)

        print(f"Video: {video}")
        print(f"cPPG:  {contact}")
        print(f"iPPG POS+DWT: {result['ippg_bpm']:.2f} BPM")
        print(f"cPPG full XLSX: {result['cppg_bpm']:.2f} BPM")
        print(f"BPM error: {result['bpm_error']:+.2f} BPM")
        print(f"Pearson r: {result['pearson_r']:+.4f}")

        try:
            if arguments.output:
                arguments.output.parent.mkdir(parents=True, exist_ok=True)
                figure.savefig(arguments.output, dpi=160, facecolor="white")
                print(f"Saved graph: {arguments.output.resolve()}")
            if arguments.metrics_json:
                arguments.metrics_json.parent.mkdir(parents=True, exist_ok=True)
                arguments.metrics_json.write_text(
                    json.dumps(metrics, indent=2, allow_nan=False), encoding="utf-8"
                )
                print(f"Saved metrics: {arguments.metrics_json.resolve()}")
            if not arguments.no_show:
                plt.show()
        finally:
            plt.close(figure)
    except (ValueError, RuntimeError, OSError, EOFError) as error:
        raise SystemExit(f"Comparison failed: {error}") from error


if __name__ == "__main__":
    main()
