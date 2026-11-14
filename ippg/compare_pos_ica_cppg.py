"""Compare POS+DWT, ICA+DWT and POS+ICA fusion with contact PPG.

Examples:
    python "ippg/compare_pos_ica_cppg.py" 21
    python "ippg/compare_pos_ica_cppg.py" 21 --output fusion.png --metrics-json fusion.json
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
from compare_gr_pos_cppg import resolve_input_paths
from compare_ppg import input_path, validate_output, validate_signal, zscore
from ippg_v2_refactored import (
    ExtractedSignals,
    FusionResult,
    IPPGConfig,
    add_wavelet_arguments,
    calculate_pos_ica_fusion,
    calculate_spectrum,
    estimate_heart_rate,
    extract_video_signals,
    wavelet_config_from_arguments,
)


@dataclass(frozen=True)
class PosIcaComparison:
    time_seconds: np.ndarray
    cppg_z: np.ndarray
    pos_z: np.ndarray
    ica_z: np.ndarray
    time_fusion_z: np.ndarray
    bpm: dict[str, float]
    fusion: FusionResult

    def metrics(self) -> dict:
        methods: dict[str, dict] = {}
        for method, values in (
            ("pos-dwt", self.pos_z),
            ("ica-dwt", self.ica_z),
            ("pos-ica-time", self.time_fusion_z),
        ):
            difference = values - self.cppg_z
            methods[method] = {
                "pearson_r": float(np.corrcoef(values, self.cppg_z)[0, 1]),
                "mae_z": float(np.mean(np.abs(difference))),
                "rmse_z": float(np.sqrt(np.mean(difference**2))),
                "fft_bpm": self.bpm[method],
                "error_bpm": self.bpm[method] - self.bpm["cppg"],
                "absolute_error_bpm": abs(self.bpm[method] - self.bpm["cppg"]),
            }
        methods["pos-ica-spectral"] = {
            "waveform_metrics": None,
            "fft_bpm": self.bpm["pos-ica-spectral"],
            "error_bpm": self.bpm["pos-ica-spectral"] - self.bpm["cppg"],
            "absolute_error_bpm": abs(
                self.bpm["pos-ica-spectral"] - self.bpm["cppg"]
            ),
        }
        return {
            "common_start_s": float(self.time_seconds[0]),
            "common_end_s": float(self.time_seconds[-1]),
            "common_samples": int(self.time_seconds.size),
            "metric_scope": "Full common interval, independent of plot zoom",
            "cppg_fft_bpm": self.bpm["cppg"],
            "methods": methods,
            "fusion": self.fusion.metadata,
            "time_fusion_bpm_full_video": self.fusion.time_bpm,
            "spectral_fusion_bpm_full_video": self.fusion.spectral_bpm,
        }


def _normalized_power(
    frequencies_hz: np.ndarray,
    spectrum: np.ndarray,
    low_hz: float,
    high_hz: float,
) -> np.ndarray:
    power = np.abs(spectrum) ** 2
    band = (frequencies_hz >= low_hz) & (frequencies_hz <= high_hz)
    total = float(np.sum(power[band]))
    if not np.any(band) or total <= 1e-12:
        raise ValueError("No usable spectral power in the comparison heart band.")
    result = np.zeros_like(power)
    result[band] = power[band] / total
    return result


def prepare_comparison(
    signals: ExtractedSignals,
    contact_time: np.ndarray,
    contact_amplitude: np.ndarray,
    contact_fps: float,
    config: IPPGConfig,
) -> PosIcaComparison:
    """Build all video methods first, then compare on one common time grid."""

    fusion = calculate_pos_ica_fusion(signals, config)
    if fusion.pos_candidate is None or fusion.selected_ica is None:
        raise RuntimeError(
            "The three-panel comparator requires both usable POS and ICA candidates; "
            f"fusion status was {fusion.status}."
        )

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
    contact_grid = contact_analysis.time_seconds
    cppg = contact_analysis.filtered_amplitude
    common = (
        (signals.time_seconds >= contact_grid[0])
        & (signals.time_seconds <= contact_grid[-1])
    )
    time = signals.time_seconds[common]
    if time.size < 3:
        raise ValueError("Fewer than three common waveform samples.")

    common_pulses = {
        "cppg": np.interp(time, contact_grid, cppg),
        "pos-dwt": fusion.pos_candidate.processed.filtered[common],
        "ica-dwt": (
            fusion.ica_polarity * fusion.selected_ica.processed.filtered[common]
        ),
        "pos-ica-time": fusion.time_fused_signal[common],
    }
    bpm: dict[str, float] = {"cppg": contact_analysis.heart_rate_bpm}
    spectra: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for name, pulse in common_pulses.items():
        if name == "cppg":
            continue
        frequencies, spectrum = calculate_spectrum(time, pulse, signals.fps)
        spectra[name] = (frequencies, spectrum)
        bpm[name] = estimate_heart_rate(
            frequencies,
            spectrum,
            config.forehead_low_hz,
            config.forehead_high_hz,
        )

    pos_frequencies, pos_spectrum = spectra["pos-dwt"]
    ica_frequencies, ica_spectrum = spectra["ica-dwt"]
    if not np.array_equal(pos_frequencies, ica_frequencies):
        raise RuntimeError("POS and ICA comparison FFT grids do not match.")
    spectral_score = (
        fusion.weights["pos"]
        * _normalized_power(
            pos_frequencies,
            pos_spectrum,
            config.forehead_low_hz,
            config.forehead_high_hz,
        )
        + fusion.weights["ica"]
        * _normalized_power(
            ica_frequencies,
            ica_spectrum,
            config.forehead_low_hz,
            config.forehead_high_hz,
        )
    )
    bpm["pos-ica-spectral"] = estimate_heart_rate(
        pos_frequencies,
        spectral_score,
        config.forehead_low_hz,
        config.forehead_high_hz,
    )
    return PosIcaComparison(
        time_seconds=time,
        cppg_z=zscore(common_pulses["cppg"]),
        pos_z=zscore(common_pulses["pos-dwt"]),
        ica_z=zscore(common_pulses["ica-dwt"]),
        time_fusion_z=zscore(common_pulses["pos-ica-time"]),
        bpm=bpm,
        fusion=fusion,
    )


def analyze_pair(
    video: Path,
    contact: Path,
    config: IPPGConfig,
) -> tuple[PosIcaComparison, dict]:
    contact_time, amplitude, contact_fps = load_contact_ppg(contact)
    config = replace(config, pulse_method="pos-ica", background_method="none")
    signals = extract_video_signals(video, config)
    data = prepare_comparison(signals, contact_time, amplitude, contact_fps, config)
    metadata = {
        "video": str(video.resolve()),
        "contact": str(contact.resolve()),
        "video_fps": signals.fps,
        "contact_fps": contact_fps,
        "bandpass_hz": [config.forehead_low_hz, config.forehead_high_hz],
        "background_method": "none",
        "roi_method": config.roi_method,
        "normalization": "Each waveform Z-scored over the full common interval",
        "time_assumption": "Common recording start; synchronization unverified",
        "lag_correction": False,
        "cppg_used_by_runtime_fusion": False,
        "ica_polarity_aligned_to": "POS only",
        "cppg_bpm_scope": "Full XLSX series after resample and band-pass",
    }
    return data, metadata


def plot_comparison(
    data: PosIcaComparison,
    metadata: dict,
    zoom: tuple[float, float] | None = None,
) -> Figure:
    """Draw three waveform panels; spectral fusion is reported as BPM only."""

    time = data.time_seconds
    left, right = (float(time[0]), float(time[-1])) if zoom is None else zoom
    if left >= right or left < time[0] or right > time[-1]:
        raise ValueError(f"Zoom must lie within {time[0]:.3f}-{time[-1]:.3f} s.")
    visible = (time >= left) & (time <= right)
    if np.count_nonzero(visible) < 2:
        raise ValueError("Zoom needs at least two common samples.")
    pulses = (data.cppg_z, data.pos_z, data.ica_z, data.time_fusion_z)
    visible_values = np.concatenate([pulse[visible] for pulse in pulses])
    padding = max(0.2, float(np.ptp(visible_values)) * 0.08)
    metrics = data.metrics()["methods"]

    with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 11}):
        figure, axes = plt.subplots(3, 1, figsize=(12, 10.5), sharex=True, sharey=True)
        figure.subplots_adjust(left=0.08, right=0.97, top=0.81, bottom=0.14, hspace=0.44)
        figure.suptitle("POS / ICA / POS+ICA compared with cPPG", fontsize=17, y=0.98)
        figure.text(
            0.08,
            0.925,
            f"Time fusion FFT: {data.bpm['pos-ica-time']:.2f} BPM | "
            f"Spectral fusion FFT: {data.bpm['pos-ica-spectral']:.2f} BPM | "
            f"cPPG: {data.bpm['cppg']:.2f} BPM",
            fontsize=10.5,
        )
        figure.text(
            0.08,
            0.89,
            f"Weights POS/ICA: {data.fusion.weights['pos']:.3f}/{data.fusion.weights['ica']:.3f} | "
            f"ICA component: {data.fusion.selected_ica.component_index} | "
            f"status: {data.fusion.status}",
            fontsize=10,
        )
        rows = (
            ("pos-dwt", "POS + DWT", data.pos_z),
            ("ica-dwt", "ICA + DWT", data.ica_z),
            ("pos-ica-time", "POS + ICA time fusion", data.time_fusion_z),
        )
        for axis, (method, label, pulse) in zip(axes, rows):
            axis.plot(time, pulse, color="#2878B5", lw=1.5, label=f"iPPG ({label})")
            axis.plot(time, data.cppg_z, color="#D77824", ls="--", lw=1.5, label="cPPG")
            axis.set_title(
                f"{label} vs cPPG | r={metrics[method]['pearson_r']:+.3f} | "
                f"FFT={metrics[method]['fft_bpm']:.2f} BPM",
                loc="left",
            )
            axis.set_xlim(left, right)
            axis.set_ylim(visible_values.min() - padding, visible_values.max() + padding)
            axis.set_ylabel("Pulse (Z-score)")
            axis.set_xlabel("Time from assumed recording start (s)")
            axis.grid(axis="y", color="#E0E3E6", lw=0.6)
            axis.spines[["top", "right"]].set_visible(False)
            axis.legend(loc="upper right", ncol=2, frameon=False)
        figure.text(
            0.08,
            0.04,
            "No lag correction. ICA polarity is aligned to POS only; cPPG is never used by runtime fusion.",
            fontsize=9,
        )
    return figure


def parse_arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    add_wavelet_arguments(parser)
    parser.add_argument("video", nargs="?", type=input_path, metavar="DATA_OR_VIDEO")
    parser.add_argument("cppg", nargs="?", type=input_path)
    parser.add_argument("--zoom", type=float, nargs=2, metavar=("START", "END"))
    parser.add_argument("--duration", type=float, help="Maximum video duration; default is whole clip")
    parser.add_argument("--output", type=input_path)
    parser.add_argument("--metrics-json", type=input_path)
    parser.add_argument("--no-show", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no-mediapipe", action="store_true")
    parser.add_argument("--landmark-model", type=input_path)
    parser.add_argument("--cascade", type=input_path)
    parser.add_argument("--preview", action="store_true")
    parser.add_argument("--roi-method", choices=("quality", "legacy"), default="legacy")
    parser.add_argument("--roi-trim-fraction", type=float, default=0.10)
    parser.add_argument("--roi-min-quality", type=float, default=0.05)
    parser.add_argument("--fusion-min-correlation", type=float, default=0.2)
    parser.add_argument("--fusion-bpm-tolerance", type=float, default=6.0)
    parser.add_argument("--ica-random-state", type=int, default=0)
    args = parser.parse_args(argv)
    if args.zoom is not None and (
        not np.all(np.isfinite(args.zoom)) or args.zoom[0] >= args.zoom[1]
    ):
        parser.error("--zoom requires finite START < END")
    return args


def main() -> None:
    args = parse_arguments()
    try:
        video, contact = resolve_input_paths(args.video, args.cppg)
        missing = [str(path) for path in (video, contact) if not path.is_file()]
        if missing:
            raise FileNotFoundError("File(s) not found:\n" + "\n".join(missing))
        validate_output(args.output, ".png", [video, contact], args.overwrite)
        validate_output(args.metrics_json, ".json", [video, contact], args.overwrite)
        if args.no_show:
            plt.switch_backend("Agg")
        config = IPPGConfig(
            max_duration_seconds=args.duration,
            show_preview=args.preview,
            prefer_mediapipe=not args.no_mediapipe,
            face_landmarker_model_path=args.landmark_model,
            face_cascade_path=args.cascade,
            roi_method=args.roi_method,
            roi_trim_fraction=args.roi_trim_fraction,
            roi_min_quality=args.roi_min_quality,
            fusion_min_correlation=args.fusion_min_correlation,
            fusion_bpm_tolerance=args.fusion_bpm_tolerance,
            ica_random_state=args.ica_random_state,
            **wavelet_config_from_arguments(args),
        )
        data, metadata = analyze_pair(video, contact, config)
        figure = plot_comparison(data, metadata, args.zoom)
        metrics = {**metadata, **data.metrics(), "plot_zoom_s": args.zoom}
        for method, values in metrics["methods"].items():
            print(
                f"{method}: FFT={values['fft_bpm']:.2f} BPM, "
                f"error={values['error_bpm']:+.2f} BPM"
            )
        try:
            if args.output:
                args.output.parent.mkdir(parents=True, exist_ok=True)
                figure.savefig(args.output, dpi=160, facecolor="white")
                print(f"Saved: {args.output.resolve()}")
            if args.metrics_json:
                args.metrics_json.parent.mkdir(parents=True, exist_ok=True)
                args.metrics_json.write_text(
                    json.dumps(metrics, indent=2, allow_nan=False), encoding="utf-8"
                )
                print(f"Saved: {args.metrics_json.resolve()}")
            if not args.no_show:
                plt.show()
        finally:
            plt.close(figure)
    except (ValueError, RuntimeError, OSError, EOFError) as error:
        raise SystemExit(f"Comparison failed: {error}") from error


if __name__ == "__main__":
    main()
