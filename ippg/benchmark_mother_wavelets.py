"""Benchmark four DWT mother wavelets on the same iPPG/cPPG recordings.

Each video is decoded and its RGB signal is extracted once.  The identical
RGB/POS signal is then processed by db4, sym4, sym8 and bior3.5, so a result
cannot be caused by a different ROI, face track, skin mask, or cPPG window.

Example (from the project root)
-------------------------------
python "ippg/benchmark_mother_wavelets.py" --output outputs/mother-wavelets

The default datasets are 11, 12, 13, 21 and 62.  Results are saved as JSON,
CSV and a two-panel PNG.  The output directory must not already exist.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import replace
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from benchmark_background_methods import load_contact_ppg
from compare_gr_pos_cppg import dataset_paths, prepare_comparison
from ippg_v2_refactored import (
    IPPGConfig,
    extract_video_signals,
    parse_wavelet_level,
)


DEFAULT_DATASETS = ("11", "12", "13", "21", "62")
DEFAULT_WAVELETS = ("db4", "sym4", "sym8", "bior3.5")
WAVELET_COLORS = {
    "db4": "#2878B5",
    "sym4": "#D77824",
    "sym8": "#3A9D5D",
    "bior3.5": "#8A55A2",
}


def parse_arguments(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "ids", nargs="*", default=list(DEFAULT_DATASETS),
        help="DATA numbers to test (default: 11 12 13 21 62)",
    )
    parser.add_argument(
        "--wavelets", nargs="+", default=list(DEFAULT_WAVELETS),
        help="Wavelets to compare (default: db4 sym4 sym8 bior3.5)",
    )
    parser.add_argument("--output", type=Path, required=True,
                        help="New directory for JSON, CSV and PNG results")
    parser.add_argument("--duration", type=float, default=None,
                        help="Optional maximum video duration in seconds")
    parser.add_argument("--wavelet-level", type=parse_wavelet_level, default="auto",
                        help="auto=min(4, supported level), or a positive integer")
    parser.add_argument("--wavelet-strength", type=float, default=1.0,
                        help="Non-negative soft-threshold multiplier (default: 1.0)")
    parser.add_argument(
        "--wavelet-strategy", choices=("protect-pulse-band", "all-details"),
        default="protect-pulse-band",
        help="Protect 0.6-2.0 Hz coefficients (default) or threshold all details",
    )
    parser.add_argument("--no-mediapipe", action="store_true",
                        help="Use the fallback face detector instead of MediaPipe")
    arguments = parser.parse_args(argv)
    if not arguments.ids:
        parser.error("Supply at least one DATA number.")
    if len(set(arguments.ids)) != len(arguments.ids):
        parser.error("DATA numbers must not be repeated.")
    if len(set(arguments.wavelets)) != len(arguments.wavelets):
        parser.error("Wavelet names must not be repeated.")
    if arguments.duration is not None and (
        not np.isfinite(arguments.duration) or arguments.duration <= 0
    ):
        parser.error("--duration must be finite and positive.")
    if not np.isfinite(arguments.wavelet_strength) or arguments.wavelet_strength < 0:
        parser.error("--wavelet-strength must be finite and non-negative.")
    return arguments


def record_from_comparison(
    dataset_id: str,
    wavelet: str,
    comparison: Any,
    source_metadata: dict[str, Any],
) -> dict[str, Any]:
    """Flatten exactly the POS+DWT vs cPPG metrics needed for comparison."""

    metrics = comparison.metrics()
    method = metrics["methods"]["pos-dwt"]
    denoise = metrics["denoise"]
    coefficient_actions = "; ".join(
        f"{item['coefficient']}:{item['action']}"
        for item in denoise.get("coefficients", [])
    )
    return {
        "dataset_id": dataset_id,
        "wavelet": wavelet,
        "video_fps": source_metadata["video_fps"],
        "contact_fps": source_metadata["contact_fps"],
        "common_samples": metrics["common_samples"],
        "common_start_s": metrics["common_start_s"],
        "common_end_s": metrics["common_end_s"],
        "cppg_fft_bpm": metrics["cppg_fft_bpm"],
        "ippg_fft_bpm": method["fft_bpm"],
        "error_bpm": method["error_bpm"],
        "absolute_error_bpm": method["absolute_error_bpm"],
        "pearson_r": method["pearson_r"],
        "waveform_mae_z": method["mae_z"],
        "waveform_rmse_z": method["rmse_z"],
        "wavelet_level": denoise["actual_level"],
        "maximum_supported_level": denoise["maximum_level"],
        "threshold": denoise["threshold"],
        "noise_sigma": denoise["sigma"],
        "coefficient_actions": coefficient_actions,
    }


def validate_records(
    records: list[dict[str, Any]], ids: list[str], wavelets: list[str],
) -> None:
    """Fail loudly when a partial or non-comparable experiment is produced."""

    if len(records) != len(ids) * len(wavelets):
        raise RuntimeError("Wavelet benchmark is incomplete; no partial summary will be written.")
    numeric_columns = (
        "video_fps", "contact_fps", "common_samples", "common_start_s", "common_end_s",
        "cppg_fft_bpm", "ippg_fft_bpm", "error_bpm", "absolute_error_bpm", "pearson_r",
        "waveform_mae_z", "waveform_rmse_z", "wavelet_level", "threshold", "noise_sigma",
    )
    for row in records:
        if not np.all(np.isfinite([row[column] for column in numeric_columns])):
            raise RuntimeError(f"Non-finite metric recorded for DATA {row['dataset_id']}, {row['wavelet']}.")
    for dataset_id in ids:
        subset = [row for row in records if row["dataset_id"] == dataset_id]
        if {row["wavelet"] for row in subset} != set(wavelets):
            raise RuntimeError(f"DATA {dataset_id} does not contain all requested wavelets.")
        reference = subset[0]
        invariant_columns = (
            "video_fps", "contact_fps", "common_samples", "common_start_s", "common_end_s",
            "cppg_fft_bpm",
        )
        for row in subset[1:]:
            if any(not np.isclose(row[column], reference[column]) for column in invariant_columns):
                raise RuntimeError(
                    f"DATA {dataset_id} changed an input/grid invariant between wavelets."
                )


def aggregate_records(records: list[dict[str, Any]], wavelets: list[str]) -> list[dict[str, Any]]:
    """Aggregate clips equally; this is deliberately not sample-count weighted."""

    output: list[dict[str, Any]] = []
    for wavelet in wavelets:
        subset = [row for row in records if row["wavelet"] == wavelet]
        errors = np.asarray([row["error_bpm"] for row in subset], dtype=float)
        output.append({
            "wavelet": wavelet,
            "clips": len(subset),
            "bpm_mae": float(np.mean(np.abs(errors))),
            "bpm_rmse": float(np.sqrt(np.mean(errors ** 2))),
            "mean_pearson_r": float(np.mean([row["pearson_r"] for row in subset])),
            "mean_waveform_mae_z": float(np.mean([row["waveform_mae_z"] for row in subset])),
            "mean_waveform_rmse_z": float(np.mean([row["waveform_rmse_z"] for row in subset])),
        })
    return output


def plot_summary(
    records: list[dict[str, Any]], ids: list[str], wavelets: list[str], output_path: Path,
) -> None:
    """Create a same-scale grouped comparison for HR and waveform error."""

    x_values = np.arange(len(ids), dtype=float)
    width = 0.78 / len(wavelets)
    figure, axes = plt.subplots(2, 1, figsize=(12, 9), sharex=True)
    figure.set_facecolor("white")
    labels = {
        "absolute_error_bpm": "Absolute FFT BPM error (BPM)",
        "waveform_mae_z": "Waveform MAE (Z-score)",
    }
    for axis, metric in zip(axes, labels):
        for index, wavelet in enumerate(wavelets):
            values = [next(
                row[metric] for row in records
                if row["dataset_id"] == dataset_id and row["wavelet"] == wavelet
            ) for dataset_id in ids]
            position = x_values - 0.39 + width / 2 + index * width
            bars = axis.bar(
                position, values, width=width, color=WAVELET_COLORS.get(wavelet),
                label=wavelet,
            )
            axis.bar_label(bars, labels=[f"{value:.2f}" for value in values],
                           padding=2, fontsize=8, rotation=90)
        axis.set_ylabel(labels[metric])
        axis.set_ylim(bottom=0)
        axis.grid(axis="y", color="#E0E3E6", linewidth=0.8)
        axis.spines[["top", "right"]].set_visible(False)
    axes[0].set_title("Mother-wavelet comparison on the same five recordings", loc="left", fontsize=15)
    axes[0].legend(title="Mother wavelet", ncols=4, frameon=False)
    axes[1].set_xticks(x_values, [f"DATA {dataset_id}" for dataset_id in ids])
    axes[1].set_xlabel("Recording")
    figure.text(
        0.10, 0.01,
        "POS + DWT/soft-threshold/IDWT; 0.6-2.0 Hz coefficients protected; "
        "same RGB and cPPG per DATA. Lower is better. No time alignment or polarity flip.",
        fontsize=9,
    )
    figure.tight_layout(rect=(0, 0.035, 1, 1))
    figure.savefig(output_path, dpi=180, facecolor="white")
    plt.close(figure)


def main(argv: list[str] | None = None) -> None:
    arguments = parse_arguments(argv)
    arguments.output.mkdir(parents=True, exist_ok=False)
    config = IPPGConfig(
        max_duration_seconds=arguments.duration,
        show_preview=False,
        pulse_method="pos",
        background_method="none",
        prefer_mediapipe=not arguments.no_mediapipe,
        denoise="dwt",
        wavelet_level=arguments.wavelet_level,
        wavelet_strength=arguments.wavelet_strength,
        wavelet_strategy=arguments.wavelet_strategy,
    )
    records: list[dict[str, Any]] = []
    run_metadata: dict[str, Any] = {
        "datasets": arguments.ids,
        "wavelets": arguments.wavelets,
        "pipeline": "RGB -> POS -> DWT -> soft threshold -> IDWT -> band-pass -> FFT",
        "controls": {
            "background_method": "none",
            "bandpass_hz": [config.forehead_low_hz, config.forehead_high_hz],
            "wavelet_level": arguments.wavelet_level,
            "wavelet_strength": arguments.wavelet_strength,
            "wavelet_strategy": arguments.wavelet_strategy,
            "time_lag_correction": False,
            "polarity_flip": False,
            "aggregation": "unweighted mean across recordings",
        },
        "source_files": {},
    }

    for dataset_id in arguments.ids:
        video_path, contact_path = dataset_paths(dataset_id)
        if not video_path.is_file() or not contact_path.is_file():
            missing = [str(path) for path in (video_path, contact_path) if not path.is_file()]
            raise FileNotFoundError("Input file(s) not found:\n" + "\n".join(missing))
        print(f"DATA {dataset_id}: extracting RGB once", flush=True)
        contact_time, contact_amplitude, contact_fps = load_contact_ppg(contact_path)
        signals = extract_video_signals(video_path, config)
        source_metadata = {
            "video": str(video_path.resolve()),
            "contact": str(contact_path.resolve()),
            "video_fps": float(signals.fps),
            "contact_fps": float(contact_fps),
        }
        run_metadata["source_files"][dataset_id] = source_metadata
        for wavelet in arguments.wavelets:
            print(f"  {wavelet}", flush=True)
            wavelet_config = replace(config, wavelet=wavelet)
            comparison = prepare_comparison(
                signals, contact_time, contact_amplitude, contact_fps, wavelet_config,
            )
            records.append(record_from_comparison(
                dataset_id, wavelet, comparison, source_metadata,
            ))

    validate_records(records, arguments.ids, arguments.wavelets)
    aggregate = aggregate_records(records, arguments.wavelets)
    summary = {
        "status": "complete",
        **run_metadata,
        "records": records,
        "aggregate_by_wavelet": aggregate,
        "validation": {
            "expected_rows": len(arguments.ids) * len(arguments.wavelets),
            "actual_rows": len(records),
            "all_rows_finite": True,
            "same_input_grid_within_each_dataset": True,
        },
    }
    json_path = arguments.output / "mother_wavelet_results.json"
    json_path.write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
    with (arguments.output / "mother_wavelet_results.csv").open(
        "w", newline="", encoding="utf-8-sig",
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    plot_summary(records, arguments.ids, arguments.wavelets,
                 arguments.output / "mother_wavelet_comparison.png")
    print(json.dumps(aggregate, indent=2), flush=True)
    print(f"Saved: {arguments.output.resolve()}", flush=True)


if __name__ == "__main__":
    main()
