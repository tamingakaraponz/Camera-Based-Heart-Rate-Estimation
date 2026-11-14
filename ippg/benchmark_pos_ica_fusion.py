"""Benchmark POS, ICA and both fusion domains on paired DATA recordings."""

from __future__ import annotations

import argparse
import csv
from dataclasses import replace
from datetime import date
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from compare_gr_pos_cppg import dataset_paths
from compare_pos_ica_cppg import analyze_pair
from ippg_v2_refactored import IPPGConfig


METHODS = ("pos-dwt", "ica-dwt", "pos-ica-time", "pos-ica-spectral")
DEFAULT_DATASETS = ("11", "12", "13", "21", "22", "23")


def choose_default(rows: list[dict]) -> tuple[str, dict]:
    """Choose POS+ICA only when its median, then mean, beats POS."""

    errors = {
        method: np.asarray(
            [row["absolute_error_bpm"] for row in rows if row["method"] == method],
            dtype=float,
        )
        for method in ("pos-dwt", "pos-ica-spectral")
    }
    if any(values.size == 0 for values in errors.values()):
        return "pos", {"reason": "incomplete benchmark"}
    statistics = {
        method: {
            "median_absolute_error_bpm": float(np.median(values)),
            "mean_absolute_error_bpm": float(np.mean(values)),
        }
        for method, values in errors.items()
    }
    pos = statistics["pos-dwt"]
    spectral = statistics["pos-ica-spectral"]
    if spectral["median_absolute_error_bpm"] < pos["median_absolute_error_bpm"]:
        selected = "pos-ica"
        reason = "spectral fusion has lower median absolute BPM error"
    elif np.isclose(
        spectral["median_absolute_error_bpm"], pos["median_absolute_error_bpm"]
    ) and spectral["mean_absolute_error_bpm"] < pos["mean_absolute_error_bpm"]:
        selected = "pos-ica"
        reason = "median tied and spectral fusion has lower mean absolute BPM error"
    else:
        selected = "pos"
        reason = "POS wins or remains tied after median and mean comparison"
    return selected, {"reason": reason, "statistics": statistics}


def summarize(rows: list[dict], datasets: list[str]) -> dict:
    summaries = {}
    for method in METHODS:
        values = np.asarray(
            [row["absolute_error_bpm"] for row in rows if row["method"] == method]
        )
        summaries[method] = {
            "count": int(values.size),
            "median_absolute_error_bpm": float(np.median(values)),
            "mean_absolute_error_bpm": float(np.mean(values)),
        }
    pos_by_data = {
        row["data_id"]: row["absolute_error_bpm"]
        for row in rows
        if row["method"] == "pos-dwt"
    }
    for method in METHODS[1:]:
        win = tie = loss = 0
        for row in rows:
            if row["method"] != method:
                continue
            baseline = pos_by_data[row["data_id"]]
            if np.isclose(row["absolute_error_bpm"], baseline):
                tie += 1
            elif row["absolute_error_bpm"] < baseline:
                win += 1
            else:
                loss += 1
        summaries[method]["versus_pos"] = {"win": win, "tie": tie, "loss": loss}
    selected, decision = choose_default(rows)
    return {
        "datasets": datasets,
        "methods": summaries,
        "recommended_pulse_method_default": selected,
        "default_decision": decision,
        "selection_rule": (
            "Choose spectral POS+ICA when median absolute BPM error is lower; "
            "use mean as tie-breaker; exact tie keeps POS."
        ),
        "caveats": [
            "No lag correction or cPPG-driven polarity adjustment.",
            "The six recordings are a preliminary benchmark, not an independent validation set.",
        ],
    }


def plot_results(rows: list[dict], datasets: list[str], output_path: Path) -> None:
    """Save the four-method absolute-error chart without a GUI backend."""

    x = np.arange(len(datasets))
    width = 0.2
    figure, axis = plt.subplots(figsize=(12, 6))
    for index, method in enumerate(METHODS):
        values = [
            row["absolute_error_bpm"] for row in rows if row["method"] == method
        ]
        axis.bar(x + (index - 1.5) * width, values, width, label=method)
    axis.set_xticks(x, [f"DATA {item}" for item in datasets])
    axis.set_ylabel("Absolute FFT BPM error")
    axis.set_title("POS, ICA and quality-weighted fusion vs cPPG")
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=160, facecolor="white")
    plt.close(figure)


def parse_arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("datasets", nargs="*", default=list(DEFAULT_DATASETS))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs") / f"pos-ica-benchmark-{date.today():%Y%m%d}",
    )
    parser.add_argument("--duration", type=float, help="Maximum video duration")
    parser.add_argument("--no-mediapipe", action="store_true")
    parser.add_argument("--roi-method", choices=("legacy", "quality"), default="legacy")
    parser.add_argument("--preview", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def main() -> None:
    args = parse_arguments()
    output_dir = args.output_dir.resolve()
    outputs = {
        "csv": output_dir / "results.csv",
        "json": output_dir / "summary.json",
        "png": output_dir / "absolute_bpm_error.png",
    }
    existing = [str(path) for path in outputs.values() if path.exists()]
    if existing and not args.overwrite:
        raise SystemExit("Output exists; use --overwrite:\n" + "\n".join(existing))
    output_dir.mkdir(parents=True, exist_ok=True)

    config = IPPGConfig(
        max_duration_seconds=args.duration,
        pulse_method="pos-ica",
        background_method="none",
        fusion_domain="spectral",
        prefer_mediapipe=not args.no_mediapipe,
        show_preview=args.preview,
        roi_method=args.roi_method,
    )
    rows: list[dict] = []
    for dataset_id in args.datasets:
        video, contact = dataset_paths(dataset_id)
        print(f"DATA {dataset_id}: {video.name} + {contact.name}")
        if not video.is_file() or not contact.is_file():
            raise SystemExit(f"Missing paired files for DATA {dataset_id}.")
        comparison, _ = analyze_pair(video, contact, replace(config))
        metrics = comparison.metrics()
        for method in METHODS:
            values = metrics["methods"][method]
            rows.append(
                {
                    "data_id": dataset_id,
                    "method": method,
                    "cppg_bpm": metrics["cppg_fft_bpm"],
                    "estimated_bpm": values["fft_bpm"],
                    "error_bpm": values["error_bpm"],
                    "absolute_error_bpm": values["absolute_error_bpm"],
                    "pos_weight": comparison.fusion.weights["pos"],
                    "ica_weight": comparison.fusion.weights["ica"],
                    "ica_component": comparison.fusion.selected_ica.component_index,
                    "fusion_status": comparison.fusion.status,
                }
            )

    with outputs["csv"].open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = summarize(rows, list(args.datasets))
    summary["roi_method"] = args.roi_method
    outputs["json"].write_text(
        json.dumps({"rows": rows, "summary": summary}, indent=2, allow_nan=False),
        encoding="utf-8",
    )

    plot_results(rows, list(args.datasets), outputs["png"])
    print(json.dumps(summary, indent=2))
    print(f"Saved benchmark: {output_dir}")


if __name__ == "__main__":
    main()
