"""Run all prespecified DATA pairs; save three-panel plots, stages and metrics.

Example: python "ippg/benchmark_wavelet.py" --output outputs/wavelet-validation
Outputs must be new (no overwrite). Inputs are read-only. No cPPG-based tuning.
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from compare_gr_pos_cppg import analyze_pair, dataset_paths, plot_comparison
from ippg_v2_refactored import IPPGConfig, add_wavelet_arguments, wavelet_config_from_arguments


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("ids", nargs="*", default=["11", "12", "13", "21", "62"])
    parser.add_argument("--output", type=Path, required=True)
    add_wavelet_arguments(parser, include_denoise=False)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    config = IPPGConfig(max_duration_seconds=None, show_preview=False,
                        **wavelet_config_from_arguments(args))
    rows = []
    for dataset_id in args.ids:
        print(f"DATA {dataset_id}", flush=True)
        try:
            data, metadata = analyze_pair(*dataset_paths(dataset_id), config)
            metrics = {"dataset_id": dataset_id, "status": "ok", **metadata, **data.metrics()}
            figure = plot_comparison(data, metadata)
            figure.savefig(args.output / f"{dataset_id}.png", dpi=160, facecolor="white")
            plt.close(figure)
            if data.time_seconds[0] <= 5 and data.time_seconds[-1] >= 10:
                figure = plot_comparison(data, metadata, (5, 10))
                figure.savefig(args.output / f"{dataset_id}_zoom.png", dpi=160, facecolor="white")
                plt.close(figure)
            stages = data.pos_dwt_processing
            np.savez_compressed(args.output / f"{dataset_id}_signals.npz",
                                common_time=data.time_seconds, cppg_z=data.cppg_z,
                                gr_z=data.gr_z, pos_z=data.pos_z, pos_dwt_z=data.pos_dwt_z,
                                raw_pos=stages.raw, reconstructed_pos=stages.reconstructed,
                                filtered_pos_dwt=stages.filtered)
            print(json.dumps(metrics["methods"]), flush=True)
        except (ValueError, RuntimeError, OSError) as error:
            metrics = {"dataset_id": dataset_id, "status": "failed", "error": str(error)}
            print(metrics, flush=True)
        rows.append(metrics)
        (args.output / f"{dataset_id}.json").write_text(json.dumps(metrics, indent=2, allow_nan=False), encoding="utf-8")
    successful = [row for row in rows if row["status"] == "ok"]
    aggregate = {}
    for method in ("g-r", "pos", "pos-dwt"):
        errors = np.array([row["methods"][method]["error_bpm"] for row in successful])
        aggregate[method] = {"clips": len(errors),
                             "mae_bpm": float(np.mean(abs(errors))) if len(errors) else None,
                             "rmse_bpm": float(np.sqrt(np.mean(errors**2))) if len(errors) else None}
    summary = {"requested_ids": args.ids, "successful_clips": len(successful),
               "failed_clips": len(rows) - len(successful), "aggregate": aggregate, "results": rows}
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(aggregate, indent=2), flush=True)
    if len(successful) != len(rows):
        raise SystemExit("Some requested clips failed; see summary.json (no failures hidden).")


if __name__ == "__main__":
    main()
