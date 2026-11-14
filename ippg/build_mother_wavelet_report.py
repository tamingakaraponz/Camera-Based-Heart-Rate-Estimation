"""Create the canonical portable-report input from a mother-wavelet benchmark.

The report HTML itself is deliberately built by the Data Analytics portable
report renderer, not by this script.  Run benchmark_mother_wavelets.py first.
"""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
import json
from pathlib import Path
import sqlite3


def compact_clip_rows(summary: dict) -> list[dict]:
    return [{
        "dataset": row["dataset_id"],
        "wavelet": row["wavelet"],
        "cppg_bpm": row["cppg_fft_bpm"],
        "ippg_bpm": row["ippg_fft_bpm"],
        "absolute_error_bpm": row["absolute_error_bpm"],
        "pearson_r": row["pearson_r"],
        "waveform_mae_z": row["waveform_mae_z"],
        "waveform_rmse_z": row["waveform_rmse_z"],
        "wavelet_level": row["wavelet_level"],
        "threshold": row["threshold"],
    } for row in summary["records"]]


def write_sqlite_database(summary: dict, database: Path) -> None:
    """Materialize the reviewed report rows so source SQL is reproducible."""

    if database.exists():
        raise FileExistsError(f"Refusing to overwrite existing database: {database}")
    records = summary["records"]
    aggregate = summary["aggregate_by_wavelet"]
    with sqlite3.connect(database) as connection:
        connection.execute("""
            CREATE TABLE mother_wavelet_clip_metrics (
                dataset_id TEXT NOT NULL,
                wavelet TEXT NOT NULL,
                cppg_bpm REAL NOT NULL,
                ippg_bpm REAL NOT NULL,
                absolute_error_bpm REAL NOT NULL,
                pearson_r REAL NOT NULL,
                waveform_mae_z REAL NOT NULL,
                waveform_rmse_z REAL NOT NULL,
                wavelet_level INTEGER NOT NULL,
                threshold REAL NOT NULL
            )
        """)
        connection.executemany("""
            INSERT INTO mother_wavelet_clip_metrics VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, [
            (
                row["dataset_id"], row["wavelet"], row["cppg_fft_bpm"], row["ippg_fft_bpm"],
                row["absolute_error_bpm"], row["pearson_r"], row["waveform_mae_z"],
                row["waveform_rmse_z"], row["wavelet_level"], row["threshold"],
            ) for row in records
        ])
        connection.execute("""
            CREATE TABLE mother_wavelet_summary (
                wavelet TEXT NOT NULL,
                clips INTEGER NOT NULL,
                bpm_mae REAL NOT NULL,
                bpm_rmse REAL NOT NULL,
                mean_pearson_r REAL NOT NULL,
                mean_waveform_mae_z REAL NOT NULL,
                mean_waveform_rmse_z REAL NOT NULL
            )
        """)
        connection.executemany("""
            INSERT INTO mother_wavelet_summary VALUES (?, ?, ?, ?, ?, ?, ?)
        """, [
            (
                row["wavelet"], row["clips"], row["bpm_mae"], row["bpm_rmse"],
                row["mean_pearson_r"], row["mean_waveform_mae_z"],
                row["mean_waveform_rmse_z"],
            ) for row in aggregate
        ])


def artifact_from_summary(summary: dict, database_path: str) -> dict:
    aggregate = summary["aggregate_by_wavelet"]
    clips = compact_clip_rows(summary)
    wavelets = [row["wavelet"] for row in aggregate]
    datasets = summary["datasets"]
    bpm_mae = aggregate[0]["bpm_mae"]
    bpm_rmse = aggregate[0]["bpm_rmse"]
    waveform_best = min(aggregate, key=lambda row: row["mean_waveform_mae_z"])
    waveform_worst = max(aggregate, key=lambda row: row["mean_waveform_mae_z"])
    waveform_gap = waveform_worst["mean_waveform_mae_z"] - waveform_best["mean_waveform_mae_z"]
    generated = datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    clip_source = {
        "id": "clip_metrics_sql",
        "label": "Recording-level mother-wavelet metrics",
        "path": database_path,
        "query": {
            "engine": "sqlite",
            "language": "sql",
            "sql": "SELECT dataset_id, wavelet, cppg_bpm, ippg_bpm, absolute_error_bpm, pearson_r, waveform_mae_z, waveform_rmse_z, wavelet_level, threshold FROM mother_wavelet_clip_metrics ORDER BY dataset_id, wavelet",
            "description": "Loads the 20 reviewed recording-level iPPG/cPPG comparisons.",
            "tables_used": ["mother_wavelet_clip_metrics"],
            "filters": ["DATA in {11, 12, 13, 21, 62}", "wavelet in {db4, sym4, sym8, bior3.5}"],
        },
    }
    summary_source = {
        "id": "wavelet_summary_sql",
        "label": "Wavelet-level aggregate metrics",
        "path": database_path,
        "query": {
            "engine": "sqlite",
            "language": "sql",
            "sql": "SELECT wavelet, clips, bpm_mae, bpm_rmse, mean_pearson_r, mean_waveform_mae_z, mean_waveform_rmse_z FROM mother_wavelet_summary ORDER BY wavelet",
            "description": "Loads equally weighted five-recording aggregates for each mother wavelet.",
            "tables_used": ["mother_wavelet_summary"],
            "filters": ["Five recordings per wavelet"],
        },
    }
    configuration_source = {
        "id": "benchmark_configuration",
        "label": "Benchmark pipeline and controls",
        "path": "ippg/benchmark_mother_wavelets.py",
    }
    return {
        "surface": "report",
        "manifest": {
            "version": 1,
            "surface": "report",
            "title": "Mother wavelet comparison for POS iPPG",
            "description": "A controlled comparison of db4, sym4, sym8 and bior3.5 on five paired iPPG/cPPG recordings.",
            "generatedAt": generated,
            "cards": [],
            "charts": [
                {
                    "id": "clip_bpm_error",
                    "title": "Absolute FFT BPM error by recording and mother wavelet",
                    "subtitle": "Five recordings; lower is better. All wavelets use the same RGB/POS and cPPG grid within a recording.",
                    "type": "bar",
                    "intent": "comparison",
                    "question": "Does any mother wavelet reduce the FFT heart-rate error across the five recordings?",
                    "rationale": "Grouped bars retain the recording-level variation while making a tie between wavelets visible.",
                    "comparisonContext": {
                        "baseline": "cPPG FFT on the common video-time grid",
                        "grain": "one recording-wavelet result",
                        "unit": "BPM",
                    },
                    "palette": {"kind": "categorical", "name": "wavelet identity"},
                    "legend": {"position": "bottom", "title": "Mother wavelet", "sort": "spec"},
                    "labels": {"values": "auto"},
                    "dataset": "clip_metrics",
                    "sourceId": "clip_metrics_sql",
                    "valueFormat": "number",
                    "encodings": {
                        "x": {"field": "dataset", "type": "nominal", "label": "Recording"},
                        "y": {"field": "absolute_error_bpm", "type": "quantitative", "label": "Absolute FFT BPM error"},
                        "color": {"field": "wavelet", "type": "nominal", "label": "Mother wavelet"},
                        "tooltip": [
                            {"field": "cppg_bpm", "type": "quantitative", "label": "cPPG FFT", "format": "number"},
                            {"field": "ippg_bpm", "type": "quantitative", "label": "iPPG FFT", "format": "number"},
                            {"field": "pearson_r", "type": "quantitative", "label": "Pearson r", "format": "number"},
                        ],
                    },
                },
                {
                    "id": "aggregate_waveform_mae",
                    "title": "Mean waveform MAE across five recordings",
                    "subtitle": "Z-score MAE over each full common interval; lower is better. Clip means are equally weighted.",
                    "type": "bar",
                    "intent": "comparison",
                    "question": "Does the waveform difference from cPPG materially differ by mother wavelet?",
                    "rationale": "A four-category bar chart makes the small aggregate waveform differences auditable without implying a trend.",
                    "comparisonContext": {
                        "baseline": "cPPG waveform interpolated to the video grid",
                        "grain": "unweighted mean across five recordings",
                        "normalization": "Z-score over each full common interval",
                        "unit": "Z-score MAE",
                    },
                    "palette": {"kind": "categorical", "name": "wavelet identity"},
                    "labels": {"values": "all"},
                    "dataset": "wavelet_summary",
                    "sourceId": "wavelet_summary_sql",
                    "valueFormat": "number",
                    "encodings": {
                        "x": {"field": "wavelet", "type": "nominal", "label": "Mother wavelet"},
                        "y": {"field": "mean_waveform_mae_z", "type": "quantitative", "label": "Mean waveform MAE (Z-score)"},
                        "tooltip": [
                            {"field": "bpm_mae", "type": "quantitative", "label": "BPM MAE", "format": "number"},
                            {"field": "bpm_rmse", "type": "quantitative", "label": "BPM RMSE", "format": "number"},
                            {"field": "mean_pearson_r", "type": "quantitative", "label": "Mean Pearson r", "format": "number"},
                        ],
                    },
                },
            ],
            "tables": [
                {
                    "id": "clip_detail",
                    "title": "Exact recording-level results",
                    "subtitle": "Twenty controlled iPPG/cPPG comparisons: five recordings multiplied by four mother wavelets.",
                    "dataset": "clip_metrics",
                    "sourceId": "clip_metrics_sql",
                    "defaultSort": {"field": "dataset", "direction": "asc"},
                    "columns": [
                        {"field": "dataset", "label": "DATA", "type": "text"},
                        {"field": "wavelet", "label": "Wavelet", "type": "text"},
                        {"field": "cppg_bpm", "label": "cPPG FFT (BPM)", "format": "number"},
                        {"field": "ippg_bpm", "label": "iPPG FFT (BPM)", "format": "number"},
                        {"field": "absolute_error_bpm", "label": "Abs. error (BPM)", "format": "number"},
                        {"field": "pearson_r", "label": "Pearson r", "format": "number"},
                        {"field": "waveform_mae_z", "label": "Waveform MAE (Z)", "format": "number"},
                        {"field": "threshold", "label": "DWT threshold", "format": "number"},
                    ],
                },
            ],
            "sources": [clip_source, summary_source, configuration_source],
            "blocks": [
                {"id": "title", "type": "markdown", "body": "# Mother wavelet comparison for POS iPPG"},
                {
                    "id": "technical_summary", "type": "markdown", "sourceId": "wavelet_summary_sql",
                    "body": (
                        "## Technical summary\n\n"
                        f"Across DATA {', '.join(datasets)}, **db4, sym4, sym8 and bior3.5 produced exactly the same FFT BPM result for every recording**. "
                        f"Their equal clip-average error is **{bpm_mae:.2f} BPM MAE** and **{bpm_rmse:.2f} BPM RMSE**. "
                        f"The smallest mean waveform MAE belongs to **{waveform_best['wavelet']}** ({waveform_best['mean_waveform_mae_z']:.6f} Z), "
                        f"but its gap to the worst value is only **{waveform_gap:.6f} Z**—too small here to justify choosing a winner."
                    ),
                },
                {
                    "id": "bpm_interpretation", "type": "markdown", "sourceId": "clip_metrics_sql",
                    "body": (
                        "## FFT heart-rate error is a four-way tie\n\n"
                        "Each colored group below shares one video-derived RGB/POS signal and one cPPG reference window. "
                        "The four bars are identical within every DATA set, so changing this mother wavelet alone did not change the dominant in-band FFT bin. "
                        "The larger error in DATA 21 (8.93 BPM) is therefore a recording/pipeline issue, not a db4-versus-sym-versus-bior choice under this configuration."
                    ),
                },
                {"id": "bpm_chart", "type": "chart", "chartId": "clip_bpm_error"},
                {
                    "id": "waveform_interpretation", "type": "markdown", "sourceId": "wavelet_summary_sql",
                    "body": (
                        "## Waveform differences are negligible at this scale\n\n"
                        f"The aggregate waveform MAE ranges from {waveform_best['mean_waveform_mae_z']:.6f} Z ({waveform_best['wavelet']}) "
                        f"to {waveform_worst['mean_waveform_mae_z']:.6f} Z ({waveform_worst['wavelet']}). "
                        "This is a descriptive difference only: it is not a statistical significance test and does not establish a better waveform reconstruction."
                    ),
                },
                {"id": "waveform_chart", "type": "chart", "chartId": "aggregate_waveform_mae"},
                {
                    "id": "scope_definitions", "type": "markdown", "sourceId": "clip_metrics_sql",
                    "body": (
                        "## Scope and metric definitions\n\n"
                        f"The cohort is five paired recordings (DATA {', '.join(datasets)}) and four wavelets ({', '.join(wavelets)}), for {len(clips)} results. "
                        "BPM error is the absolute difference between iPPG and cPPG FFT peaks on the full common video-time grid. "
                        "Waveform MAE and Pearson r compare Z-scored signals over that same full interval; r is a correlation coefficient, not an accuracy percentage."
                    ),
                },
                {
                    "id": "methodology", "type": "markdown", "sourceId": "benchmark_configuration",
                    "body": (
                        "## Controlled method\n\n"
                        "For each recording, the pipeline extracts RGB once, calculates POS once, then runs `DWT → soft threshold → IDWT → 0.6–2.0 Hz band-pass → FFT` four times. "
                        "DWT uses level auto (actual level 4), symmetric boundary handling, strength 1.0 and `protect-pulse-band`: approximation coefficients and detail bands overlapping 0.6–2.0 Hz are preserved; only non-overlapping detail bands are soft-thresholded. "
                        "Background method is `none`; there is no lag correction, polarity flip, or cPPG-based tuning."
                    ),
                },
                {
                    "id": "limitations", "type": "markdown", "sourceId": "clip_metrics_sql",
                    "body": (
                        "## Limits and robustness checks\n\n"
                        "The benchmark completed all 20 planned rows; every metric is finite, and the video rate, cPPG rate, common sample count, common time span and cPPG FFT value were verified invariant across wavelets within each recording. "
                        "It remains a five-clip descriptive test. cPPG/iPPG synchronization is assumed rather than independently verified, and the protected-band design intentionally prevents the threshold from changing much of the pulse-frequency content."
                    ),
                },
                {
                    "id": "next_steps", "type": "markdown", "sourceId": "benchmark_configuration",
                    "body": (
                        "## Recommended next step\n\n"
                        "Keep **db4** as the default because it is the established baseline and no alternative wins in this controlled test. "
                        "To find a meaningful wavelet effect, evaluate a pre-registered strength/level grid on more recordings, include motion/illumination strata, and report paired confidence intervals after independently checking synchronization."
                    ),
                },
                {"id": "detail_table", "type": "table", "tableId": "clip_detail"},
                {
                    "id": "further_questions", "type": "markdown", "sourceId": "benchmark_configuration",
                    "body": (
                        "## Further question\n\n"
                        "Should the next experiment compare mother wavelets under the legacy `all-details` threshold strategy as well? That strategy is more aggressive, but it can suppress components near the pulse band and must be assessed against cPPG without tuning on the test clips."
                    ),
                },
            ],
        },
        "snapshot": {
            "version": 1,
            "generatedAt": generated,
            "status": "ready",
            "datasets": {"clip_metrics": clips, "wavelet_summary": aggregate},
            "accessIssues": [],
        },
        "sources": [clip_source, summary_source, configuration_source],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results", type=Path,
        default=Path("outputs/mother-wavelet-comparison-20260830/mother_wavelet_results.json"),
    )
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    summary = json.loads(arguments.results.read_text(encoding="utf-8"))
    if summary.get("status") != "complete":
        raise ValueError("Benchmark summary must have status=complete.")
    database = arguments.output.with_name("mother_wavelet_results.sqlite")
    write_sqlite_database(summary, database)
    artifact = artifact_from_summary(summary, database.as_posix())
    arguments.output.write_text(json.dumps(artifact, indent=2, allow_nan=False), encoding="utf-8")
    print(f"Saved: {arguments.output.resolve()}")


if __name__ == "__main__":
    main()
