"""Read-only source analysis: five preselected pulse samples and a PNG.

Chart contract: static two-series line plots (full overlap + fixed zoom),
then an exact five-row lookup table. Blue/solid versus orange/dashed.
No lag optimization, polarity flipping, or source-data edits. Matching file
names and relative start times do not establish hardware synchronization.
"""

from pathlib import Path
import json
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "ippg"))
from ippg_v2_refactored import (
    IPPGConfig, extract_video_signals, analyze_signals,
)
from benchmark_background_methods import analyze_contact_ppg


def zscore(values):
    std = float(np.std(values))
    if std <= 1e-12:
        raise ValueError("Cannot standardize a constant pulse.")
    return (values - np.mean(values)) / std


def main():
    output = Path(__file__).resolve().parent
    video = ROOT / "DATA/temp/11.avi"
    contact = ROOT / "DATAcppg/11.xlsx"
    config = IPPGConfig(max_duration_seconds=None, show_preview=False,
                        pulse_method="pos", background_method="none")
    signals = extract_video_signals(video, config)
    result = analyze_signals(signals, config)
    contact_analysis = analyze_contact_ppg(
        contact, config.forehead_low_hz, config.forehead_high_hz, filter_order=3
    )
    ct = contact_analysis.time_seconds
    cf = contact_analysis.filtered_amplitude
    # Retain the existing pipeline's video grid; interpolate cPPG, not new measurements.
    common = ((signals.time_seconds >= ct[0]) & (signals.time_seconds <= ct[-1]))
    t = signals.time_seconds[common]
    ip = zscore(result.filtered_pulse_signal[common])
    cp = zscore(np.interp(t, ct, cf))
    sample_times = np.array([5.0, 5.2, 5.4, 5.6, 5.8])
    assert t[0] < sample_times[0] < sample_times[-1] < t[-1]
    sample_ip = np.interp(sample_times, t, ip)
    sample_cp = np.interp(sample_times, t, cp)
    difference = sample_ip - sample_cp
    r = float(np.corrcoef(ip, cp)[0, 1])
    metrics = {
        "video": str(video), "contact": str(contact),
        "time_assumption": "Common recording start; no lag correction or polarity flip",
        "bandpass_hz": [0.6, 2.0], "filter_orders": {"ippg": 6, "cppg": 3},
        "normalization": "Each waveform z-scored over full common interval",
        "common_start_s": float(t[0]), "common_end_s": float(t[-1]),
        "common_samples": len(t), "video_fps": signals.fps,
        "contact_fps": contact_analysis.fps,
        "cppg_fft_bpm": contact_analysis.heart_rate_bpm,
        "cppg_bpm_scope": "Full XLSX series after resample and band-pass",
        "full_overlap_pearson_r": r,
        "five_sample_mae_z": float(np.mean(np.abs(difference))),
        "samples": [{"time_s": float(s), "ippg_z": float(i), "cppg_z": float(c),
                     "absolute_difference_z": float(abs(i-c))}
                    for s, i, c in zip(sample_times, sample_ip, sample_cp)],
    }
    print(json.dumps(metrics, indent=2))
    (output / "comparison_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    np.savez_compressed(output / "comparison_arrays.npz", time=t, ippg_z=ip, cppg_z=cp)

    blue, orange, ink, grid = "#2878B5", "#D77824", "#24272B", "#E0E3E6"
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
                         "text.color": ink, "axes.labelcolor": ink,
                         "xtick.color": ink, "ytick.color": ink})
    fig = plt.figure(figsize=(11, 9.6), facecolor="white", layout="constrained")
    gs = fig.add_gridspec(3, 1, height_ratios=[1, 1.35, 0.95])
    ax1, ax2, ax3 = [fig.add_subplot(gs[i]) for i in range(3)]
    fig.suptitle("iPPG vs contact PPG | recording 11\n"
                 "POS + none; band-pass 0.6-2.0 Hz; independently Z-scored",
                 fontsize=16)
    for ax in (ax1, ax2):
        ax.plot(t, ip, color=blue, lw=1.6, label="iPPG (POS)")
        ax.plot(t, cp, color=orange, lw=1.6, ls="--", label="cPPG")
        ax.axhline(0, color=grid, lw=0.8)
        ax.grid(axis="y", color=grid, linewidth=0.6)
        ax.set_ylabel("Pulse (Z-score)")
        ax.set_xlabel("Time from assumed recording start (s)")
        ax.spines[["top", "right"]].set_visible(False)
        ax.spines[["left", "bottom"]].set_color("#AEB3B8")
    ax1.set_xlim(t[0], t[-1])
    ax1.set_title(f"Full overlap: {t[0]:.2f}-{t[-1]:.2f} s | Pearson r = {r:+.3f}", loc="left")
    ax1.legend(loc="upper right", ncol=2, frameon=False)
    ax1.axvspan(4.8, 6.2, color=grid, alpha=0.4)

    ax2.set_xlim(4.8, 6.2)
    zoom = (t >= 4.8) & (t <= 6.2)
    ylow = min(ip[zoom].min(), cp[zoom].min())
    yhigh = max(ip[zoom].max(), cp[zoom].max())
    ax2.set_ylim(ylow-0.3, yhigh+0.5)
    ax2.set_title("Five preselected timestamps (zoom)", loc="left")
    ax2.scatter(sample_times, sample_ip, color=blue, marker="o", s=45, zorder=4)
    ax2.scatter(sample_times, sample_cp, color=orange, marker="s", s=45, zorder=4)
    for n, (s, i, c) in enumerate(zip(sample_times, sample_ip, sample_cp), 1):
        ax2.plot([s,s], [i,c], color="#777777", linewidth=1.0, zorder=1)
        ax2.annotate(str(n), (s,max(i,c)), xytext=(0,12), textcoords="offset points",
                     ha="center", fontsize=11)
    ax3.axis("off")
    rows = [[str(n), f"{s:.1f}", f"{i:+.3f}", f"{c:+.3f}", f"{abs(i-c):.3f}"]
            for n,(s,i,c) in enumerate(zip(sample_times,sample_ip,sample_cp),1)]
    table = ax3.table(cellText=rows,
                      colLabels=["Point", "Time (s)", "iPPG (Z)", "cPPG (Z)", "Absolute gap (Z)"],
                      cellLoc="center", bbox=[0,0.35,1,0.62])
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    for (row,col),cell in table.get_celld().items():
        cell.set_edgecolor(grid)
        cell.set_linewidth(0.5)
        if row == 0:
            cell.set_facecolor("#F1F3F5")
        elif col in (2,3):
            cell.get_text().set_color(blue if col==2 else orange)
    ax3.text(0,0.20,f"Five-point mean absolute gap = {metrics['five_sample_mae_z']:.3f} Z units (not BPM or %).",
             transform=ax3.transAxes)
    ax3.text(0,0.01,"No lag correction or sign flip. Synchronization unverified. Five points are illustrative only.\n"
             "Sources: DATA/temp/11.avi + DATAcppg/11.xlsx; cPPG interpolated to the video grid.",
             fontsize=9, transform=ax3.transAxes, va="bottom")
    fig.savefig(output / "pulse_five_samples.png", dpi=160, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()
