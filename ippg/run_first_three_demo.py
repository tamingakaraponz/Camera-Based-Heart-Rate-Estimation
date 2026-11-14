"""Reproducible 11/12/13 POS versus protected-DWT experiment.

Observes existing extraction without changing its returned frames or masks.
No contact-reference correction or algorithm tuning is performed.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
from hashlib import sha256
import importlib.metadata
import json
from pathlib import Path
import platform
import time
from unittest.mock import patch

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import ippg_v2_refactored as core


ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def extract_observed(video, config, folder):
    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    state = {"frames": [], "tracker": None, "snapshot": None}
    original_factory = core._create_region_tracker
    original_skin = core.AdaptiveSkinDetector

    class TrackerObserver:
        def __init__(self, tracker):
            self.tracker = tracker
            state["tracker"] = tracker.name

        def detect(self, frame, timestamp_ms):
            regions = self.tracker.detect(frame, timestamp_ms)
            state["frame"] = frame
            state["regions"] = regions
            item = {"frame": len(state["frames"]) + 1, "face_found": regions is not None, "skin_counts": {}}
            state["current"] = item
            state["frames"].append(item)
            state["masks"] = {}
            return regions

        def close(self):
            self.tracker.close()

    class SkinObserver(original_skin):
        def create_mask(self, region, geometry_mask, region_name):
            mask = super().create_mask(region, geometry_mask, region_name)
            item = state["current"]
            item["skin_counts"][region_name] = int(np.count_nonzero(mask))
            item["calibrated"] = self.ready
            state["masks"][region_name] = mask.copy()
            if (self.ready and item["frame"] / fps >= 5.0 and state["snapshot"] is None
                    and len(item["skin_counts"]) == 3
                    and min(item["skin_counts"].values()) >= config.minimum_skin_pixels_per_region):
                display = state["frame"].copy()
                regions = state["regions"]
                for name, color in (("forehead", (0, 255, 255)), ("left_cheek", (255, 255, 0)), ("right_cheek", (255, 255, 0))):
                    box = getattr(regions, name + "_box")
                    core._overlay_skin_pixels(display, box, state["masks"][name], getattr(regions, name + "_geometry"), color)
                display = cv2.resize(display, (960, round(display.shape[0] * 960 / display.shape[1])))
                cv2.putText(display, "Actual extraction | yellow: forehead | cyan: cheeks", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2)
                target = folder / (video.stem + "_roi.png")
                if not cv2.imwrite(str(target), display):
                    raise RuntimeError("Could not save ROI preview")
                state["snapshot"] = {"path": target.name, "frame": item["frame"], "video_time_s": item["frame"] / fps}
            return mask

    # Both wrappers return the original outputs untouched. Preview remains disabled.
    with patch.object(core, "_create_region_tracker", lambda c: TrackerObserver(original_factory(c))), patch.object(core, "AdaptiveSkinDetector", SkinObserver):
        signals = core.extract_video_signals(video, config)
    valid = [r for r in state["frames"] if r.get("calibrated") and len(r["skin_counts"]) == 3 and min(r["skin_counts"].values()) >= config.minimum_skin_pixels_per_region]
    valid_ids = np.asarray([r["frame"] for r in valid], dtype=int)
    telemetry = {
        "tracker": state["tracker"], "frames_observed": len(state["frames"]),
        "frames_without_face": sum(not r["face_found"] for r in state["frames"]),
        "frames_used_before_interpolation": len(valid),
        "frames_not_used_including_calibration": len(state["frames"]) - len(valid),
        "first_used_video_time_s": float(valid_ids[0] / fps) if len(valid_ids) else None,
        "last_used_video_time_s": float(valid_ids[-1] / fps) if len(valid_ids) else None,
        "max_missing_frames_between_used": int(np.max(np.diff(valid_ids)) - 1) if len(valid_ids) > 1 else None,
        "roi_snapshot": state["snapshot"],
    }
    return signals, telemetry


def plot_stages(code, signals, baseline, dwt, path):
    times = signals.time_seconds
    fig, ax = plt.subplots(5, 1, figsize=(13, 15), layout="constrained")
    fig.suptitle(f"DATA {code} | POS: {baseline.heart_rate_bpm:.2f} BPM | POS + DWT: {dwt.heart_rate_bpm:.2f} BPM\nSame RGB, time interval and band-pass | No sensor timing correction", fontsize=14)
    rgb = signals.combined_rgb
    for i, (label, color) in enumerate((("R", "#c84848"), ("G", "#30905a"), ("B", "#3879be"))):
        ax[0].plot(times, rgb[:, i] / np.mean(rgb[:, i]) - 1, label=label, color=color, linewidth=0.9)
    ax[0].set_title("1. Relative RGB changes from selected skin (display normalization only)")
    ax[0].set_ylabel("Fractional change")
    ax[1].plot(times, signals.pulse_signal, label="Raw POS", color="#3879be", linewidth=1)
    ax[1].plot(times, dwt.reconstructed_pulse_signal, label="After protected DWT", color="#d57b20", linestyle="--", linewidth=1)
    ax[1].set_title("2. POS before band-pass: raw versus reconstructed")
    ax[1].set_ylabel("Amplitude (a.u.)")
    for a in ax[2:4]:
        a.plot(times, baseline.filtered_pulse_signal, label="POS + band-pass", color="#3879be", linewidth=1)
        a.plot(times, dwt.filtered_pulse_signal, label="POS + DWT + band-pass", color="#d57b20", linestyle="--", linewidth=1)
        a.set_ylabel("Amplitude (a.u.)")
    ax[2].set_title("3. Filtered pulse over the full analysis interval")
    ax[3].set_title("4. Same filtered pulse, zoomed to 5-10 s (no recalculation)")
    ax[3].set_xlim(max(5, times[0]), min(10, times[-1]))
    for a in ax[:4]:
        a.set_xlabel("Video time (s)")
    for result, label, color, style in ((baseline, "POS", "#3879be", "-"), (dwt, "POS + DWT", "#d57b20", "--")):
        bpm = result.frequencies_hz * 60
        selected = (bpm >= 36) & (bpm <= 120)
        power = result.compensated_spectrum[selected]
        ax[4].plot(bpm[selected], power / np.max(power), label=label, color=color, linestyle=style)
        ax[4].axvline(result.heart_rate_bpm, color=color, linestyle=style, linewidth=0.8)
    ax[4].set_title("5. FFT power: strongest frequency in the configured 36-120 BPM range")
    ax[4].set_xlabel("Frequency expressed as BPM")
    ax[4].set_ylabel("Normalized power")
    ax[4].set_xlim(36, 120)
    for a in ax:
        a.grid(alpha=0.2)
        a.legend(loc="upper right", fontsize=9)
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    folder = args.output.resolve()
    folder.mkdir(parents=True, exist_ok=False)
    config = core.IPPGConfig(max_duration_seconds=34.333, pulse_method="pos", roi_method="legacy", background_method="none", show_preview=False)
    versions = {p: importlib.metadata.version(p) for p in ("numpy", "scipy", "matplotlib", "opencv-contrib-python", "mediapipe", "PyWavelets")}
    summary = {"config": asdict(config), "versions": versions, "python": platform.python_version(), "core_sha256": digest(Path(core.__file__)), "model_sha256": digest(ROOT / "models/face_landmarker.task"), "reference_status": "reference_time_unverified; cPPG not used in this experiment", "records": []}
    for code in ("11", "12", "13"):
        started = time.perf_counter()
        video = ROOT / "DATA/temp" / (code + ".avi")
        before = digest(video)
        record = {"data_id": code, "video": str(video), "video_sha256": before}
        try:
            signals, telemetry = extract_observed(video, config, folder)
            extracted = time.perf_counter()
            baseline = core.analyze_signals(signals, replace(config, denoise="none"))
            base_done = time.perf_counter()
            dwt = core.analyze_signals(signals, replace(config, denoise="dwt"))
            dwt_done = time.perf_counter()
            record.update(status="success", telemetry=telemetry, fps=signals.fps,
                          uniform_samples=len(signals.time_seconds), analysis_start_s=float(signals.time_seconds[0]), analysis_end_s=float(signals.time_seconds[-1]),
                          pos_bpm=float(baseline.heart_rate_bpm), pos_dwt_bpm=float(dwt.heart_rate_bpm),
                          bpm_difference=float(dwt.heart_rate_bpm-baseline.heart_rate_bpm),
                          fft_bin_spacing_bpm=float(np.diff(baseline.frequencies_hz)[0]*60),
                          filtered_max_abs_difference=float(np.max(np.abs(baseline.filtered_pulse_signal-dwt.filtered_pulse_signal))),
                          dwt=dwt.denoise_metadata,
                          timing_seconds={"extraction":extracted-started,"pos_analysis":base_done-extracted,"dwt_analysis":dwt_done-base_done})
            np.savez_compressed(folder / (code + "_signals.npz"), times=signals.time_seconds, rgb=signals.combined_rgb,
                                raw_pos=signals.pulse_signal, reconstructed=dwt.reconstructed_pulse_signal,
                                pos_filtered=baseline.filtered_pulse_signal, dwt_filtered=dwt.filtered_pulse_signal,
                                frequencies_hz=baseline.frequencies_hz, pos_power=baseline.compensated_spectrum, dwt_power=dwt.compensated_spectrum)
            plot_stages(code, signals, baseline, dwt, folder / (code + "_stages.png"))
        except Exception as error:
            record.update(status="failed", error_type=type(error).__name__, error=str(error))
        record["source_video_unchanged"] = digest(video) == before
        record["total_seconds_including_artifacts"] = time.perf_counter() - started
        write_json(folder / (code + ".json"), record)
        summary["records"].append(record)
        write_json(folder / "summary.json", summary)
        print(json.dumps({k:record.get(k) for k in ("data_id","status","pos_bpm","pos_dwt_bpm","error","total_seconds_including_artifacts")}), flush=True)


if __name__ == "__main__":
    main()
