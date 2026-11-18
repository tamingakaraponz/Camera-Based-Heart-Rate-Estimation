"""Compare cheek-only and forehead-plus-cheek POS on matched video windows."""
import csv
import json
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import ippg_v2_refactored as core
import replay_realtime as replay
from replay_realtime import Observation, TimeBuffer, evaluate, load_reference

# Offline interpolation may use the next sensor sample to bracket the final frame.
_reference_window = replay.reference_window
def offline_reference(reference, now, grid, fps, config, max_gap):
    return _reference_window(reference, now + max_gap, grid, fps, config, max_gap)

replay.reference_window = offline_reference

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "outputs/band_ab_20261116_150137_572560/manifest.json"
OUT = ROOT / "outputs/roi_paired_comparison"


def run_clip(clip, config):
    video = Path(clip["video"])
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise ValueError(f"Cannot open {video}")
    fps = capture.get(cv2.CAP_PROP_FPS)
    tracker = core._create_region_tracker(config)
    detector = core.AdaptiveSkinDetector(config)
    reference = load_reference(video.parent / "ground_truth.txt")
    buffers = [TimeBuffer(16), TimeBuffer(16)]
    rows = []
    fig, axes = plt.subplots(4, 3, figsize=(16, 11), layout="constrained")
    specfig, specaxes = plt.subplots(4, 3, figsize=(16, 11), layout="constrained")
    try:
        for end in (20, 30, 40, 50):
            while capture.get(cv2.CAP_PROP_POS_FRAMES) / fps < end:
                ok, frame = capture.read()
                if not ok:
                    raise ValueError(f"Video ended before {end}s")
                n = int(capture.get(cv2.CAP_PROP_POS_FRAMES))
                t = n / fps
                regions = tracker.detect(frame, int(round((n - 1) * 1000 / fps)))
                colors = [None, None]
                if regions is not None:
                    detector.calibrate(regions, t)
                    try:
                        values = []
                        for region, geometry, name in (
                            (regions.forehead, regions.forehead_geometry, "forehead"),
                            (regions.left_cheek, regions.left_cheek_geometry, "left_cheek"),
                            (regions.right_cheek, regions.right_cheek_geometry, "right_cheek"),
                        ):
                            rgb, _, count = core.mean_skin_rgb(region, geometry, name, detector, config)
                            values.append((rgb, count))
                        if detector.ready:
                            colors = [core._weighted_rgb(values[1:]), core._weighted_rgb(values)]
                    except ValueError:
                        pass
                for buffer, rgb in zip(buffers, colors):
                    buffer.append(Observation(t, rgb))
            # Use the last complete frame as the shared endpoint.
            now = buffers[0].last_time
            for j, seconds in enumerate((5, 10, 15)):
                results = [evaluate(b, now, seconds, fps, config, reference) for b in buffers]
                ax = axes[(end // 10) - 2, j]
                sax = specaxes[(end // 10) - 2, j]
                row = dict(subject=clip["subject"], end_s=end, window_s=seconds)
                for name, (result, stages) in zip(("cheeks", "combined"), results):
                    row[name + "_status"] = result["reason"] or result["reference_reason"] or result["status"]
                    row[name + "_bpm"] = result["ippg_bpm"]
                    row[name + "_ppg_bpm"] = result["reference_bpm"]
                    if result["status"] == "estimated" and result["reference_status"] == "estimated":
                        x, y = stages["filtered"], stages["reference_filtered"]
                        if np.std(x) > 1e-12 and np.std(y) > 1e-12:
                            row[name + "_correlation"] = float(np.corrcoef(x, y)[0, 1])
                            frequencies = stages["frequencies_hz"]
                            band = (frequencies >= .6) & (frequencies <= 3)
                            power = stages["power"][band]
                            _, reference_fft = core.calculate_spectrum(stages["time_seconds"], y, fps)
                            reference_power = np.abs(reference_fft[band]) ** 2
                            if power.sum() > 0 and reference_power.sum() > 0:
                                row[name + "_spectral_overlap"] = float(np.minimum(power / power.sum(), reference_power / reference_power.sum()).sum())
                                sax.plot(frequencies[band] * 60, power / power.sum(), label=name)
                                if name == "cheeks":
                                    sax.plot(frequencies[band] * 60, reference_power / reference_power.sum(), label="PPG", alpha=.7)
                            ax.plot(stages["time_seconds"], (x-x.mean())/x.std(), label=name)
                            if name == "cheeks":
                                ax.plot(stages["time_seconds"], (y-y.mean())/y.std(), label="PPG", alpha=.7)
                if all(k + "_correlation" in row for k in ("cheeks", "combined")):
                    a, b = results[0][1], results[1][1]
                    np.testing.assert_allclose(a["time_seconds"], b["time_seconds"])
                    np.testing.assert_allclose(a["reference_filtered"], b["reference_filtered"])
                    for name in ("cheeks", "combined"):
                        row[name + "_error"] = abs(row[name + "_bpm"] - row[name + "_ppg_bpm"])
                ax.set_title(f"{end-seconds}–{end}s")
                sax.set_title(f"{end-seconds}–{end}s")
                sax.grid(alpha=.2)
                ax.grid(alpha=.2)
                rows.append(row)
        axes[0, 0].legend(fontsize=7)
        fig.savefig(OUT / f'{clip["subject"]}_waveforms.png', dpi=120)
        specfig.savefig(OUT / f'{clip["subject"]}_spectra.png', dpi=120)
        return rows
    finally:
        plt.close(fig)
        plt.close(specfig)
        tracker.close()
        capture.release()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    source = json.loads(SOURCE.read_text(encoding="utf8"))
    config = core.IPPGConfig(**source["configs"]["new"])
    all_rows = []
    for clip in source["clips"]:
        print("Running", clip["subject"], flush=True)
        all_rows.extend(run_clip(clip, config))
        fields = list(dict.fromkeys(k for row in all_rows for k in row))
        with (OUT / "windows.csv").open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fields)
            writer.writeheader()
            writer.writerows(all_rows)
        print("Completed", clip["subject"], flush=True)


if __name__ == "__main__":
    main()

