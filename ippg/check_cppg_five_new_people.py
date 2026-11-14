"""Fixed-factor exploratory check on subjects 2-6, all three trials each.

Runs the unchanged POS + band-pass pipeline with the first-three-demo settings.
The reference BPM multiplier is fixed in advance and is never fitted per person.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
import importlib.metadata
from pathlib import Path
import platform
import time
import json

import numpy as np

import ippg_v2_refactored as core
from benchmark_background_methods import analyze_contact_ppg
from run_first_three_demo import digest, write_json


ROOT = Path(__file__).resolve().parents[1]
SUBJECTS = (2, 3, 4, 5, 6)
FACTOR = 34.0 / 32.0
CONFIG = core.IPPGConfig(max_duration_seconds=34.333, pulse_method="pos", denoise="none",
                        roi_method="legacy", background_method="none", show_preview=False)


def run_one(code, folder):
    started = time.perf_counter()
    video = ROOT / "DATA/temp" / (code + ".avi")
    contact = ROOT / "DATAcppg" / (code + ".xlsx")
    before = {"video": digest(video), "cppg": digest(contact)}
    record = {"data_id": code, "subject": int(code)//10, "trial": int(code)%10,
              "source_hashes": before, "status": "failed"}
    try:
        signals = core.extract_video_signals(video, CONFIG)
        result = core.analyze_signals(signals, CONFIG)
        reference = analyze_contact_ppg(contact, 0.6, 2.0, filter_order=3)
        ippg = float(result.heart_rate_bpm)
        raw = float(reference.heart_rate_bpm)
        scaled = raw * FACTOR
        difference_before = abs(ippg-raw)
        difference_after = abs(ippg-scaled)
        reduction = difference_before - difference_after
        relation = "closer" if reduction > 1e-9 else "farther" if reduction < -1e-9 else "tie"
        record.update(status="success", ippg_bpm=ippg, cppg_original_bpm=raw,
                      cppg_multiplied_bpm=scaled, absolute_difference_before_bpm=difference_before,
                      absolute_difference_after_bpm=difference_after, reduction_bpm=reduction,
                      outcome=relation, fps=signals.fps, samples=len(signals.time_seconds),
                      analysis_start_s=float(signals.time_seconds[0]), analysis_end_s=float(signals.time_seconds[-1]),
                      cppg_duration_s=float(reference.time_seconds[-1]-reference.time_seconds[0]))
        np.savez_compressed(folder / (code+"_signals.npz"), time_seconds=signals.time_seconds,
                            rgb=signals.combined_rgb, raw_pos=signals.pulse_signal,
                            filtered_pos=result.filtered_pulse_signal, frequencies_hz=result.frequencies_hz,
                            pos_power=result.compensated_spectrum)
    except Exception as error:
        record.update(error_type=type(error).__name__, error=str(error))
    record["source_files_unchanged"] = {"video":digest(video)==before["video"], "cppg":digest(contact)==before["cppg"]}
    record["runtime_seconds"] = time.perf_counter()-started
    write_json(folder / (code+".json"),record)
    return record


def aggregate(rows):
    valid = [r for r in rows if r["status"] == "success"]
    return {"attempted":len(rows), "success":len(valid), "failed":len(rows)-len(valid),
            "closer":sum(r["outcome"]=="closer" for r in valid),
            "farther":sum(r["outcome"]=="farther" for r in valid),
            "tie":sum(r["outcome"]=="tie" for r in valid),
            "mean_absolute_difference_before_bpm":float(np.mean([r["absolute_difference_before_bpm"] for r in valid])) if valid else None,
            "mean_absolute_difference_after_bpm":float(np.mean([r["absolute_difference_after_bpm"] for r in valid])) if valid else None}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output",type=Path,required=True)
    args = parser.parse_args()
    folder = args.output.resolve()
    folder.mkdir(parents=True,exist_ok=False)
    codes = [str(s*10+t) for s in SUBJECTS for t in (1,2,3)]
    summary = {"purpose":"exploratory fixed-factor numerical agreement check, not accuracy validation",
               "subjects_selected_before_run":SUBJECTS, "scheduled_data_ids":codes,
               "selection_rule":"next five subjects after subject 1; all 3 trials, no outcome-based selection",
               "factor":FACTOR,"factor_formula":"34/32", "factor_fitted":False,
               "config":asdict(CONFIG), "core_sha256":digest(Path(core.__file__)),
               "model_sha256":digest(ROOT/"models/face_landmarker.task"), "python":platform.python_version(),
               "versions":{p:importlib.metadata.version(p) for p in ("numpy","scipy","opencv-contrib-python","mediapipe")},
               "reference_status":"reference_time_unverified",
               "scope":"iPPG accepted interval within first 34.333 video seconds; cPPG full workbook; no alignment",
               "caveats":["Subjects 2-6 are from the approximately 36-second recording group, unlike subject 1; 34/32 is a fixed what-if, not a justified correction for this group.",
                          "Only calculated cPPG BPM is multiplied, not amplitudes or timestamps; no band-pass recalculation on altered timestamps.",
                          "Closer values alone do not confirm correction or device accuracy.",
                          "No DWT or ICA in iPPG, matching the POS + band-pass comparator from the first-three experiment."],
               "records":[]}
    write_json(folder/"summary.json",summary)
    with ProcessPoolExecutor(max_workers=2) as executor:
        futures = {executor.submit(run_one,code,folder):code for code in codes}
        for future in as_completed(futures):
            code = futures[future]
            try:
                record = future.result()
            except Exception as error:
                record = {"data_id":code,"subject":int(code)//10,"trial":int(code)%10,"status":"failed","error":str(error)}
            summary["records"].append(record)
            summary["records"].sort(key=lambda r:int(r["data_id"]))
            summary["overall"] = aggregate(summary["records"])
            summary["per_subject"] = {str(s):aggregate([r for r in summary["records"] if r["subject"]==s]) for s in SUBJECTS}
            write_json(folder/"summary.json",summary)
            print(json.dumps({k:record.get(k) for k in ("data_id","status","ippg_bpm","cppg_original_bpm","cppg_multiplied_bpm","outcome","error")}),flush=True)
    print("FINISHED " + json.dumps(summary["overall"]),flush=True)


if __name__ == "__main__":
    main()
