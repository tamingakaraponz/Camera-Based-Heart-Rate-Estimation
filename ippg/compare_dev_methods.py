"""Exploratory POS/CHROM comparison on the new development split only."""
import csv
from dataclasses import replace
from datetime import datetime
import hashlib
import json
from pathlib import Path
from time import perf_counter

import numpy as np
from scipy.signal import butter, filtfilt, windows
import ippg_v2_refactored as core
from benchmark_new_dataset import ROOT, development_inputs
from replay_realtime import TimeBuffer, evaluate, replay_video


def chrom(rgb, fps):
    """CHROM overlap-add, based on de Haan/Jeanne; trailing partial window omitted."""
    length=int(np.ceil(1.6*fps))
    length+=length%2
    half=length//2
    if len(rgb)<length:
        raise ValueError("insufficient_chrom_samples")
    b,a=butter(3,[.7,2.5],btype="bandpass",fs=fps)
    output=np.zeros(len(rgb))
    stop=0
    for start in range(0,len(rgb)-length+1,half):
        part=rgb[start:start+length]
        mean=part.mean(axis=0)
        if np.any(np.abs(mean)<1e-8):
            raise ValueError("invalid_chrom_rgb")
        normalized=part/mean
        x=3*normalized[:,0]-2*normalized[:,1]
        y=1.5*normalized[:,0]+normalized[:,1]-1.5*normalized[:,2]
        x,y=filtfilt(b,a,x),filtfilt(b,a,y)
        if np.std(y)<1e-12:
            raise ValueError("flat_chrom_projection")
        output[start:start+length]+=(x-np.std(x)/np.std(y)*y)*windows.hann(length)
        stop=start+length
    return output[:stop]


def reference_hr(reference, start, now):
    """Mean of provided HR values on nominal timestamps; independent of estimator."""
    t,hr=reference
    mask=(t>=start)&(t<=now)
    t,hr=t[mask],hr[mask]
    if len(t)<2 or t[0]>start+.1 or t[-1]<now-.1 or np.max(np.diff(t))>.25:
        return None
    unique,inverse,counts=np.unique(t,return_inverse=True,return_counts=True)
    hr=np.bincount(inverse,weights=hr)/counts
    if not np.all(np.isfinite(hr)) or np.any(hr<=0):
        return None
    return float(np.mean(hr))


def run_subject(item, writer, output_stream):
    cfg=core.IPPGConfig(max_duration_seconds=None,show_preview=False,pulse_method="pos",roi_method="legacy",denoise="dwt",background_method="none")
    data=np.loadtxt(item["reference"],ndmin=2)
    if data.shape[0]!=3 or not np.all(np.isfinite(data)) or np.any(np.diff(data[2])<0):
        raise ValueError("invalid reference")
    reference=(data[2]-data[2,0],data[1])
    meta={}; buffer=TimeBuffer(15); tick=None; frames=0; counts={}
    stream=replay_video(Path(item["video"]),cfg,meta)
    try:
        for observation in stream:
            frames+=1
            if tick is None:
                tick=observation.time_s+1
            while tick < observation.time_s-1e-9:
                for duration in (5,10,15):
                    row,stages=evaluate(buffer,tick,duration,meta["fps"],cfg)
                    ref=reference_hr(reference,tick-duration,tick)
                    estimates={"POS":(row["status"],row["reason"],row["ippg_bpm"],row["compute_ms"])}
                    # CHROM can succeed even when POS fails; share only input-support checks.
                    if row["max_gap_s"] is not None and row["max_gap_s"]<=.25:
                        began=perf_counter()
                        try:
                            valid=[o for o in buffer.select(tick,duration) if o.rgb is not None]
                            grid,rgb=core.resample_to_fps(np.array([o.time_s for o in valid]),np.array([o.rgb for o in valid]),meta["fps"])
                            pulse=chrom(rgb,meta["fps"])
                            # Same external DWT + bandpass + FFT as POS, without relabeling CHROM as POS.
                            reconstructed,_=core.wavelet_denoise(pulse,cfg,meta["fps"])
                            filtered=core.bandpass_filter(reconstructed,meta["fps"],cfg.forehead_low_hz,cfg.forehead_high_hz,cfg.forehead_filter_order)
                            f,spectrum=core.calculate_spectrum(grid[:len(filtered)],filtered,meta["fps"])
                            bpm=core.estimate_heart_rate(f,spectrum,cfg.forehead_low_hz,cfg.forehead_high_hz)
                            estimates["CHROM"]=("estimated","",bpm,(perf_counter()-began)*1000)
                        except (ValueError,RuntimeError) as error:
                            estimates["CHROM"]=("unavailable",str(error),None,(perf_counter()-began)*1000)
                    else:
                        estimates["CHROM"]=(row["status"],row["reason"],None,0.)
                    for method,(status,reason,bpm,ms) in estimates.items():
                        result=dict(subject=item["subject"],update_time_s=tick,window_seconds=duration,method=method,
                            status=status,reason=reason,bpm=bpm,reference_bpm=ref,
                            difference_bpm=bpm-ref if bpm is not None and ref is not None else None,compute_ms=ms)
                        writer.writerow(result)
                        counts[status]=counts.get(status,0)+1
                output_stream.flush()
                tick+=1
            buffer.append(observation)
    finally:
        stream.close()
    capture_expected=cv_expected(item["video"])
    return dict(subject=item["subject"],frames_read=frames,expected_frames=capture_expected,
                complete=frames==capture_expected,tracker=meta.get("tracker"),counts=counts)


def cv_expected(path):
    import cv2
    c=cv2.VideoCapture(path)
    try:
        return int(c.get(cv2.CAP_PROP_FRAME_COUNT))
    finally:
        c.release()


def main():
    inputs=development_inputs(ROOT/"new_dataset_split.json")
    out=ROOT/"outputs/dev_method_comparison"/datetime.now().strftime("comparison_%Y%m%d_%H%M%S_%f")
    out.mkdir(parents=True,exist_ok=False)
    fields=["subject","update_time_s","window_seconds","method","status","reason","bpm","reference_bpm","difference_bpm","compute_ms"]
    manifest=dict(status="running",subjects=[x["subject"] for x in inputs],held_out_opened=False,
        reference="Mean supplied HR row on nominal timestamps in window; same for all methods; not verified synchronized",
        max_gap_s=.25,chrom="CHROM 1.6s, 50% overlap, 0.7-2.5Hz internal filter, Hann, DWT and 0.6-2Hz external filter; trailing partial omitted",
        note="Exploratory development run; supplied device HR smoothing unknown; no pretrained model or live web",
        source_sha256={Path(__file__).name:hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})
    def save():
        (out/"manifest.json").write_text(json.dumps(manifest,indent=2,ensure_ascii=False),encoding="utf-8")
    save()
    try:
        with (out/"results.csv").open("w",newline="",encoding="utf-8-sig") as stream:
            writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader();stream.flush()
            completed=[]
            for item in inputs:
                print(f"Processing {item['subject']}",flush=True)
                completed.append(run_subject(item,writer,stream));stream.flush()
                manifest["completed"]=completed;save()
        manifest["status"]="complete" if all(x["complete"] for x in completed) else "incomplete_decode"
        save()
        print(f"Saved results: {out}",flush=True)
    except Exception as error:
        manifest.update(status="failed",error=str(error));save();raise


if __name__=="__main__":
    main()
