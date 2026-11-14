"""Frame-by-frame, unpaced local replay. No claim of live-camera throughput.

Reuses the unchanged legacy ROI/POS/DWT/filter/FFT functions. Each estimate
restarts processing using only the recent RGB observations already received.
"""
from __future__ import annotations

import argparse
from collections import deque
from dataclasses import asdict, dataclass
from datetime import datetime
from hashlib import file_digest, sha256
import csv
import json
from pathlib import Path
import sys
from time import perf_counter

PROJECT_PYTHON = Path(__file__).resolve().parents[2]/".venv/Scripts/python.exe"
if __name__=="__main__" and PROJECT_PYTHON.is_file() and str(Path(sys.executable).resolve()).casefold()!=str(PROJECT_PYTHON.resolve()).casefold():
    import subprocess
    raise SystemExit(subprocess.run([str(PROJECT_PYTHON),*sys.argv]).returncode)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import cv2
import numpy as np
from scipy.signal import butter, sosfiltfilt

import ippg_v2_refactored as core

DEFAULT_VIDEO = Path(r"C:\Users\Akkarapon\Downloads\subject1-20261104T091024Z-1-001\subject1\vid.avi")


@dataclass(frozen=True)
class Observation:
    time_s: float
    rgb: np.ndarray | None
    status: str = "valid"
    extraction_ms: float = 0.


class TimeBuffer:
    def __init__(self, seconds=15.):
        if not np.isfinite(seconds) or seconds <= 0:
            raise ValueError("Invalid buffer duration")
        self.seconds = seconds
        self.items = deque()
        self.last_time = -np.inf
        self.first_valid_time = None

    def append(self, item):
        if not np.isfinite(item.time_s) or item.time_s <= self.last_time:
            raise ValueError("Observations must have strictly increasing finite times")
        if item.rgb is not None and (np.shape(item.rgb) != (3,) or not np.all(np.isfinite(item.rgb))):
            raise ValueError("RGB must have three finite channels")
        self.last_time = item.time_s
        if item.rgb is not None and self.first_valid_time is None:
            self.first_valid_time = item.time_s
        self.items.append(item)
        self.prune(item.time_s)

    def prune(self, now):
        while self.items and self.items[0].time_s <= now-self.seconds:
            self.items.popleft()

    def select(self, now, seconds):
        if not 0 < seconds <= self.seconds:
            raise ValueError("Window exceeds buffer")
        if self.last_time > now+1e-9:
            raise ValueError("Future observation in buffer")
        self.prune(now)
        return [o for o in self.items if now-seconds < o.time_s <= now+1e-9]


def replay_video(path, config, metadata):
    """Yield ORIGINAL per-frame RGB before any full-clip interpolation."""
    capture = cv2.VideoCapture(str(path))
    tracker = None
    try:
        if not capture.isOpened():
            raise ValueError(f"Cannot open video: {path}")
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        if not np.isfinite(fps) or fps <= 0:
            raise ValueError("Invalid video FPS")
        metadata["fps"] = fps
        tracker = core._create_region_tracker(config)
        metadata["tracker"] = tracker.name
        detector = core.AdaptiveSkinDetector(config)
        while True:
            begun = perf_counter()
            ok, frame = capture.read()
            if not ok:
                break
            frame_number = capture.get(cv2.CAP_PROP_POS_FRAMES)
            t = frame_number/fps
            # Same file-clock convention as the baseline, not a webcam timestamp.
            regions = tracker.detect(frame, int(round((frame_number-1)*1000/fps)))
            rgb, status = None, "face_missing"
            if regions is not None:
                detector.calibrate(regions, t)
                try:
                    values = []
                    for region, geometry, label in (
                        (regions.forehead, regions.forehead_geometry, "forehead"),
                        (regions.left_cheek, regions.left_cheek_geometry, "left_cheek"),
                        (regions.right_cheek, regions.right_cheek_geometry, "right_cheek"),
                    ):
                        value, _, count = core.mean_skin_rgb(region, geometry, label, detector, config)
                        values.append((value, count))
                    combined = core._weighted_rgb(values)
                    if detector.ready:
                        rgb, status = combined, "valid"
                    else:
                        status = "calibrating"
                except ValueError as error:
                    status = f"invalid_skin: {error}"
            yield Observation(t, rgb, status, (perf_counter()-begun)*1000)
    finally:
        capture.release()
        if tracker is not None:
            tracker.close()


def load_reference(path):
    data = np.loadtxt(path, ndmin=2)
    if data.shape[0] != 3 or not np.all(np.isfinite(data)):
        raise ValueError("Reference must contain PPG, supplied HR and timestamp rows")
    t = data[2]-data[2,0]
    if len(t)<2 or np.any(np.diff(t)<0):
        raise ValueError("Invalid reference timestamps")
    return t, data[0]


def reference_window(reference, now, grid, fps, config, max_gap):
    # Although loaded from disk, the estimator is handed only the prefix <=now.
    times, values = reference
    end = np.searchsorted(times, now, side="right")
    start = max(0, np.searchsorted(times[:end], grid[0], side="left")-1)
    t, x = times[start:end], values[start:end]
    unique, inverse, counts = np.unique(t, return_inverse=True, return_counts=True)
    if len(unique)<2 or unique[0]>grid[0]+1e-9 or unique[-1]<grid[-1]-1e-9:
        raise ValueError("reference_support_incomplete")
    if np.max(np.diff(unique))>max_gap:
        raise ValueError("reference_gap")
    x = np.bincount(inverse,weights=x)/counts
    sampled = np.interp(grid,unique,x)
    if np.ptp(sampled)<=1e-12:
        raise ValueError("reference_flat")
    sos = butter(3,[config.forehead_low_hz,config.forehead_high_hz],btype="bandpass",fs=fps,output="sos")
    filtered = sosfiltfilt(sos,sampled-sampled.mean(),padlen=3*(2*len(sos)+1))
    f, fft = core.calculate_spectrum(grid,filtered,fps)
    bpm = core.estimate_heart_rate(f,fft,config.forehead_low_hz,config.forehead_high_hz)
    return bpm, sampled, filtered, float(t[-1])


def evaluate(buffer, now, seconds, fps, config, reference=None, max_gap=.25):
    begun = perf_counter()
    row = dict(window_seconds=seconds,update_time_s=now,status="collecting",reason="insufficient_history",
               ippg_bpm=None,reference_bpm=None,reference_status="not_requested",reference_reason="",
               difference_bpm=None,input_start_s=None,input_end_s=None,valid_samples=0,
               max_gap_s=None,compute_ms=None,reference_compute_ms=None,
               newest_input_age_s=None,reference_last_input_s=None,fft_bin_spacing_bpm=None)
    stages = {}
    observations = buffer.select(now, seconds)
    valid = [o for o in observations if o.rgb is not None]
    if buffer.first_valid_time is not None and now-buffer.first_valid_time >= seconds-1/fps-1e-8:
        row.update(status="unavailable",reason="insufficient_recent_support")
    if valid:
        t = np.array([o.time_s for o in valid])
        row.update(input_start_s=float(t[0]),input_end_s=float(t[-1]),valid_samples=len(valid),
                   newest_input_age_s=float(now-t[-1]))
        # One frame tolerance at the start; no pretending missing startup is data.
        if t[0] <= now-seconds+1/fps+1e-8 and len(t)>=2:
            gap = float(max(t[0]-(now-seconds),now-t[-1],np.max(np.diff(t))))
            row["max_gap_s"] = gap
            if gap>max_gap:
                row.update(status="unavailable",reason="input_gap")
            else:
                try:
                    grid, rgb = core.resample_to_fps(t,np.array([o.rgb for o in valid]),fps)
                    assert grid[-1]<=now+1e-8 and grid[0]>=now-seconds-1e-8
                    pulse = core.calculate_pos(rgb,fps,config.pos_window_seconds,config.pos_epsilon)
                    processed = core.process_pulse(pulse,fps,"pos",config)
                    f, fft = core.calculate_spectrum(grid,processed.filtered,fps)
                    bpm = core.estimate_heart_rate(f,fft,config.forehead_low_hz,config.forehead_high_hz)
                    row.update(status="estimated",reason="",ippg_bpm=bpm,fft_bin_spacing_bpm=float((f[1]-f[0])*60))
                    stages=dict(time_seconds=grid,rgb=rgb,raw_pos=pulse,after_dwt=processed.reconstructed,
                                filtered=processed.filtered,frequencies_hz=f,power=np.abs(fft)**2)
                except ValueError as error:
                    row.update(status="unavailable",reason=str(error))
    row["compute_ms"] = (perf_counter()-begun)*1000
    if row["status"]=="estimated" and reference is not None:
        ref_begun = perf_counter()
        try:
            bpm, raw, filtered, last = reference_window(reference,now,stages["time_seconds"],fps,config,max_gap)
            row.update(reference_bpm=bpm,reference_status="estimated",reference_last_input_s=last,
                       difference_bpm=row["ippg_bpm"]-bpm)
            stages.update(reference_raw=raw,reference_filtered=filtered)
        except ValueError as error:
            row.update(reference_status="unavailable",reference_reason=str(error))
        row["reference_compute_ms"] = (perf_counter()-ref_begun)*1000
    return row, stages


class ReplayEngine:
    """Schedule ticks BEFORE accepting a frame from after that tick."""
    def __init__(self, fps, config, reference=None, windows=(5,10,15), step=1., max_gap=.25):
        if not np.isfinite(fps) or fps<=0 or not np.isfinite(step) or step<=0:
            raise ValueError("Invalid FPS or step")
        if not windows or any(not np.isfinite(w) or w<=0 for w in windows) or not np.isfinite(max_gap) or max_gap<=0:
            raise ValueError("Invalid windows or gap")
        self.fps,self.config,self.reference=fps,config,reference
        self.windows,self.step,self.max_gap=tuple(windows),step,max_gap
        self.buffer=TimeBuffer(max(windows))
        self.next_tick=None
        self.rows=[]
        self.stages={}
        self.max_buffer_samples=0

    def _tick(self):
        for seconds in self.windows:
            row, arrays=evaluate(self.buffer,self.next_tick,seconds,self.fps,self.config,self.reference,self.max_gap)
            index=len(self.rows)
            self.rows.append(row)
            for name,values in arrays.items():
                self.stages[f"row{index:04d}_{name}"]=values
        self.next_tick+=self.step

    def push(self, observation):
        if self.next_tick is None:
            self.next_tick=observation.time_s+self.step
        while self.next_tick < observation.time_s-1e-9:
            self._tick()
        self.buffer.append(observation)
        self.max_buffer_samples=max(self.max_buffer_samples,len(self.buffer.items))
        if abs(self.next_tick-observation.time_s)<=1e-9:
            self._tick()


def summarize(rows, windows):
    result=[]
    common={r["update_time_s"] for r in rows if r["window_seconds"]==max(windows) and r["difference_bpm"] is not None}
    for w in windows:
        own=[r for r in rows if r["window_seconds"]==w]
        estimated=[r for r in own if r["status"]=="estimated"]
        paired=[r for r in estimated if r["difference_bpm"] is not None]
        # Full shared successful times; no mismatched denominators across windows.
        shared=[r for r in paired if r["update_time_s"] in common and all(
            any(q["window_seconds"]==v and q["update_time_s"]==r["update_time_s"] and q["difference_bpm"] is not None for q in rows) for v in windows)]
        def mae(data):
            return float(np.mean([abs(r["difference_bpm"]) for r in data])) if data else None
        ms=[r["compute_ms"] for r in estimated]
        result.append(dict(window_seconds=w,requested_ticks=len(own),collecting_ticks=sum(r["status"]=="collecting" for r in own),
            estimated_ticks=len(estimated),unavailable_ticks=sum(r["status"]=="unavailable" for r in own),
            reference_paired_ticks=len(paired),first_estimate_time_s=estimated[0]["update_time_s"] if estimated else None,
            paired_MAE_bpm=mae(paired),shared_paired_ticks=len(shared),shared_MAE_bpm=mae(shared),
            paired_max_error_bpm=max((abs(r["difference_bpm"]) for r in paired),default=None),
            paired_p95_error_bpm=float(np.percentile([abs(r["difference_bpm"]) for r in paired],95)) if paired else None,
            compute_median_ms=float(np.median(ms)) if ms else None,compute_p95_ms=float(np.percentile(ms,95)) if ms else None,
            compute_max_ms=max(ms) if ms else None,
            median_abs_step_bpm=float(np.median([abs(b["ippg_bpm"]-a["ippg_bpm"]) for a,b in zip(own,own[1:])
                if a["ippg_bpm"] is not None and b["ippg_bpm"] is not None])) if sum(a["ippg_bpm"] is not None and b["ippg_bpm"] is not None for a,b in zip(own,own[1:])) else None))
    return result


def digest(path):
    with Path(path).open("rb") as stream:
        return file_digest(stream,"sha256").hexdigest()


def write_table(path, rows):
    with path.open("w",newline="",encoding="utf-8-sig") as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def make_plot(rows, summaries, path):
    fig, axes=plt.subplots(3,1,figsize=(13,11),layout="constrained")
    for summary in summaries:
        w=summary["window_seconds"]
        data=[r for r in rows if r["window_seconds"]==w]
        t=[r["update_time_s"] for r in data]
        y=[r["ippg_bpm"] if r["ippg_bpm"] is not None else np.nan for r in data]
        line,=axes[0].plot(t,y,label=f"iPPG {w}s")
        axes[0].plot(t,[r["reference_bpm"] if r["reference_bpm"] is not None else np.nan for r in data],
                     ls="--",color=line.get_color(),alpha=.65,label=f"Reference {w}s")
        axes[1].plot(t,[abs(r["difference_bpm"]) if r["difference_bpm"] is not None else np.nan for r in data],label=f"{w}s")
        axes[2].plot(t,[r["compute_ms"] if r["status"]=="estimated" else np.nan for r in data],label=f"{w}s")
    axes[0].set(title="Rolling estimates: solid = iPPG, dashed = reference; gaps = no estimate",ylabel="Heart rate (BPM)")
    axes[1].set(title="Nominal-clock paired error (synchronization unverified)",ylabel="Absolute difference (BPM)")
    axes[2].set(title="RGB-window estimation only; excludes face tracking / browser / reference",ylabel="Compute time (ms)")
    for ax in axes:
        ax.grid(alpha=.25)
        ax.legend(loc="upper right",ncol=3)
        ax.set_xlabel("Video time (s), file frame/FPS clock")
    fig.suptitle("5 / 10 / 15-second replay | past data only | unpaced Python experiment")
    fig.savefig(path,dpi=130)
    plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video",nargs="?",type=Path,default=DEFAULT_VIDEO)
    parser.add_argument("--reference",type=Path)
    parser.add_argument("--output-root",type=Path,default=Path(__file__).resolve().parents[1]/"outputs/realtime_replay")
    args=parser.parse_args()
    video=args.video.resolve()
    refpath=(args.reference or video.parent/"ground_truth.txt").resolve()
    if not video.is_file() or not refpath.is_file():
        parser.error("Video and reference must exist")
    config=core.IPPGConfig(max_duration_seconds=None,show_preview=False,pulse_method="pos",roi_method="legacy",denoise="dwt",background_method="none")
    output=args.output_root.resolve()/datetime.now().strftime("replay_%Y%m%d_%H%M%S_%f")
    output.mkdir(parents=True,exist_ok=False)
    source_info={str(p):dict(sha256=digest(p),size_bytes=p.stat().st_size,mtime_ns=p.stat().st_mtime_ns) for p in (video,refpath)}
    info=dict(status="running",python=sys.executable,mode="unpaced file replay; NOT webcam or paced throughput",source=source_info,
              config=asdict(config),windows_s=[5,10,15],update_step_s=1.,max_gap_s=.25,
              quality_policy="Only support/gap/flat checks; thresholds provisional, no calibrated confidence",
              clock="Baseline-compatible frame_number/FPS; not hardware synchronization",
              reference_policy="Prefix timestamps <= update time only; duplicate averaging; same available grid and FPS; order-3 SOS; no full-clip filtering",
              future_policy="Tick before accepting any frame after it; no full-clip RGB interpolation or filtering",
              reference_synchronization_verified=False)
    manifest=output/"manifest.json"
    def save_info():
        manifest.write_text(json.dumps(info,ensure_ascii=False,indent=2,default=str,allow_nan=False),encoding="utf-8")
    save_info()
    try:
        reference=load_reference(refpath)
        metadata={}
        frames=[]
        engine=None
        started=perf_counter()
        observations=replay_video(video,config,metadata)
        try:
            for observation in observations:
                if engine is None:
                    engine=ReplayEngine(metadata["fps"],config,reference)
                engine.push(observation)
                rgb=observation.rgb if observation.rgb is not None else [None]*3
                frames.append(dict(time_s=observation.time_s,status=observation.status,R=rgb[0],G=rgb[1],B=rgb[2],extraction_ms=observation.extraction_ms))
                if len(frames)%300==0:
                    print(f"Frames {len(frames)}, video time {observation.time_s:.1f}s, updates {len(engine.rows)}",flush=True)
        finally:
            observations.close()
        if engine is None or not engine.rows:
            raise ValueError("No replay updates")
        info.update(metadata,replay_wall_s=perf_counter()-started,frame_count=len(frames),
            max_buffer_samples=engine.max_buffer_samples,summaries=summarize(engine.rows,engine.windows),
            frame_extraction_p95_ms=float(np.percentile([r["extraction_ms"] for r in frames],95)),
            code_sha256={p.name:digest(p) for p in (Path(__file__),Path(core.__file__))})
        model=config.face_landmarker_model_path or Path(__file__).resolve().parents[1]/"models/face_landmarker.task"
        info["face_model_sha256"]=digest(model) if "MediaPipe" in metadata["tracker"] and Path(model).is_file() else None
        info["reference_unavailable_reasons"]={reason:sum(r["reference_reason"]==reason for r in engine.rows)
            for reason in sorted({r["reference_reason"] for r in engine.rows}) if reason}
        write_table(output/"observations.csv",frames)
        write_table(output/"updates.csv",engine.rows)
        write_table(output/"summary.csv",info["summaries"])
        np.savez_compressed(output/"window_stages.npz",**engine.stages)
        make_plot(engine.rows,info["summaries"],output/"comparison.png")
        text="# ผลทดลองข้อมูลทยอยเข้า 5/10/15 วินาที\n\n"
        text+="รัน Python ในเครื่องจากคลิปทีละเฟรม ไม่ใช่เว็บหรือกล้องสด และไม่ได้ป้อนตามความเร็วจริง\n\n"
        text+="| หน้าต่าง | ผลแรกที่เวลาคลิป | คำนวณได้/รอบที่ขอ | จับคู่ reference | MAE บนเวลาร่วม | เวลา P95 ต่อรอบ |\n|---|---|---|---|---|---|\n"
        for s in info["summaries"]:
            text+=f"| {s['window_seconds']} s | {s['first_estimate_time_s']} s | {s['estimated_ticks']}/{s['requested_ticks']} | {s['reference_paired_ticks']} | {s['shared_MAE_bpm']} BPM | {s['compute_p95_ms']} ms |\n"
        text+="\nMAE ใช้เฉพาะรอบเวลาร่วมที่ทั้งสามแบบและ reference คำนวณได้ รายงานจำนวนไว้ใน summary.csv ไม่ใช่ความแม่นยำทั่วไป "
        text+="ช่วงเริ่มเก็บข้อมูลรวมในจำนวนรอบที่ขอ; reference ใช้เวลาเดิมที่ยังไม่ยืนยัน synchronization และตัวกรองต่าง order จาก iPPG\n\n"
        text+="ผลแรกเป็นเวลาตามคลิป ไม่ใช่เวลาเริ่มกล้องจริง; เวลา P95 รวมเฉพาะเตรียมหน้าต่างและประมาณ iPPG ไม่รวมตรวจหน้า "
        text+="มีตรวจข้อมูลขาดสูงสุด 0.25 วินาทีและสัญญาณแบน แต่ยังไม่ใช่ quality gate ครบ ไม่มีการเลื่อน/ยืดเวลา กลับขั้ว หรือ smoothing BPM\n\n"
        text+="บางรอบ reference ไม่มีจุดที่มาถึงแล้วครอบคลุม grid ทั้งช่วง จึงไม่ extrapolate และไม่นำรอบนั้นไปคิด error; จำนวนและเหตุผลอยู่ใน manifest/updates "
        text+="ดู MAE ทุกคู่และความผิดพลาดสูงสุดใน summary.csv ควบคู่กับ MAE บนเวลาร่วม เพราะเวลาร่วมตัดช่วงเริ่มต้นบางส่วนออก "
        text+="buffer ที่ใช้ประมาณสัญญาณจำกัด 15 วินาที แต่ไฟล์บันทึก/สัญญาณตรวจสอบสะสมทั้งการทดลอง โหมดนี้ยังไม่ใช่ระบบวัดต่อเนื่องที่ควบคุมหน่วยความจำทุกส่วน\n\n"
        text+="observations.csv เก็บ RGB รายเฟรมก่อน interpolation; updates.csv รวม collecting และ unavailable; "
        text+="window_stages.npz เก็บสัญญาณทุกขั้นของแต่ละแถวที่คำนวณได้ ตรวจความแม่นยำกับหลายคนและกล้องจริงต่อก่อนเลือกช่วงใช้งาน\n"
        (output/"RESULT.md").write_text(text,encoding="utf-8")
        for p in (video,refpath):
            if p.stat().st_size!=source_info[str(p)]["size_bytes"] or p.stat().st_mtime_ns!=source_info[str(p)]["mtime_ns"]:
                raise RuntimeError("Source changed during replay")
        info["files"]={p.name:digest(p) for p in output.iterdir() if p!=manifest}
        info["status"]="complete"
        save_info()
        print(json.dumps(info["summaries"],indent=2))
        print(f"Saved replay: {output}")
    except Exception as error:
        info.update(status="failed",error=str(error))
        save_info()
        raise


if __name__=="__main__":
    main()
