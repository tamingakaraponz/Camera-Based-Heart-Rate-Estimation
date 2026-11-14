"""Local, exploratory comparison of two saved signal runs; no pipeline changes."""
import argparse
import csv
from datetime import datetime
import hashlib
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def standardize(x):
    x = np.asarray(x, dtype=float)
    if not np.all(np.isfinite(x)) or np.std(x) < 1e-12:
        raise ValueError("Nonfinite or flat signal cannot be compared")
    return (x - x.mean()) / x.std()


def agreement(x, y):
    x, y = standardize(x), standardize(y)
    return {"pearson_r": float(np.mean(x*y)),
            "normalized_rmse": float(np.sqrt(np.mean((x-y)**2)))}


def read_run(folder, kind):
    folder = Path(folder).resolve()
    m = json.loads((folder/"manifest.json").read_text(encoding="utf-8"))
    if m["status"] != "complete" or m["kind"] != kind:
        raise ValueError(f"Not a complete {kind} run: {folder}")
    # Verify saved artifacts against the producing run's manifest.
    for name, expected in m["files"].items():
        actual = hashlib.sha256((folder/name).read_bytes()).hexdigest()
        if actual != expected["sha256"]:
            raise ValueError(f"Saved artifact hash mismatch: {folder/name}")
    with np.load(folder/"signals.npz", allow_pickle=False) as data:
        t, x = data["time_seconds"].copy(), data["filtered_signal"].copy()
    if t.ndim != 1 or x.shape != t.shape or len(t) < 3 or not np.all(np.isfinite(t)) or not np.all(np.diff(t)>0):
        raise ValueError("Invalid time/signal arrays")
    standardize(x)
    return folder, m, t, x


def estimate_lag(t, x, rt, rx, fs, bound):
    # Every candidate uses precisely the same central interval. Positive offset
    # compares iPPG(t) to PPG(t+offset), i.e. moves PPG left when plotting.
    start, end = max(t[0], rt[0]+bound), min(t[-1], rt[-1]-bound)
    grid = np.arange(start, end, 1/fs)
    if len(grid) < fs*5:
        raise ValueError("Too little overlap for bounded lag exploration")
    a = np.interp(grid, t, x)
    candidates = np.arange(-int(np.floor(bound*fs)), int(np.floor(bound*fs))+1)/fs
    scores = np.array([agreement(a, np.interp(grid+d, rt, rx))["pearson_r"] for d in candidates])
    best = int(np.argmax(scores))
    lag = float(candidates[best])
    before = agreement(a, np.interp(grid, rt, rx))
    after = agreement(a, np.interp(grid+lag, rt, rx))
    return grid, a, lag, before, after, candidates, scores


def fft_hr(x, fs, low=.6, high=2.):
    standardize(x)
    f = np.fft.rfftfreq(len(x), d=1/fs)
    power = np.abs(np.fft.rfft(x-np.mean(x)))**2
    valid = np.flatnonzero((f>=low)&(f<=high))
    if not len(valid) or max(power[valid]) <= 1e-12:
        raise ValueError("No FFT peak")
    return float(f[valid[np.argmax(power[valid])]]*60)


def write_csv(path, names, rows):
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        w=csv.writer(stream)
        w.writerow(names)
        w.writerows(rows)


def compare(ippg_folder, reference_folder, output_root):
    ip, im, it, ix = read_run(ippg_folder, "ippg")
    rp, rm, rt, rx = read_run(reference_folder, "reference_ppg")
    if Path(im["source"]["path"]).parent != Path(rm["source"]["path"]).parent:
        raise ValueError("Different recording directories: explicitly verify the pair before comparing")
    fs = min(im["metadata"]["sampling_rate_hz"], rm["metadata"]["sampling_rate_hz"])
    # Fixed before inspecting correlation: original overlap, +/-1s lag,
    # 20s windows / 10s step; zoom begins 5s into the original overlap.
    start, end = max(it[0], rt[0]), min(it[-1], rt[-1])
    t = np.arange(start, end, 1/fs)
    x, y = np.interp(t,it,ix), np.interp(t,rt,rx)
    full = agreement(x,y)
    ct, cx, lag, before, after, offsets, scores = estimate_lag(it,ix,rt,rx,fs,1.)
    window_n, step_n = int(round(20*fs)), int(round(10*fs))
    rows=[]
    for k in range(0,len(t)-window_n+1,step_n):
        a,b=x[k:k+window_n], y[k:k+window_n]
        ih,rh=fft_hr(a,fs),fft_hr(b,fs)
        rows.append([t[k],t[k+window_n-1],ih,rh,ih-rh,agreement(a,b)["pearson_r"]])
    if not rows:
        raise ValueError("No complete 20-second windows")
    differences=np.asarray(rows)[:,4]
    output=Path(output_root).resolve()/datetime.now().strftime("comparison_%Y%m%d_%H%M%S_%f")
    output.mkdir(parents=True,exist_ok=False)
    write_csv(output/"paired_signals.csv",["time_s","ippg_filtered","reference_filtered","ippg_z","reference_z"],zip(t,x,y,standardize(x),standardize(y)))
    cy=np.interp(ct,rt,rx)
    shifted=np.interp(ct+lag,rt,rx)
    write_csv(output/"lag_exploration.csv",["time_s","ippg_z","reference_original_z","reference_shifted_z"],zip(ct,standardize(cx),standardize(cy),standardize(shifted)))
    write_csv(output/"lag_candidates.csv",["reference_query_offset_s","pearson_r"],zip(offsets,scores))
    write_csv(output/"bpm_windows.csv",["start_s","last_sample_s","ippg_bpm","reference_bpm","difference_bpm","waveform_r_original_time"],rows)
    result={"status":"complete","exploratory_single_recording":True,"python_executable":sys.executable,
        "ippg_run":str(ip),"reference_run":str(rp),"input_manifest_sha256":{str(p):hashlib.sha256((p/"manifest.json").read_bytes()).hexdigest() for p in (ip,rp)},
        "script_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "time_basis":"Original saved time axes, linear interpolation, no extrapolation; synchronization unverified",
        "polarity_flip":False,"time_scaling":False,"filter_changes":False,"comparison_fs_hz":fs,
        "original_overlap_s":[float(t[0]),float(t[-1])],"sample_count":len(t),"original_overlap_metrics":full,
        "common_overlap_fft_bpm":{"ippg":fft_hr(x,fs),"reference":fft_hr(y,fs)},
        "lag_exploration":{"bound_s":1.,"candidate_step_s":1/fs,"offset_s":lag,
            "offset_definition":"iPPG(t) vs reference(t+offset); positive moves reference left",
            "evaluation_interval_s":[float(ct[0]),float(ct[-1])],"before":before,"after":after,
            "maximum_at_search_boundary":bool(np.argmax(scores) in (0,len(scores)-1)),
            "warning":"Fit on this same recording; cycle ambiguity possible; not hardware synchronization or held-out accuracy"},
        "bpm_windows":{"length_samples":window_n,"actual_duration_s":window_n/fs,"step_samples":step_n,"count":len(rows),
            "overlap":True,"fft_bin_spacing_bpm":fs/window_n*60,"MAE_bpm":float(np.mean(abs(differences))),
            "RMSE_bpm":float(np.sqrt(np.mean(differences**2))),"bias_bpm":float(np.mean(differences)),
            "warning":"Overlapping windows from one person, not independent subjects; no confidence interval or accuracy percent"},
        "limitations":["Video frame/FPS and stored reference timestamps use different time bases",
            "iPPG order-6 b/a filter and reference order-3 SOS filter differ; morphology comparison is conditional",
            "No beat-level, HRV, clinical accuracy or population validation claim",
            "No new quality gate: completed windows are not certified clean measurements"]}
    (output/"metrics.json").write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False),encoding="utf-8")

    fig,ax=plt.subplots(4,1,figsize=(13,12),layout="constrained")
    ax[0].plot(t,standardize(x),label="iPPG",lw=.9)
    ax[0].plot(t,standardize(y),label="Reference PPG",lw=.9,alpha=.8)
    ax[0].set(title=f"Original time axes | full-overlap r = {full['pearson_r']:.3f}",ylabel="Z-score (a.u.)")
    zstart,zend=start+5,start+10
    ax[1].plot(t,standardize(x),label="iPPG")
    ax[1].plot(t,standardize(y),label="Reference PPG",alpha=.8)
    ax[1].set(xlim=(zstart,zend),title="Original time axes | fixed 5-second zoom",ylabel="Z-score (a.u.)")
    ax[2].plot(ct,standardize(cx),label="iPPG")
    ax[2].plot(ct,standardize(shifted),label=f"Reference query offset {lag:+.3f}s",alpha=.8)
    ax[2].set(xlim=(zstart,zend),title=f"Exploratory offset fit | SAME interval r: {before['pearson_r']:.3f} -> {after['pearson_r']:.3f}",ylabel="Z-score (a.u.)")
    r=np.array(rows)
    centers=(r[:,0]+r[:,1])/2
    ax[3].plot(centers,r[:,2],"o-",label="iPPG FFT")
    ax[3].plot(centers,r[:,3],"s--",label="Reference FFT")
    ax[3].set(title=f"20-second windows, 10-second step | MAE {np.mean(abs(differences)):.2f} BPM | bin spacing {fs/window_n*60:.2f} BPM",ylabel="Heart rate (BPM)")
    for a in ax:
        a.set_xlabel("Time (s)")
        a.grid(alpha=.25)
        a.legend(loc="upper right")
    fig.suptitle("Subject1: local exploratory comparison | synchronization unverified",fontsize=14)
    fig.savefig(output/"comparison.png",dpi=130)
    plt.close(fig)
    text=(f"# ผลทดลองเปรียบเทียบ subject1 บนเครื่อง\n\n"
        f"- Python: {sys.executable}\n- iPPG run: {ip}\n- Reference run: {rp}\n"
        f"- ช่วงร่วมบนเวลาเดิม: {t[0]:.3f}–{t[-1]:.3f} วินาที\n"
        f"- Correlation เวลาเดิมทั้งช่วง: {full['pearson_r']:.4f}\n"
        f"- ทดลอง offset PPG: {lag:+.4f} วินาที (เทียบ iPPG(t) กับ PPG(t+offset))\n"
        f"- Correlation บนช่วงเดียวกัน ก่อน/หลังเลื่อน: {before['pearson_r']:.4f} / {after['pearson_r']:.4f}\n"
        f"- BPM {len(rows)} ช่วง ช่วงละประมาณ 20 วินาที เลื่อนทีละ 10 วินาที: MAE {np.mean(abs(differences)):.4f} BPM\n"
        f"- ความห่าง bin FFT: {fs/window_n*60:.3f} BPM ไม่ใช่ความแม่นของการวัด\n\n"
        "## ข้อจำกัด\n\nยังไม่ยืนยัน synchronization; offset เลือกจากคลิปเดียวกับที่ให้คะแนนและอาจสับสนรอบการเต้น "
        "ไม่มีการกลับขั้วหรือยืด/หดเวลา คะแนนหลังเลื่อนเป็นผลสำรวจ ไม่ใช่หลักฐานการบันทึกตรงเวลา "
        "ช่วง BPM ซ้อนกันและมาจากคนเดียว ยังสรุปความแม่นยำทั่วไปไม่ได้ "
        "สอง pipeline ใช้ filter ต่าง order กัน จึงยังไม่ใช่การตรวจรูปคลื่นแบบควบคุม filter ให้ตรงกัน\n\n"
        "comparison.png = กราฟ, metrics.json = รายละเอียดวิธี/คะแนน, *.csv = ค่าที่ใช้คำนวณจริง\n")
    (output/"RESULT.md").write_text(text,encoding="utf-8")
    print(json.dumps(result,ensure_ascii=False,indent=2))
    print(f"Saved comparison: {output}")
    return output


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("ippg_run",type=Path)
    parser.add_argument("reference_run",type=Path)
    parser.add_argument("--output-root",type=Path,default=Path(__file__).resolve().parents[1]/"outputs/signal_comparisons")
    args=parser.parse_args()
    compare(args.ippg_run,args.reference_run,args.output_root)
