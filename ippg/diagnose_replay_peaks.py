"""Inspect saved replay FFT choices; exploratory controls, no core changes."""
import argparse
import csv
from dataclasses import replace
from datetime import datetime
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import butter, sosfiltfilt

import ippg_v2_refactored as core

ROOT=Path(__file__).resolve().parents[1]
DEFAULT=ROOT/"outputs/realtime_replay/replay_20261111_183034_887089"


def inventory():
    videos={}
    for p in (ROOT/"DATA").rglob("*"):
        if p.is_file() and p.suffix.lower() in {".avi",".mp4",".mov",".mkv"} and p.stem.isdigit():
            code=int(p.stem)
            if 1<=code//10<=20 and code%10 in (1,2,3):
                videos.setdefault(code,[]).append(str(p))
    refs={int(p.stem):str(p) for p in (ROOT/"DATAcppg").glob("*.xlsx") if p.stem.isdigit()}
    codes=sorted(set(videos)&set(refs))
    return dict(old_subjects=sorted({c//10 for c in codes}),old_paired_trials=len(codes),
        trials=[dict(code=c,subject=c//10,trial=c%10,videos=videos[c],reference=refs[c]) for c in codes],
        counting_rule="Unique subject/trial codes, not video copies; other test files have unknown identities")


def fft_values(t,y,fs):
    gt=np.arange(t[0],t[-1],1/fs)
    values=np.interp(gt,t,y)
    return values-values.mean()


def spectrum(t,y,fs,hann=False,dense=False):
    x=fft_values(t,y,fs)
    if hann:
        x=x*np.hanning(len(x))
    n=len(x)*32 if dense else len(x)
    f=np.fft.rfftfreq(n,1/fs)
    power=abs(np.fft.rfft(x,n=n))**2
    ids=np.flatnonzero((f>=.6)&(f<=2))
    peak=int(ids[np.argmax(power[ids])])
    top=sorted(ids,key=lambda i:power[i],reverse=True)[:3]
    return dict(bpm=float(f[peak]*60),bin_index=peak,spacing_bpm=float((f[1]-f[0])*60),
        top_bins=[dict(bpm=float(f[i]*60),relative_power=float(power[i]/power[peak])) for i in top]),f,power


def zscore(x):
    return (x-x.mean())/max(float(x.std()),1e-12)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run",nargs="?",type=Path,default=DEFAULT)
    args=parser.parse_args()
    source=args.run.resolve()
    manifest=json.loads((source/"manifest.json").read_text(encoding="utf-8"))
    if manifest["status"]!="complete":
        raise ValueError("Replay incomplete")
    for name,digest in manifest["files"].items():
        assert hashlib.sha256((source/name).read_bytes()).hexdigest()==digest, name
    rows=list(csv.DictReader((source/"updates.csv").open(encoding="utf-8-sig")))
    fs=manifest["fps"]
    cfg=core.IPPGConfig(max_duration_seconds=None,show_preview=False,pulse_method="pos",roi_method="legacy",denoise="dwt",background_method="none")
    raw_reference=np.loadtxt(next(p for p in manifest["source"] if p.endswith("ground_truth.txt")))
    ref_t=raw_reference[2]-raw_reference[2,0]
    controls=[]
    bad=[]
    all_diagnostics=[]
    with np.load(source/"window_stages.npz") as stages:
        for i,row in enumerate(rows):
            if float(row["window_seconds"])!=5 or not row["difference_bpm"]:
                continue
            def a(name):
                return stages[f"row{i:04d}_{name}"].copy()
            t, pulse, filtered, refraw, reffiltered=a("time_seconds"),a("raw_pos"),a("filtered"),a("reference_raw"),a("reference_filtered")
            ip,f,ipower=spectrum(t,filtered,fs)
            ref,_,rpower=spectrum(t,reffiltered,fs)
            assert abs(ip["bpm"]-float(row["ippg_bpm"]))<1e-10
            assert abs(ref["bpm"]-float(row["reference_bpm"]))<1e-10
            # Change exactly the named stage; do not use controls as a new pipeline.
            nodwt=core.process_pulse(pulse,fs,"pos",replace(cfg,denoise="none")).filtered
            iporder3=core.process_pulse(pulse,fs,"pos",replace(cfg,forehead_filter_order=3)).filtered
            sos=butter(6,[.6,2],btype="bandpass",fs=fs,output="sos")
            reforder6=sosfiltfilt(sos,refraw-refraw.mean(),padlen=3*(2*len(sos)+1))
            methods={
                "baseline":(ip["bpm"],ref["bpm"]),
                "ippg_without_dwt":(spectrum(t,nodwt,fs)[0]["bpm"],ref["bpm"]),
                "ippg_order3":(spectrum(t,iporder3,fs)[0]["bpm"],ref["bpm"]),
                "reference_order6":(ip["bpm"],spectrum(t,reforder6,fs)[0]["bpm"]),
                "hann_both":(spectrum(t,filtered,fs,hann=True)[0]["bpm"],spectrum(t,reffiltered,fs,hann=True)[0]["bpm"]),
                "dense_fft_both":(spectrum(t,filtered,fs,dense=True)[0]["bpm"],spectrum(t,reffiltered,fs,dense=True)[0]["bpm"]),
            }
            for method,(ib,rb) in methods.items():
                controls.append(dict(row_index=i,update_time_s=float(row["update_time_s"]),method=method,ippg_bpm=ib,reference_bpm=rb,difference_bpm=ib-rb))
            rawtimes=ref_t[(ref_t>=t[0])&(ref_t<=float(row["update_time_s"]))]
            details=dict(row_index=i,start_s=float(t[0]),end_s=float(t[-1]),update_time_s=float(row["update_time_s"]),
                ippg=ip,reference=ref,difference_bpm=ip["bpm"]-ref["bpm"],
                selected_bin_difference=ip["bin_index"]-ref["bin_index"],
                input_max_gap_s=float(row["max_gap_s"]),reference_max_gap_s=float(np.max(np.diff(rawtimes))),
                supplied_hr_mean_bpm=float(raw_reference[1,(ref_t>=t[0])&(ref_t<=t[-1])].mean()),
                methods={k:dict(ippg_bpm=v[0],reference_bpm=v[1],difference_bpm=v[0]-v[1]) for k,v in methods.items()})
            all_diagnostics.append(details)
            if abs(details["difference_bpm"])>5:
                bad.append(details)
        worst=max(bad,key=lambda x:abs(x["difference_bpm"]))
        i=worst["row_index"]
        def worst_a(name):
            return stages[f"row{i:04d}_{name}"].copy()
        t=worst_a("time_seconds")
        ix,rx=worst_a("filtered"),worst_a("reference_filtered")
        _,f,ipower=spectrum(t,ix,fs)
        _,_,rpower=spectrum(t,rx,fs)
        _,fd,ipdense=spectrum(t,ix,fs,dense=True)
        _,_,rdense=spectrum(t,rx,fs,dense=True)

    output=ROOT/"outputs/replay_peak_diagnostic"/datetime.now().strftime("diagnostic_%Y%m%d_%H%M%S_%f")
    output.mkdir(parents=True,exist_ok=False)
    summary={}
    for method in sorted({r["method"] for r in controls}):
        errors=[abs(r["difference_bpm"]) for r in controls if r["method"]==method]
        summary[method]=dict(count=len(errors),MAE_bpm=float(np.mean(errors)),max_error_bpm=float(max(errors)),over_5_bpm=sum(e>5 for e in errors))
    with (output/"controls.csv").open("w",encoding="utf-8-sig",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=list(controls[0]))
        writer.writeheader(); writer.writerows(controls)
    result=dict(source_run=str(source),source_manifest_sha256=hashlib.sha256((source/"manifest.json").read_bytes()).hexdigest(),
        diagnostic_code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        selection_rule="All paired 5-second windows; flag |baseline difference|>5 BPM; illustrative worst absolute error",
        paired_5s_count=len(all_diagnostics),flagged_count=len(bad),flagged=bad,all_5s_windows=all_diagnostics,
        control_summary=summary,inventory=inventory(),
        new_subject1_reference=str(next(p for p in manifest["source"] if p.endswith("ground_truth.txt"))),
        caveats=["No core or replay algorithm changed", "Controls explored on this already-seen recording, not held-out validation",
            "Dense FFT interpolates spectrum, does not add observation time or prove accuracy", "Reference clock unverified; supplied HR is an extra signal, not ground truth for exact instant",
            "Equal nominal sample counts do not verify frame-to-sensor synchronization"])
    (output/"diagnostic.json").write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False),encoding="utf-8")
    fig,ax=plt.subplots(3,1,figsize=(12,11),layout="constrained")
    ax[0].plot(t,zscore(ix),label="iPPG filtered")
    ax[0].plot(t,zscore(rx),label="Reference filtered",alpha=.8)
    ax[0].set(title=f"Worst paired 5s window | {t[0]:.2f}-{t[-1]:.2f}s | original nominal clocks",ylabel="Z-score (a.u.)",xlabel="Time (s)")
    keep=(f>=.6)&(f<=2)
    ax[1].plot(f[keep]*60,ipower[keep]/max(ipower[keep]),"o-",label="iPPG FFT bins")
    ax[1].plot(f[keep]*60,rpower[keep]/max(rpower[keep]),"s--",label="Reference FFT bins")
    kd=(fd>=.6)&(fd<=2)
    ax[1].set(title="Different winning bins; normalize each spectrum separately",xlabel="Frequency as BPM",ylabel="Relative power")
    ax[2].plot(fd[kd]*60,ipdense[kd]/max(ipdense[kd]),label="iPPG dense FFT")
    ax[2].plot(fd[kd]*60,rdense[kd]/max(rdense[kd]),label="Reference dense FFT")
    ax[2].set(title="32x zero-padding: spectrum interpolation only, not validated accuracy",xlabel="Frequency as BPM",ylabel="Relative power")
    for a in ax:
        a.grid(alpha=.25); a.legend()
    fig.suptitle("Exploratory peak diagnostic | one subject | clock synchronization unverified")
    fig.savefig(output/"worst_window.png",dpi=130)
    plt.close(fig)
    text="# ตรวจช่วง 5 วินาทีที่ต่างจาก PPG มาก\n\n"
    text+=f"ตรวจ {len(all_diagnostics)} คู่ที่มี reference โดยไม่เปลี่ยน pipeline เดิม พบส่วนต่างมากกว่า 5 BPM จำนวน {len(bad)} รอบ\n\n"
    text+="| เวลาอัปเดต | iPPG | PPG | ส่วนต่าง | ห่างกี่ช่อง FFT |\n|---|---:|---:|---:|---:|\n"
    for b in bad:
        text+=f"| {b['update_time_s']:.3f} s | {b['ippg']['bpm']:.3f} | {b['reference']['bpm']:.3f} | {b['difference_bpm']:.3f} | {b['selected_bin_difference']} |\n"
    text+="\nทุกกรณีนี้เลือกช่องข้างกัน ระยะห่างช่องประมาณ 12 BPM เป็นกลไกที่ทำให้ตัวเลขต่างเป็นขั้น ไม่ใช่หลักฐานว่าชีพจรจริงต่างกัน 12 BPM "
    text+="ยังต้องดูว่าเหตุใดสเปกตรัมต่างกัน ซึ่งเกี่ยวข้องได้กับสัญญาณ วิธีกรอง ขอบหน้าต่าง และการจับคู่เวลา\n\n"
    text+="## ทดลองเปลี่ยนทีละส่วนเพื่อหาสาเหตุ (ยังไม่ใช้แทนวิธีหลัก)\n\n| วิธี | คู่ที่เทียบ | MAE | สูงสุด | ต่างเกิน 5 BPM |\n|---|---:|---:|---:|---:|\n"
    for method,s in summary.items():
        text+=f"| {method} | {s['count']} | {s['MAE_bpm']:.3f} | {s['max_error_bpm']:.3f} | {s['over_5_bpm']} |\n"
    text+="\nการเปลี่ยนหน้าต่าง FFT เป็น Hann เปลี่ยนน้ำหนักบริเวณขอบ; dense FFT เติมศูนย์ 32 เท่าเพื่อดูสเปกตรัมละเอียดขึ้น ไม่ได้เพิ่มข้อมูลจริง "
    text+="ห้ามเลือกวิธีจาก MAE คลิปเดียวโดยไม่ทดสอบแยก การทดลองทั้งหมดใช้ reference/time mapping เดิมที่ยังไม่ยืนยัน\n\n"
    text+="## จำนวนข้อมูลที่นับได้\n\nชุดเดิม 20 รหัสบุคคล × 3 รอบ = 60 คู่ มีวิดีโอบางรหัสซ้ำหลายไฟล์จึงไม่นับเป็นคนเพิ่ม "
    text+="ชุดใหม่ subject1 มี 1 คู่และเป็นคู่เดียวที่ทำ replay 5/10/15 แล้ว ไม่รวมไฟล์ชื่อ test/ชื่ออื่นที่ยังยืนยันตัวบุคคลไม่ได้ "
    text+="จำนวนนี้นับตามรหัสในแต่ละชุด ไม่ได้ตรวจเอกลักษณ์บุคคลข้ามชุด\n"
    (output/"RESULT.md").write_text(text,encoding="utf-8")
    print(json.dumps(dict(flagged=bad,summary=summary),ensure_ascii=False,indent=2))
    print(f"Saved diagnostic: {output}")


if __name__=="__main__":
    main()
