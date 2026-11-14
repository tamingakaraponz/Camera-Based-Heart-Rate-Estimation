"""Explain disagreement between two references without choosing a winner."""
import csv
from datetime import datetime
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import periodogram

ROOT=Path(__file__).resolve().parents[1]
COMPARE=ROOT/"outputs/dev_method_comparison/comparison_20261113_194322_362905"
REPLAY=ROOT/"outputs/subject3_investigation/replay_20261113_195450_719391"


def main():
    with (COMPARE/"results.csv").open(encoding="utf-8-sig",newline="") as f:
        primary=list(csv.DictReader(f))
    with (REPLAY/"updates.csv").open(encoding="utf-8-sig",newline="") as f:
        replay=list(csv.DictReader(f))
    manifest=json.loads((REPLAY/"manifest.json").read_text(encoding="utf-8"))
    if manifest["status"]!="complete":
        raise ValueError("Replay incomplete")
    a={(round(float(r["update_time_s"]),6),int(r["window_seconds"])):r for r in primary
       if r["subject"]=="subject3" and r["method"]=="POS" and r["bpm"] and r["reference_bpm"]}
    records=[]
    for index,r in enumerate(replay):
        key=(round(float(r["update_time_s"]),6),int(r["window_seconds"]))
        if key not in a or not r["reference_bpm"] or not r["ippg_bpm"]:
            continue
        p=a[key]
        if abs(float(p["bpm"])-float(r["ippg_bpm"]))>1e-8:
            raise ValueError("POS results differ across runs")
        pos,device,wave=map(float,(r["ippg_bpm"],p["reference_bpm"],r["reference_bpm"]))
        records.append(dict(row_index=index,update_s=key[0],window_s=key[1],pos_bpm=pos,
            device_hr_bpm=device,ppg_fft_bpm=wave,device_vs_wave_bpm=device-wave,
            pos_vs_device_bpm=pos-device,pos_vs_wave_bpm=pos-wave))
    if not records:
        raise ValueError("No shared windows")
    groups=[]
    for duration in (5,10,15):
        r=[x for x in records if x["window_s"]==duration]
        groups.append(dict(window_s=duration,count=len(r),
            references_mae=float(np.mean([abs(x["device_vs_wave_bpm"]) for x in r])),
            pos_device_mae=float(np.mean([abs(x["pos_vs_device_bpm"]) for x in r])),
            pos_wave_mae=float(np.mean([abs(x["pos_vs_wave_bpm"]) for x in r]))))
    worst=max((r for r in records if r["window_s"]==15),key=lambda r:abs(r["device_vs_wave_bpm"]))
    out=ROOT/"outputs/subject3_reference_diagnostic"/datetime.now().strftime("diagnostic_%Y%m%d_%H%M%S_%f")
    out.mkdir(parents=True,exist_ok=False)
    with (out/"paired_windows.csv").open("w",encoding="utf-8-sig",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
    with np.load(REPLAY/"window_stages.npz") as z:
        prefix=f"row{worst['row_index']:04d}_"
        t=z[prefix+"time_seconds"]
        values={key:z[prefix+key] for key in ("filtered","reference_raw","reference_filtered")}
    fs=manifest["fps"]
    fig,axes=plt.subplots(2,1,figsize=(12,8),layout="constrained")
    for key,label in (("filtered","iPPG POS"),("reference_raw","PPG interpolated"),("reference_filtered","PPG filtered")):
        x=values[key]; axes[0].plot(t,(x-x.mean())/max(x.std(),1e-12),label=label,lw=.9)
        f,power=periodogram(x,fs=fs,detrend="constant",window="boxcar")
        band=(f>=.6)&(f<=3)
        axes[1].plot(f[band]*60,power[band]/max(power[band]),label=label)
    axes[0].set(title=f"15-second window ending {worst['update_s']:.2f}s | Original nominal clocks",xlabel="Time (s)",ylabel="Standardized amplitude")
    axes[1].axvline(worst["device_hr_bpm"],color="black",ls="--",label="Supplied HR mean")
    axes[1].set(title="Spectral comparison | each curve scaled separately",xlabel="Frequency (BPM)",ylabel="Relative power")
    for ax in axes:
        ax.grid(alpha=.25);ax.legend()
    fig.savefig(out/"reference_comparison.png",dpi=140);plt.close(fig)
    result=dict(subject="subject3",matched_windows=groups,example=worst,
        reference_1="Mean of device-provided HR row over requested window; internal device averaging unknown",
        reference_2="PPG waveform filtered and FFT-estimated 0.6-2Hz on iPPG grid; order3; shared FFT range may bias agreement",
        clocks="Nominal stored timestamps; physical synchronization not verified",
        caveat="The references are correlated and use different processing; neither is established as the true HR. Observational, one subject.",
        held_out_opened=False)
    (out/"diagnostic.json").write_text(json.dumps(result,indent=2,ensure_ascii=False,allow_nan=False),encoding="utf-8")
    lines=["# ตรวจค่าอ้างอิง subject3", "", "ใช้เฉพาะรอบที่มีทั้งค่า BPM จากไฟล์และค่า FFT ของคลื่น PPG พร้อมกัน; POS ได้ค่าเดียวกันในสองรอบรัน", "", "| ช่วง | จำนวนรอบร่วม | ค่าอ้างอิงสองแบบต่างกันเฉลี่ย | POS ต่างจาก BPM ที่ไฟล์ให้ | POS ต่างจาก FFT คลื่น PPG |", "|---|---:|---:|---:|---:|"]
    for g in groups:
        lines.append(f"| {g['window_s']} s | {g['count']} | {g['references_mae']:.2f} BPM | {g['pos_device_mae']:.2f} BPM | {g['pos_wave_mae']:.2f} BPM |")
    lines += ['',f"ตัวอย่างช่วง 15 วินาทีที่อัปเดต {worst['update_s']:.2f} s: POS {worst['pos_bpm']:.2f}, BPM จากไฟล์ {worst['device_hr_bpm']:.2f}, FFT จากคลื่น PPG {worst['ppg_fft_bpm']:.2f} BPM",'',"**ข้อจำกัด:** ค่าแถว BPM อาจผ่านการเฉลี่ยภายในอุปกรณ์ที่ไม่ทราบ; PPG FFT ใช้ช่วงค้นหาเดียวกับ POS และกรองต่างกัน จึงอาจดูใกล้กันโดยวิธีคำนวณ ไม่รู้ว่าวิธีอ้างอิงใดถูกต้องจริง และยังไม่ยืนยันเวลาบันทึกตรงกัน", "การตรวจนี้เป็นการสำรวจใน subject3 หนึ่งคน ไม่ใช้ปรับวิธีหรือเลือกผลอ้างอิงให้คะแนนดีที่สุด และยังไม่เปิดกลุ่มที่กันไว้"]
    (out/"RESULT.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    print(json.dumps(result,indent=2,ensure_ascii=False))
    print(f"Saved diagnostic: {out}")


if __name__=="__main__":
    main()
