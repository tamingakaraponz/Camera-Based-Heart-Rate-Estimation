"""Summarize common paired development windows, retaining per-person results."""
import csv
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main(folder):
    manifest=json.loads((folder/"manifest.json").read_text(encoding="utf-8"))
    if manifest["status"]!="complete":
        raise ValueError("Comparison run not complete")
    with (folder/"results.csv").open(encoding="utf-8-sig",newline="") as stream:
        rows=list(csv.DictReader(stream))
    paired={}
    for row in rows:
        key=(row["subject"],row["update_time_s"],row["window_seconds"])
        paired.setdefault(key,{})[row["method"]]=row
    summary=[]
    for subject in manifest["subjects"]:
        for duration in (5,10,15):
            groups=[pair for (s,t,w),pair in paired.items() if s==subject and int(w)==duration]
            common=[pair for pair in groups if set(pair)=={"POS","CHROM"} and all(
                pair[m]["status"]=="estimated" and pair[m]["reference_bpm"] for m in ("POS","CHROM"))]
            for method in ("POS","CHROM"):
                errors=[abs(float(p[method]["bpm"])-float(p[method]["reference_bpm"])) for p in common]
                for p in common:
                    assert p["POS"]["reference_bpm"]==p["CHROM"]["reference_bpm"]
                summary.append(dict(subject=subject,window_seconds=duration,method=method,
                    scheduled=len(groups),estimated=sum(pair.get(method,{}).get("status")=="estimated" for pair in groups),
                    common_paired=len(common),mae_bpm=float(np.mean(errors)) if errors else None,
                    max_error_bpm=max(errors) if errors else None))
    with (folder/"summary.csv").open("w",encoding="utf-8-sig",newline="") as stream:
        w=csv.DictWriter(stream,fieldnames=list(summary[0]));w.writeheader();w.writerows(summary)
    fig,axes=plt.subplots(1,3,figsize=(13,4),sharey=True,layout="constrained")
    subjects=manifest["subjects"]
    for ax,duration in zip(axes,(5,10,15)):
        x=np.arange(len(subjects))
        for offset,method in ((-.18,"POS"),(.18,"CHROM")):
            vals=[next(r["mae_bpm"] for r in summary if r["subject"]==s and r["window_seconds"]==duration and r["method"]==method) for s in subjects]
            ax.bar(x+offset,[np.nan if v is None else v for v in vals],width=.36,label=method)
        ax.set(xticks=x,xticklabels=subjects,title=f"{duration} second windows",xlabel="Development participant")
        ax.grid(axis="y",alpha=.2);ax.legend()
    axes[0].set_ylabel("Mean absolute difference (BPM)")
    fig.suptitle("Development comparison | same paired windows | nominal clocks unverified")
    fig.savefig(folder/"comparison.png",dpi=140);plt.close(fig)
    lines=["# เปรียบเทียบ POS และ CHROM: กลุ่มพัฒนา", "", "ใช้เฉพาะ subject3, subject5, subject8; PPG แถว BPM ที่มากับอุปกรณ์เป็น reference เดียวกันในแต่ละหน้าต่าง", "", "| คน | ช่วง | รอบที่เทียบร่วมกัน | POS ต่างเฉลี่ย | CHROM ต่างเฉลี่ย |", "|---|---:|---:|---:|---:|"]
    for subject in subjects:
        for duration in (5,10,15):
            a=next(r for r in summary if r["subject"]==subject and r["window_seconds"]==duration and r["method"]=="POS")
            b=next(r for r in summary if r["subject"]==subject and r["window_seconds"]==duration and r["method"]=="CHROM")
            def fmt(v): return "ไม่มีคู่" if v is None else f"{v:.2f} BPM"
            lines.append(f"| {subject} | {duration} s | {a['common_paired']} | {fmt(a['mae_bpm'])} | {fmt(b['mae_bpm'])} |")
    lines+=['',"## ข้อจำกัด",'',"เวลาเทียบตามตัวเลขเวลาในไฟล์ ยังไม่ยืนยันเวลาอุปกรณ์ตรงกัน; ค่า BPM อ้างอิงที่อุปกรณ์ให้มาอาจมีการเฉลี่ยภายในที่ไม่ทราบวิธี", "CHROM ใช้กรองภายในและตัดหน้าต่างย่อยสุดท้ายที่ไม่ครบ จึงใช้ช่วงจริงสั้นกว่า POS ได้เล็กน้อย; ทั้งคู่ใช้ DWT และกรองภายนอกเหมือนกัน โดย reference เฉลี่ยตลอดหน้าต่างที่ขอ", "ตัวเลขนี้เป็นผลกลุ่มพัฒนา ไม่ใช่การทดสอบกับคนที่กันไว้ และยังไม่ใช่ผลกล้องสดหรือเว็บ", "รอบที่ reference ไม่ครอบคลุมหรือสัญญาณไม่พร้อมไม่ได้อยู่ใน MAE; จำนวนรอบทั้งหมดและผลวัดได้แยกอยู่ใน summary.csv/results.csv"]
    (folder/"RESULT.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    print("\n".join(lines))


if __name__=="__main__":
    main(Path(sys.argv[1]).resolve(strict=True))
