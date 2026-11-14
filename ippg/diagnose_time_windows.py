"""Predeclared temporal/cropping sensitivity check; no production changes."""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import statistics

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks

import ippg_v2_refactored as core
from benchmark_background_methods import analyze_contact_ppg

ROOT = Path(__file__).resolve().parents[1]
FACTOR = 34 / 32
WINDOWS = [("common32", 2.1, 34.1)] + [
    (f"short{i+1}", a, a+10) for i,a in enumerate((2.1,12.1,22.1))
] + [(f"long{i+1}",a,a+16) for i,a in enumerate((2.1,10.1,18.1))]


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def write_json(path,obj):
    path.write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding="utf-8")


def describe_peak(frequency,power):
    good=np.flatnonzero((frequency>=0.6)&(frequency<=2))
    if not len(good) or np.max(power[good])<=1e-12:
        raise ValueError("No usable in-band peak")
    index=int(good[np.argmax(power[good])])
    spacing=float(frequency[1]-frequency[0])
    delta=0.0
    interpolation_status="not_available"
    if 0<index<len(power)-1 and index not in (good[0],good[-1]):
        left,middle,right=np.log(np.maximum(power[index-1:index+2],np.finfo(float).tiny))
        curvature=left-2*middle+right
        if curvature<0:
            delta=float(np.clip(0.5*(left-right)/curvature,-0.5,0.5))
            interpolation_status="quadratic_log_power"
    local=find_peaks(power)[0]
    candidates=[index]+sorted((int(i) for i in local if i in good and abs(frequency[i]-frequency[index])>=0.1),key=lambda i:power[i],reverse=True)
    top=[]
    for i in candidates:
        if all(abs(frequency[i]-frequency[k])>=0.1 for k in top):
            top.append(i)
        if len(top)==3:
            break
    return {"bin_bpm":float(frequency[index]*60),"interpolated_bpm":float((frequency[index]+delta*spacing)*60),
            "interpolation_status":interpolation_status,"bin_spacing_bpm":spacing*60,
            "peak_at_search_boundary":index in (good[0],good[-1]),
            "runner_up_to_main_power":float(power[top[1]]/power[index]) if len(top)>1 else None,
            "separated_peaks":[{"bpm":float(frequency[i]*60),"relative_power":float(power[i]/power[index])} for i in top]}


def analyze_window(t,y,start,end):
    grid=start+np.arange(round((end-start)*30))/30
    if t[0]>grid[0]+1e-8 or t[-1]<grid[-1]-1e-8:
        raise ValueError(f"Incomplete [{start},{end}) window: data {t[0]}–{t[-1]}")
    values=np.interp(grid,t,y)
    values=values-np.mean(values)
    frequency=np.fft.rfftfreq(len(values),1/30)
    power=np.abs(np.fft.rfft(values))**2
    return describe_peak(frequency,power),frequency,power


def differences(i,c):
    raw=abs(i-c)
    scaled=abs(i-FACTOR*c)
    return {"before_bpm":float(raw),"after_bpm":float(scaled),"outcome":"closer" if raw-scaled>1e-9 else "farther" if raw-scaled < -1e-9 else "tie"}


def standardize(y):
    return (y-np.mean(y))/max(float(np.std(y)),1e-12)


def plot_clip(code,it,iy,ct,cy,spectra,path):
    fig,axes=plt.subplots(4,2,figsize=(15,13),layout="constrained")
    fig.suptitle(f"DATA {code} | Fixed time-window diagnostic | Relative clocks NOT synchronized",fontsize=15)
    axes[0,0].plot(it,standardize(iy),color="#2878b5",lw=0.9)
    axes[0,0].set_title("Video pulse (same saved POS + band-pass)")
    axes[0,1].plot(ct,standardize(cy),color="#c77822",lw=0.9)
    axes[0,1].set_title("Sensor pulse (full-workbook band-pass)")
    for a in axes[0]:
        a.set_xlabel("Own nominal relative time (s)")
        a.set_ylabel("Standardized amplitude")
        a.set_xlim(0,36)
        for _,s,e in WINDOWS[1:4]:
            a.axvline(s,color="gray",lw=0.6,alpha=0.5)
    for row in range(1,4):
        for col,name in enumerate((f"short{row}",f"long{row}")):
            f,ip,cp,start,end=spectra[name]
            keep=(f>=0.5)&(f<=2.2)
            a=axes[row,col]
            a.plot(f[keep]*60,ip[keep]/max(ip[keep]),label="iPPG",color="#2878b5")
            a.plot(f[keep]*60,cp[keep]/max(cp[keep]),label="cPPG original",color="#c77822")
            a.plot(f[keep]*60*FACTOR,cp[keep]/max(cp[keep]),label="cPPG x 1.0625 (what-if)",color="#8a499e",ls="--")
            a.set_title(f"{start:g}–{end:g} s on own axes | {end-start:g} s window")
            a.set_xlabel("Frequency expressed as BPM")
            a.set_ylabel("Normalized FFT power")
            a.set_xlim(36,120)
            a.legend(fontsize=8)
    for a in axes.flat:
        a.grid(alpha=0.2)
    fig.savefig(path,dpi=115)
    plt.close(fig)


def mean(values):
    return float(statistics.mean(values)) if values else None


def summarize(rows):
    good=[r for r in rows if r["status"]=="success"]
    output={"attempted":len(rows),"success":len(good),"failed":len(rows)-len(good)}
    for method in ("bin","interpolated"):
        before=[r["windows"]["common32"][method+"_difference"]["before_bpm"] for r in good]
        after=[r["windows"]["common32"][method+"_difference"]["after_bpm"] for r in good]
        short=[r["windows"][f"short{i}"][method+"_difference"] for r in good for i in (1,2,3)]
        long=[r["windows"][f"long{i}"][method+"_difference"] for r in good for i in (1,2,3)]
        output[method]={"common32_mean_before_bpm":mean(before),"common32_mean_after_bpm":mean(after),
                        "short10_mean_before_bpm":mean([v["before_bpm"] for v in short]),"short10_mean_after_bpm":mean([v["after_bpm"] for v in short]),
                        "long16_mean_before_bpm":mean([v["before_bpm"] for v in long]),"long16_mean_after_bpm":mean([v["after_bpm"] for v in long]),
                        "short10_outcomes":{k:sum(x["outcome"]==k for x in short) for k in ("closer","farther","tie")},
                        "long16_outcomes":{k:sum(x["outcome"]==k for x in long) for k in ("closer","farther","tie")},
                        "iPPG_short_range_mean_bpm":mean([r["variability"][method]["ippg_short_range_bpm"] for r in good]),
                        "cPPG_short_range_mean_bpm":mean([r["variability"][method]["cppg_short_range_bpm"] for r in good]),
                        "iPPG_long_range_mean_bpm":mean([r["variability"][method]["ippg_long_range_bpm"] for r in good]),
                        "cPPG_long_range_mean_bpm":mean([r["variability"][method]["cppg_long_range_bpm"] for r in good])}
    return output


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    folder=args.output.resolve()
    folder.mkdir(parents=True,exist_ok=False)
    source=ROOT/"outputs/five-new-people-factor-20261102"
    old=json.loads((source/"summary.json").read_text(encoding="utf-8"))
    if digest(Path(core.__file__))!=old["core_sha256"]:
        raise RuntimeError("Production core has changed since baseline")
    protocol=ROOT/"outputs/_inspection/temporal_diagnostic_protocol.md"
    report={"protocol":str(protocol),"protocol_sha256":digest(protocol),"source_summary_sha256":digest(source/"summary.json"),
            "core_sha256":old["core_sha256"],"factor":FACTOR,"synchronization":"unverified; nominal axes only", "windows":WINDOWS,
            "interpolation":"secondary spectral diagnostic only; not production or ground truth","records":[]}
    for oldrow in old["records"]:
        code=oldrow["data_id"]
        record={"data_id":code,"subject":oldrow["subject"],"trial":oldrow["trial"],"status":"failed"}
        p=source/(code+"_signals.npz")
        contact=ROOT/"DATAcppg"/(code+".xlsx")
        before={"npz":digest(p),"cppg":digest(contact)}
        try:
            with np.load(p) as z:
                it=z["time_seconds"].copy();iy=z["filtered_pos"].copy()
                raw=z["raw_pos"].copy()
            f,s=core.calculate_spectrum(it,iy,oldrow["fps"])
            reproduce=core.estimate_heart_rate(f,s,0.6,2.0)
            if abs(reproduce-oldrow["ippg_bpm"])>1e-9:
                raise RuntimeError("Saved signal does not reproduce prior BPM")
            c=analyze_contact_ppg(contact,0.6,2.0,filter_order=3)
            if abs(c.heart_rate_bpm-oldrow["cppg_original_bpm"])>1e-9:
                raise RuntimeError("Reference no longer reproduces prior BPM")
            ct=c.time_seconds;cy=c.filtered_amplitude
            windows={};spectra={}
            for name,start,end in WINDOWS:
                ip,f,ipow=analyze_window(it,iy,start,end)
                cp,f2,cpow=analyze_window(ct,cy,start,end)
                assert np.array_equal(f,f2)
                windows[name]={"start_s":start,"end_s":end,"ippg":ip,"cppg":cp,
                               "bin_difference":differences(ip["bin_bpm"],cp["bin_bpm"]),
                               "interpolated_difference":differences(ip["interpolated_bpm"],cp["interpolated_bpm"])}
                spectra[name]=(f,ipow,cpow,start,end)
            variability={}
            for method in ("bin","interpolated"):
                def spread(prefix,signal):
                    vals=[windows[f"{prefix}{i}"][signal][method+"_bpm"] for i in (1,2,3)]
                    return float(max(vals)-min(vals))
                variability[method]={f"{signal}_{prefix}_range_bpm":spread(prefix,signal) for signal in ("ippg","cppg") for prefix in ("short","long")}
            record.update(status="success",windows=windows,variability=variability,
                          old_whole_interval={k:oldrow[k] for k in ("ippg_bpm","cppg_original_bpm","absolute_difference_before_bpm","absolute_difference_after_bpm")},
                          original_bpm_reproduced=True)
            plot_clip(code,it,iy,ct,cy,spectra,folder/(code+"_windows.png"))
        except Exception as error:
            record.update(error_type=type(error).__name__,error=str(error))
        record["input_hashes"]=before
        record["inputs_unchanged"]={"npz":digest(p)==before["npz"],"cppg":digest(contact)==before["cppg"]}
        report["records"].append(record)
        report["overall"]=summarize(report["records"])
        report["per_subject"]={str(s):summarize([r for r in report["records"] if r["subject"]==s]) for s in (2,3,4,5,6)}
        write_json(folder/(code+".json"),record)
        write_json(folder/"summary.json",report)
        print(code,record["status"],record.get("error",""),flush=True)
    print(json.dumps(report["overall"],indent=2),flush=True)


if __name__=="__main__":
    main()
