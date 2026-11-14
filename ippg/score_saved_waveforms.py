"""Exploratory fixed-clock waveform and spectrum metrics on existing plots."""
from pathlib import Path
import csv, json
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
SRC=ROOT/'outputs/waveform_comparison/comparison_20261113_202655_978447'
OUT=SRC/'numeric_comparison.csv'
with (SRC/'results.csv').open(encoding='utf-8-sig',newline='') as fp:
    rows=list(csv.DictReader(fp))
records=[]
for r in rows:
    data=np.load(SRC/f"{r['subject']}_{r['window_seconds']}s.npz")
    t=data['time_s']; fs=1/np.median(np.diff(t)); ref=data['ppg_filtered']
    n=len(t); freqs=np.fft.rfftfreq(n,1/fs); mask=(freqs>=.6)&(freqs<=2.)
    def dist(x):
        p=np.abs(np.fft.rfft(x-np.mean(x)))**2
        p=p[mask]; return p/p.sum()
    rp=dist(ref)
    for method in ('pos','chrom'):
        x=data[method+'_filtered']; xp=dist(x)
        corr=float(np.corrcoef(ref,x)[0,1])
        overlap=float(np.minimum(rp,xp).sum())
        diff=abs(float(r[method+'_fft_bpm'])-float(r['ppg_fft_bpm']))
        records.append(dict(subject=r['subject'],window_seconds=r['window_seconds'],method=method.upper(),correlation=corr,spectral_overlap=overlap,fft_bpm_difference=diff))
with OUT.open('w',encoding='utf-8-sig',newline='') as fp:
    w=csv.DictWriter(fp,fieldnames=records[0]);w.writeheader();w.writerows(records)
print('Saved',OUT)
for r in records:
    if r['subject']=='subject3' and r['window_seconds']=='10': print(json.dumps(r))
