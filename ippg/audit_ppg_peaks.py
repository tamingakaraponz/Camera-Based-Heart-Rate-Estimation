"""Audit PPG peaks already used in the six-subject band comparison."""
import csv,json
from collections import Counter
from datetime import datetime
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

ROOT=Path(__file__).resolve().parents[1]; SRC=ROOT/'outputs/band_ab_20261116_150137_572560'; DATA=ROOT.parent/'dataset for ipph'

def main():
 out=ROOT/'outputs'/('ppg_peak_audit_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'));out.mkdir(parents=True)
 with (SRC/'windows.csv').open(encoding='utf-8-sig',newline='') as f: rows=list(csv.DictReader(f))
 audits=[];peaks_rows=[];failures=json.loads((SRC/'manifest.json').read_text(encoding='utf-8'))['failures']
 for num in (3,5,8,9,10,11):
  subject=f'subject{num}';matches=list(DATA.glob(f'{subject}-*/{subject}/ground_truth.txt'))
  if len(matches)!=1:raise ValueError(f'{subject}: expected one ground_truth, got {len(matches)}')
  original=np.loadtxt(matches[0],ndmin=2);native=original[2]-original[2,0]
  fig,axes=plt.subplots(4,3,figsize=(16,11),layout='constrained',sharey=False)
  for i,end in enumerate((20,30,40,50)):
   for j,seconds in enumerate((5,10,15)):
    ax=axes[i,j];items=[r for r in rows if r['subject']==subject and float(r['end_s'])==end and int(r['window_s'])==seconds]
    if not items:ax.text(.5,.5,'No paired result',ha='center',va='center',transform=ax.transAxes);ax.set_title(f'End {end}s / {seconds}s: unavailable');continue
    row=items[0];d=np.load(SRC/f'{subject}_{end}_{seconds}.npz');t=d['time'];x=d['ppg_filtered'];fps=1/np.median(np.diff(t));idx,_=find_peaks(x,distance=int(np.ceil(fps*.3)),prominence=.35*np.std(x)); intervals=np.diff(t[idx]);bpm=60/intervals.mean() if len(intervals) else float('nan')
    if not np.isclose(bpm,float(row['ppg_bpm']),atol=1e-7):raise AssertionError(f'Peak BPM differs: {subject} {end} {seconds}')
    fft=float(row['ppg_raw_fft']);diff=abs(bpm-fft); jumps=np.abs(np.diff(intervals))/intervals[:-1] if len(intervals)>1 else np.array([]);maxjump=float(jumps.max()) if len(jumps) else None
    used=native[(native>=t[0])&(native<=t[-1])];gaps=np.diff(used);maxgap=float(gaps.max()) if len(gaps) else None
    flags=[]
    if diff>10:flags.append('peak_vs_fft_gt_10_bpm')
    if maxjump is not None and maxjump>.3:flags.append('interval_change_gt_30pct')
    if len(idx)<4:flags.append('fewer_than_4_peaks')
    record=dict(subject=subject,end_s=end,window_s=seconds,start_s=float(t[0]),last_s=float(t[-1]),samples=len(t),num_peaks=len(idx),peak_bpm=float(bpm),raw_fft_bpm=fft,abs_bpm_difference=diff,interval_min_s=float(intervals.min()),interval_max_s=float(intervals.max()),adjacent_interval_change_max_pct=100*maxjump if maxjump is not None else None,native_max_gap_s=maxgap,flags=';'.join(flags))
    audits.append(record)
    for k,peak in enumerate(idx):peaks_rows.append(dict(subject=subject,end_s=end,window_s=seconds,peak_index=k,time_s=float(t[peak]),interval_before_s=float(t[peak]-t[idx[k-1]]) if k else None))
    y=(x-x.mean())/x.std();ax.plot(t,y,color='#496d91',lw=.9);ax.scatter(t[idx],y[idx],s=14,color='#d34335',zorder=3);ax.grid(alpha=.2)
    ax.set_title(f'{end-seconds}–{end}s | peak {bpm:.1f}, FFT {fft:.1f} | '+('REVIEW' if flags else 'no flag'),fontsize=9,color='#b13d2d' if flags else '#222222')
    if i==3:ax.set_xlabel('Time (s)')
    if j==0:ax.set_ylabel('PPG z-score')
  fig.suptitle(f'{subject}: red dots = detected PPG peaks | thresholds flag review, not errors',fontsize=13);fig.savefig(out/f'{subject}_peaks.png',dpi=130);plt.close(fig);print('Done',subject,flush=True)
 with (out/'windows_audit.csv').open('w',encoding='utf-8-sig',newline='') as f:
  w=csv.DictWriter(f,fieldnames=audits[0]);w.writeheader();w.writerows(audits)
 with (out/'detected_peaks.csv').open('w',encoding='utf-8-sig',newline='') as f:
  w=csv.DictWriter(f,fieldnames=peaks_rows[0]);w.writeheader();w.writerows(peaks_rows)
 summary={}
 for num in (3,5,8,9,10,11):
  subject=f'subject{num}';a=[r for r in audits if r['subject']==subject];summary[subject]=dict(valid=len(a),attempted=12,flagged=sum(bool(r['flags']) for r in a),peak_fft_gt_10=sum('peak_vs_fft_gt_10_bpm' in r['flags'] for r in a),interval_change_gt_30pct=sum('interval_change_gt_30pct' in r['flags'] for r in a),mean_difference_bpm=float(np.mean([r['abs_bpm_difference'] for r in a])),max_difference_bpm=max(r['abs_bpm_difference'] for r in a))
 report=dict(source=str(SRC),summary=summary,missing_windows=failures,rules={'peak_detection':'filtered PPG, min distance ceil(0.3*video FPS) samples, prominence 0.35 standard deviations; peaks >=3 to estimate','fft':'32x zero-padded FFT on interpolated PPG before filtering, 0.6–3 Hz','flags':'exploratory review markers: |peak minus FFT| >10 BPM OR consecutive intervals change >30% OR <4 detected peaks; not verified clinical thresholds','limitations':'70 previously successful paired windows only; both estimates use interpolated shared grid; overlapping windows; missing windows inherited; no automatic ground-truth verdict; native PPG max gap computed separately'},flag_counts=dict(Counter(x for r in audits for x in r['flags'].split(';') if x)))
 (out/'manifest.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print('OUTPUT',out);print(json.dumps(report['summary'],indent=2))
if __name__=='__main__':main()
