"""Exploratory template bank, leave-one-person-out on three saved windows."""
from pathlib import Path
import csv,json
import numpy as np
from scipy.signal import find_peaks,butter,sosfiltfilt
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]
SRC=ROOT/'outputs/waveform_metrics/dev_20261113_213340_139393'
OUT=ROOT/'outputs/template_candidates';OUT.mkdir(parents=True,exist_ok=True)
subjects=('subject3','subject5','subject8');phase=np.arange(200)/200
waves={}
for subject in subjects:
 with np.load(SRC/f'{subject}_t30_15s.npz') as z:waves[subject]={k:z[k].copy() for k in ('time_s','ppg','pos','chrom')}
def bpm_fft(x,fs):
 f=np.fft.rfftfreq(len(x),1/fs);p=np.abs(np.fft.rfft(x-x.mean()))**2;ok=(f>=.6)&(f<=2)
 return float(f[np.flatnonzero(ok)[np.argmax(p[ok])]]*60)
def beats(x,fs):
 period=fs*60/bpm_fft(x,fs); peaks,_=find_peaks(x,distance=max(1,int(period*.6)))
 out=[]
 for a,b in zip(peaks[:-1],peaks[1:]):
  if .7*period<=b-a<=1.3*period:
   y=np.interp(phase,np.arange(b-a)/(b-a),x[a:b]);y=(y-y.mean())/max(y.std(),1e-12);out.append(y)
 return out
def bank(held):
 templates={'sine':np.cos(2*np.pi*phase),'harmonics':np.cos(2*np.pi*phase)+.3*np.cos(4*np.pi*phase+.6)+.1*np.cos(6*np.pi*phase+1)}
 dist=lambda p,c:((p-c+.5)%1)-.5
 templates['two_gaussians']=np.exp(-.5*(dist(phase,.15)/.07)**2)+.35*np.exp(-.5*(dist(phase,.48)/.11)**2)
 counts={}
 for method in ('pos','chrom'):
  parts=[]
  for person in subjects:
   if person==held:continue
   z=waves[person];parts+=beats(z[method],1/np.median(np.diff(z['time_s'])))
  if not parts:raise ValueError('No empirical beats')
  templates['empirical_'+method]=np.median(parts,axis=0);counts[method]=len(parts)
 return templates,counts
def main():
 rows=[]; scores=[];counts_all={}
 for held in subjects:
  templates,counts=bank(held);counts_all[held]=counts;z=waves[held];t=z['time_s']-z['time_s'][0];fs=1/np.median(np.diff(t));sos=butter(6,[.6,2],btype='bandpass',fs=fs,output='sos')
  for method in ('pos','chrom'):
   x=z[method]-z[method].mean();energy=np.dot(x,x);ref=bpm_fft(z['ppg'],fs)
   for name,shape in templates.items():
    best=(-np.inf,None)
    for rate in range(36,121):
     value=-np.inf
     for shift in np.arange(16)/16:
      y=np.interp((t*rate/60+shift)%1,phase,shape,period=1)
      # Empirical templates already came from filtered pulses.
      if not name.startswith('empirical_'):
       y=sosfiltfilt(sos,y)
      y-=y.mean();den=np.dot(y,y)
      fit=np.dot(x,y)**2/(energy*den) if den>1e-20 else -np.inf
      value=max(value,float(fit))
     scores.append(dict(subject=held,method=method,candidate=name,bpm=rate,r2=value))
     if value>best[0]:best=(value,rate)
    rows.append(dict(subject=held,method=method,candidate=name,bpm=best[1],r2=best[0],reference_ppg_fft=ref,abs_error=abs(best[1]-ref),baseline_fft=bpm_fft(x,fs)))
  print('Done',held,flush=True)
 for name,data in [('results.csv',rows),('scores.csv',scores)]:
  with (OUT/name).open('w',encoding='utf-8-sig',newline='') as f:
   w=csv.DictWriter(f,fieldnames=data[0]);w.writeheader();w.writerows(data)
 templates,_=bank('subject3');fig,axes=plt.subplots(5,1,figsize=(9,9),layout='constrained')
 for ax,(name,y) in zip(axes,templates.items()):
  ax.plot(phase,(y-y.mean())/y.std());ax.set_title(name);ax.set_ylabel('Amplitude');ax.grid(alpha=.2)
 axes[-1].set_xlabel('Fraction of one cycle');fig.savefig(OUT/'candidate_shapes.png',dpi=120)
 (OUT/'manifest.json').write_text(json.dumps(dict(subjects=subjects,window='saved 15s ending 30s; common support',bank_counts=counts_all,bootstrap='FFT period + peaks from training subjects only',phase_trials=16,bpm_grid='36..120 step 1',filter='analytic templates filtered once; empirical templates already filtered and not filtered again',warning='Three subjects and one overlapping window per person; filtered empirical shapes; candidate filter edges differ from original extraction; diagnostic, not validation'),indent=2),encoding='utf-8')
 print('Saved',OUT)

if __name__=='__main__': main()
