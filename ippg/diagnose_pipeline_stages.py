"""Read-only pipeline diagnostics: per-region RGB/masks and POS spectra."""
import json
from datetime import datetime
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, butter, freqz
import ippg_v2_refactored as core
from replay_realtime import replay_video,TimeBuffer,evaluate,load_reference
from benchmark_dwt_ab import ppg_reference

ROOT=Path(__file__).resolve().parents[1];DATA=Path(r'C:\Users\Akkarapon\Downloads\Project scius\dataset for ipph')
OUT=ROOT/'outputs'/('stage_diagnostic_'+datetime.now().strftime('%Y%m%d_%H%M%S'))
def z(x):
 x=np.asarray(x);return (x-x.mean())/x.std() if x.std()>1e-12 else x*0

def spectrum(x,fs):
 f=np.fft.rfftfreq(len(x)*32,1/fs);p=abs(np.fft.rfft(x-x.mean(),n=len(x)*32))**2
 return f,p

def main():
 OUT.mkdir(parents=True);results=[]
 for subject in ('subject5','subject11'):
  video=next(DATA.glob(f'{subject}-*/{subject}/vid.avi'));cfg=core.IPPGConfig(roi_method='legacy',pulse_method='pos',denoise='none',background_method='none',show_preview=False);ref=load_reference(video.parent/'ground_truth.txt')
  logs=[];snapshots={};current=[0.];orig_cal=core.AdaptiveSkinDetector.calibrate;orig_mean=core.mean_skin_rgb
  def cal(self,regions,t):
   current[0]=t
   return orig_cal(self,regions,t)
  def mean(region,geometry,label,detector,config):
   value,mask,count=orig_mean(region,geometry,label,detector,config)
   if 25<current[0]<=40:
    logs.append(dict(time=current[0],region=label,rgb=value.tolist(),pixels=count,coverage=count/float(mask.size),shape=list(mask.shape)))
    for sec in (26,30,35,39):
     if sec<=current[0]<sec+.05 and (sec,label) not in snapshots:
      snapshots[(sec,label)]=(region.copy(),mask.copy())
   return value,mask,count
  core.AdaptiveSkinDetector.calibrate=cal;core.mean_skin_rgb=mean
  meta={};buf=TimeBuffer(15);stream=replay_video(video,cfg,meta);row=None;s=None
  try:
   for obs in stream:
    if obs.time_s>40:
     row,s=evaluate(buf,40.,15,meta['fps'],cfg);break
    buf.append(obs)
  finally:
   stream.close();core.AdaptiveSkinDetector.calibrate=orig_cal;core.mean_skin_rgb=orig_mean
  if row is None or row['status']!='estimated':raise RuntimeError(str(row))
  t=s['time_seconds'];avail=ref[0][ref[0]<=40];n=min(len(t),np.searchsorted(t,avail[-1],side='right'));t=t[:n];fs=meta['fps'];raw_ppg,ppg,pb,peaks=ppg_reference(ref,t,40.,fs)
  raw=s['raw_pos'][:n];filtered=s['filtered'][:n]
  fig,axs=plt.subplots(5,1,figsize=(13,12),layout='constrained')
  regional={};stats={}
  for label,c in [('forehead','#bd6134'),('left_cheek','#377dc2'),('right_cheek','#329574')]:
   a=[r for r in logs if r['region']==label];tt=np.array([r['time'] for r in a]);rgb=np.array([r['rgb'] for r in a]);pixels=np.array([r['pixels'] for r in a]);regional[label]=dict(time=tt,rgb=rgb,pixels=pixels)
   axs[0].plot(tt,rgb[:,1],label=label,color=c);axs[1].plot(tt,pixels,label=label,color=c)
   stats[label]=dict(valid_frames=len(a),skin_pixels_min=int(pixels.min()),skin_pixels_max=int(pixels.max()),green_step_p95=float(np.percentile(abs(np.diff(rgb[:,1])),95)),coverage_min=min(r['coverage'] for r in a),coverage_max=max(r['coverage'] for r in a))
  axs[0].set_title('Regional green channel (actual pixel intensity)');axs[0].legend(ncol=3);axs[1].set_title('Selected skin pixel counts (each ROI)')
  for j,c in enumerate(('red','green','blue')):axs[2].plot(t,z(s['rgb'][:n,j]),color=c,label='RGB'[j])
  axs[2].set_title('Combined RGB (standardized per channel)');axs[2].legend(ncol=3)
  axs[3].plot(t,z(raw),color='#9462ac',label='POS before filtering');axs[3].set_title('POS output before bandpass (no DWT)')
  axs[4].plot(t,z(filtered),label='POS after bandpass',color='#2879b8');axs[4].plot(t,z(ppg),label='PPG reference (0.6–3 Hz)',color='#dd8a32',alpha=.8);axs[4].legend();axs[4].set_title('After filtering vs PPG (no time/sign adjustment)')
  for ax in axs:ax.grid(alpha=.2)
  axs[-1].set_xlabel('Nominal recording time (s)');fig.suptitle(f'{subject} | 25–40 s | baseline without DWT');fig.savefig(OUT/f'{subject}_stages.png',dpi=140);plt.close(fig)
  fig,axs=plt.subplots(3,1,figsize=(12,9),layout='constrained');spec_stats={}
  for ax,(name,x) in zip(axs,[('POS before filter',raw),('POS after filter',filtered),('PPG raw',raw_ppg)]):
   f,p=spectrum(x,fs);broad=(f>=.6)&(f<=3);inband=(f>=.6)&(f<=2);peak=60*f[broad][np.argmax(p[broad])];pick=60*f[inband][np.argmax(p[inband])]
   ax.plot(f[broad]*60,p[broad]/p[broad].max());ax.axvline(pb,color='#df8a22',ls='--',label=f'PPG peaks: {pb:.1f} BPM');ax.axvspan(120,180,color='gray',alpha=.15,label='Outside baseline search');ax.set(title=f'{name} | largest peak 36–180: {peak:.1f}; 36–120: {pick:.1f}',ylabel='Relative power');ax.grid(alpha=.2);ax.legend(fontsize=8)
   spec_stats[name]=dict(peak_36_180=peak,peak_36_120=pick,reference_power_to_broad_max=float(np.interp(pb/60,f,p)/p[broad].max()))
  axs[-1].set_xlabel('Frequency (BPM)');fig.savefig(OUT/f'{subject}_spectra.png',dpi=150);plt.close(fig)
  fig,axs=plt.subplots(4,6,figsize=(14,9),layout='constrained')
  for i,sec in enumerate((26,30,35,39)):
   for j,label in enumerate(('forehead','left_cheek','right_cheek')):
    item=snapshots.get((sec,label))
    for col in (2*j,2*j+1):axs[i,col].axis('off')
    if item:
     region,mask=item;axs[i,2*j].imshow(region[:,:,::-1]);axs[i,2*j+1].imshow(mask,cmap='gray',vmin=0,vmax=255);axs[i,2*j].set_title(f'{sec}s {label}',fontsize=8);axs[i,2*j+1].set_title('Selected skin',fontsize=8)
  fig.suptitle(subject+' | ROI crops and masks (patch sizes rescaled for display)');fig.savefig(OUT/f'{subject}_roi_masks.png',dpi=140);plt.close(fig)
  # Inspect POS overlap-add contribution counts without changing its algorithm.
  size=max(2,int(round(cfg.pos_window_seconds*fs)));contrib=np.convolve(np.ones(len(s['rgb'])-size+1),np.ones(size));alpha=[]
  rgb=s['rgb']
  for k in range(len(rgb)-size+1):
   cn=(rgb[k:k+size]/rgb[k:k+size].mean(axis=0)).T;first=cn[1]-cn[2];second=cn[1]+cn[2]-2*cn[0];alpha.append(float(first.std()/second.std()) if second.std()>cfg.pos_epsilon else 0.)
  b,a=butter(6,[.6,2],btype='bandpass',fs=fs);_,h=freqz(b,a,worN=[2*np.pi*(pb/60)/fs]);gain=float(abs(h[0])**2)
  result=dict(subject=subject,tracker=meta['tracker'],ppg_peak_bpm=float(pb),reference_ppg_fft_bpm=spec_stats['PPG raw']['peak_36_180'],rgb_pairwise_corr=np.corrcoef(s['rgb'][:n].T).tolist(),regions=stats,spectra=spec_stats,pos_alpha_min=float(min(alpha)),pos_alpha_max=float(max(alpha)),overlap_add_edge_counts=[float(contrib[0]),float(contrib[-1])],overlap_add_max=float(contrib.max()),ideal_filtfilt_amplitude_at_reference=gain,notes='One diagnostic window; no causal conclusion from correlations or selected snapshots. Reference peak estimator provisional.')
  results.append(result);np.savez_compressed(OUT/f'{subject}_arrays.npz',time=t,rgb=s['rgb'][:n],pos_raw=raw,pos_filtered=filtered,ppg_raw=raw_ppg,ppg_filtered=ppg,pos_alpha=alpha,overlap_counts=contrib)
  (OUT/f'{subject}_roi_log.json').write_text(json.dumps(logs),encoding='utf-8');print(json.dumps(result),flush=True)
 (OUT/'diagnostics.json').write_text(json.dumps(results,indent=2),encoding='utf-8');print('OUTPUT',OUT)
if __name__=='__main__':main()
