"""Five-subject comparison; PPG peak timing is the sole HR reference."""
import csv,ctypes,json,os,time
from ctypes import wintypes
from pathlib import Path
import numpy as np
from scipy.signal import find_peaks
import ippg_v2_refactored as core
from replay_realtime import TimeBuffer,replay_video,evaluate,load_reference,reference_window
from compare_dev_methods import chrom
ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'outputs/fft_ppg_peaks_five';OUT.mkdir(parents=True,exist_ok=True)
DATA=Path(r'C:\Users\Akkarapon\Downloads\Project scius\dataset for ipph')
class MemoryCounters(ctypes.Structure):
 _fields_=[('cb',wintypes.DWORD),('PageFaultCount',wintypes.DWORD),('PeakWorkingSetSize',ctypes.c_size_t),('WorkingSetSize',ctypes.c_size_t),('QuotaPeakPagedPoolUsage',ctypes.c_size_t),('QuotaPagedPoolUsage',ctypes.c_size_t),('QuotaPeakNonPagedPoolUsage',ctypes.c_size_t),('QuotaNonPagedPoolUsage',ctypes.c_size_t),('PagefileUsage',ctypes.c_size_t),('PeakPagefileUsage',ctypes.c_size_t)]
def memory():
 m=MemoryCounters();m.cb=ctypes.sizeof(m)
 ctypes.windll.kernel32.GetCurrentProcess.restype=wintypes.HANDLE
 proc=ctypes.windll.kernel32.GetCurrentProcess()
 ctypes.windll.psapi.GetProcessMemoryInfo.argtypes=[wintypes.HANDLE,ctypes.POINTER(MemoryCounters),wintypes.DWORD]
 if not ctypes.windll.psapi.GetProcessMemoryInfo(proc,ctypes.byref(m),m.cb):raise OSError('Process memory unavailable')
 return m.PeakWorkingSetSize/1048576

def peak_bpm(x,fps):
 sd=float(np.std(x))
 if sd<1e-12:raise ValueError('flat_ppg')
 peaks,_=find_peaks(x,distance=int(np.ceil(fps*.5)),prominence=.35*sd)
 if len(peaks)<3:raise ValueError('fewer_than_3_ppg_peaks')
 intervals=np.diff(peaks)/fps
 if np.any((intervals<.5)|(intervals>1/.6)):raise ValueError('ppg_intervals_outside_range')
 return 60/float(np.mean(intervals)),len(peaks)

def get_bpm(method,t,x,fps):
 if method=='sine':
  centered=x-x.mean();best=(-np.inf,None)
  for bpm in range(36,121):
   design=np.column_stack((np.sin(2*np.pi*bpm/60*t),np.cos(2*np.pi*bpm/60*t),np.ones(len(t))))
   residual=centered-design@np.linalg.lstsq(design,centered,rcond=None)[0]
   score=-np.dot(residual,residual)
   if score>best[0]:best=(score,bpm)
  return float(best[1])
 uniform=np.arange(t[0],t[-1],1/fps);v=np.interp(uniform,t,x);v-=v.mean()
 size=len(v)*(32 if method=='dense_fft' else 1)
 spectrum=np.fft.rfft(v,n=size);frequency=np.fft.rfftfreq(size,d=1/fps)
 return float(core.estimate_heart_rate(frequency,spectrum,.6,2))

rows=[];failures=[];started=time.perf_counter();cpu_start=time.process_time();peak_start=memory();counts={}
for subject in ('subject3','subject5','subject8','subject9','subject10'):
 video=next(DATA.glob(f'{subject}-*/{subject}/vid.avi'));reference=load_reference(video.parent/'ground_truth.txt');cfg=core.IPPGConfig(roi_method='legacy',pulse_method='pos',denoise='dwt',background_method='none',show_preview=False);buf=TimeBuffer(15);meta={};tick=20.;stream=replay_video(video,cfg,meta);frames=0
 try:
  for obs in stream:
   frames+=1
   while tick<obs.time_s-1e-9 and tick<=50:
    for seconds in (5,10,15):
     try:
      row,stage=evaluate(buf,tick,seconds,meta['fps'],cfg)
      if row['status']!='estimated':raise ValueError('pos_'+row['reason'])
      c=chrom(stage['rgb'],meta['fps']);c=core.wavelet_denoise(c,cfg,meta['fps'])[0];c=core.bandpass_filter(c,meta['fps'],.6,2,6)
      n=min(len(c),len(stage['filtered']));grid=stage['time_seconds'][:n];available=reference[0][reference[0]<=tick]
      if len(available)==0:raise ValueError('no_ppg')
      n=min(n,int(np.searchsorted(grid,available[-1],side='right')));grid=grid[:n]
      if n<2:raise ValueError('short_common_support')
      _,_,ppg,_=reference_window(reference,tick,grid,meta['fps'],cfg,.25)
      bpm_ref,n_peaks=peak_bpm(ppg,meta['fps']);t=grid-grid[0]
      for name,signal in [('POS',stage['filtered'][:n]),('CHROM',c[:n])]:
       corr=float(np.corrcoef(signal,ppg)[0,1]);f=np.fft.rfftfreq(n,1/meta['fps']);band=(f>=.6)&(f<=2);px=np.abs(np.fft.rfft(signal-signal.mean()))[band]**2;py=np.abs(np.fft.rfft(ppg-ppg.mean()))[band]**2;overlap=float(np.minimum(px/px.sum(),py/py.sum()).sum())
       for method in ('fft','dense_fft','sine'):
        wall=time.perf_counter();cpu=time.process_time();bpm=get_bpm(method,t,signal,meta['fps']);cpu_ms=1000*(time.process_time()-cpu);wall_ms=1000*(time.perf_counter()-wall)
        rows.append(dict(subject=subject,end_s=tick,window_s=seconds,actual_start_s=float(grid[0]),actual_end_s=float(grid[-1]),method=name,estimator=method,bpm=bpm,reference_ppg_peaks=bpm_ref,reference_peaks=n_peaks,error_bpm=abs(bpm-bpm_ref),waveform_correlation=corr,spectral_overlap=overlap,wall_ms=wall_ms,cpu_ms=cpu_ms))
     except (ValueError,RuntimeError) as e:failures.append(dict(subject=subject,end_s=tick,window_s=seconds,reason=str(e)))
    print(subject,'scored',tick,flush=True);tick+=10
   if tick>50:break
   buf.append(obs)
 finally:stream.close()
 counts[subject]=frames
with (OUT/'windows.csv').open('w',encoding='utf-8-sig',newline='') as f:
 w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
wall=time.perf_counter()-started;cpu=time.process_time()-cpu_start
info=dict(subjects=list(counts),frames=counts,rows=len(rows),failures=failures,run_wall_s=wall,run_process_cpu_s=cpu,logical_processors=os.cpu_count(),cpu_one_core_equivalent_pct=100*cpu/wall,cpu_total_capacity_pct=100*cpu/wall/(os.cpu_count() or 1),peak_working_set_mib=memory(),initial_peak_working_set_mib=peak_start,measurement_scope='Full single-process replay includes video decoding, tracking, wavelet, all three estimators and PPG evaluation; not standalone webcam throughput. Estimator timings exclude tracking/filtering and PPG reference.',reference='Mean inter-peak interval of filtered PPG; >=3 peaks, prominence .35 SD, distance .5s, intervals 0.5–1.667s; not independently validated; timestamp alignment nominal.',notes='FFT 32x zero padding; sine least squares step 1 BPM. Results only windows with valid PPG peaks; subject10 now development, subject11 not opened.')
(OUT/'manifest.json').write_text(json.dumps(info,indent=2),encoding='utf-8');print(json.dumps(info,indent=2),flush=True)
