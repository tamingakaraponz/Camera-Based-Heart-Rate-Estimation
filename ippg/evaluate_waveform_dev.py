"""Development-only exploratory fixed-clock waveform metrics every five seconds."""
import csv, json, hashlib
from datetime import datetime
from pathlib import Path
import numpy as np
import ippg_v2_refactored as core
from benchmark_new_dataset import ROOT, development_inputs
from replay_realtime import TimeBuffer, replay_video, evaluate, load_reference, reference_window
from compare_dev_methods import chrom, reference_hr

DURATIONS=(5,10,15)
STEP=5.0
FIRST=20.0

def measure(buf, now, duration, fps, cfg, ref, device):
    row, stage=evaluate(buf,now,duration,fps,cfg)
    if row['status']!='estimated': raise ValueError('POS: '+row['reason'])
    t=stage['time_seconds']; signal=chrom(stage['rgb'],fps)
    signal=core.wavelet_denoise(signal,cfg,fps)[0]
    signal=core.bandpass_filter(signal,fps,cfg.forehead_low_hz,cfg.forehead_high_hz,cfg.forehead_filter_order)
    available=ref[0][ref[0]<=now]
    if len(available)==0: raise ValueError('no reference')
    n=min(len(t),len(signal),len(stage['filtered']),int(np.searchsorted(t,available[-1],side='right')))
    t=t[:n]
    if n<2: raise ValueError('too few common samples')
    _,raw,ppg,_=reference_window(ref,now,t,fps,cfg,.25)
    ppg=ppg[:n]; signals={'POS':stage['filtered'][:n], 'CHROM':signal[:n]}
    def spectrum(x):
        f,z=core.calculate_spectrum(t,x,fps)
        bpm=core.estimate_heart_rate(f,z,cfg.forehead_low_hz,cfg.forehead_high_hz)
        band=(f>=cfg.forehead_low_hz)&(f<=cfg.forehead_high_hz)
        power=np.abs(z[band])**2
        if np.sum(power)<=1e-20: raise ValueError('zero spectral power')
        return bpm,power/power.sum()
    pb,ps=spectrum(ppg)
    device_bpm=reference_hr(device,float(t[0]),float(t[-1]))
    result=[]
    for name,x in signals.items():
        if np.std(x)<1e-12 or np.std(ppg)<1e-12: raise ValueError('flat waveform')
        b,s=spectrum(x)
        result.append(dict(method=name,correlation=float(np.corrcoef(ppg,x)[0,1]),spectral_overlap=float(np.minimum(ps,s).sum()),ippg_bpm=b,ppg_fft_bpm=pb,abs_bpm_error=abs(b-pb),supplied_hr_mean=device_bpm,abs_bpm_error_supplied=abs(b-device_bpm) if device_bpm is not None else None))
    return t,result,ppg,signals

def main():
    inputs=development_inputs(ROOT/'new_dataset_split.json')
    out=ROOT/'outputs/waveform_metrics'/datetime.now().strftime('dev_%Y%m%d_%H%M%S_%f');out.mkdir(parents=True)
    cfg=core.IPPGConfig(max_duration_seconds=None,show_preview=False,pulse_method='pos',roi_method='legacy',denoise='dwt',background_method='none')
    rows=[]; failures=[]; clips=[]
    for item in inputs:
        print('Processing',item['subject'],flush=True)
        ref=load_reference(Path(item['reference'])); d=np.loadtxt(item['reference'],ndmin=2); device=(d[2]-d[2,0],d[1]);meta={};buffer=TimeBuffer(15);tick=FIRST;frames=0
        stream=replay_video(Path(item['video']),cfg,meta)
        try:
            for obs in stream:
                frames+=1
                while tick<obs.time_s-1e-9:
                    for duration in DURATIONS:
                        try:
                            t,vals,ppg,signals=measure(buffer,tick,duration,meta['fps'],cfg,ref,device)
                            np.savez_compressed(out/f"{item['subject']}_t{int(tick)}_{duration}s.npz",time_s=t,ppg=ppg,pos=signals['POS'],chrom=signals['CHROM'])
                            for val in vals: rows.append(dict(subject=item['subject'],end_s=tick,requested_window_s=duration,common_start_s=float(t[0]),common_end_s=float(t[-1]),samples=len(t),**val))
                        except (ValueError,RuntimeError) as e: failures.append(dict(subject=item['subject'],end_s=tick,requested_window_s=duration,reason=str(e)))
                    tick+=STEP
                buffer.append(obs)
        finally: stream.close()
        clips.append(dict(subject=item['subject'],frames_read=frames,expected_frames=int(__import__('cv2').VideoCapture(item['video']).get(__import__('cv2').CAP_PROP_FRAME_COUNT))))
    fields=['subject','end_s','requested_window_s','common_start_s','common_end_s','samples','method','correlation','spectral_overlap','ippg_bpm','ppg_fft_bpm','abs_bpm_error','supplied_hr_mean','abs_bpm_error_supplied']
    with (out/'windows.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
    summaries=[]
    for subject in sorted(i['subject'] for i in inputs):
      for duration in DURATIONS:
       for method in ('POS','CHROM'):
        r=[x for x in rows if x['subject']==subject and x['requested_window_s']==duration and x['method']==method]
        attempted=len(r)+sum(x['subject']==subject and x['requested_window_s']==duration for x in failures)
        if not r: continue
        summaries.append(dict(subject=subject,window_s=duration,method=method,success=len(r),attempted=attempted,mean_r=float(np.mean([x['correlation'] for x in r])),median_r=float(np.median([x['correlation'] for x in r])),mean_overlap=float(np.mean([x['spectral_overlap'] for x in r])),mean_abs_bpm_error=float(np.mean([x['abs_bpm_error'] for x in r]))))
    with (out/'summary.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['subject','window_s','method','success','attempted','mean_r','median_r','mean_overlap','mean_abs_bpm_error']);w.writeheader();w.writerows(summaries)
    info=dict(status='complete' if all(x['frames_read']==x['expected_frames'] for x in clips) else 'incomplete_decode',subjects=[i['subject'] for i in inputs],held_out_opened=False,clips=clips,failures=failures,first_update=FIRST,step=STEP,durations=DURATIONS,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),notes=['Nominal time only; no lag or polarity adjustment','CHROM support truncates all signals; filtering applied separately before truncation except PPG','Same common-grid FFT as BPM; spectral overlap is normalized common-band power overlap','Overlapping windows are not independent; exploratory development set, not clinical accuracy'])
    (out/'manifest.json').write_text(json.dumps(info,indent=2,ensure_ascii=False),encoding='utf-8')
    lines=['# เปรียบเทียบคลื่นหลายช่วง: ผลสำรวจกลุ่มพัฒนา','','คำนวณทุก 5 วินาทีตั้งแต่เวลา 20 วินาที หน้าต่างที่ขอ 5/10/15 วินาที; หน้าต่างซ้อนกันได้ เวลาเป็นเวลาที่ไฟล์ระบุ ยังไม่ยืนยันการตรงกันจริง','', '| คน | ช่วง | วิธี | ช่วงที่ใช้ได้/ทั้งหมด | correlation เฉลี่ย | correlation มัธยฐาน | ความถี่ซ้อนทับเฉลี่ย | BPM ต่างจาก PPG FFT เฉลี่ย |','|---|---:|---|---:|---:|---:|---:|---:|']
    for x in summaries: lines.append(f"| {x['subject']} | {x['window_s']} s | {x['method']} | {x['success']}/{x['attempted']} | {x['mean_r']:.3f} | {x['median_r']:.3f} | {x['mean_overlap']:.3f} | {x['mean_abs_bpm_error']:.2f} |")
    lines+=['','Correlation แบบมีเครื่องหมายโดยไม่เลื่อนเวลาหรือกลับขั้ว; ค่าเป็นลบอาจเกิดจากขั้ว/เวลา ไม่แปลว่าไม่มีสัญญาณชีพจร การซ้อนทับความถี่ 0–1 ไม่ใช่เปอร์เซ็นต์ความแม่นยำ','', 'ทุกวิธีใช้ช่วงเวลาร่วมจริงที่ CHROM ตัดปลาย; PPG กรองบนช่วงร่วม ส่วน POS และ CHROM กรองก่อนตัด ขอบกรองและตัวกรองร่วมอาจกระทบความคล้าย ช่วง 5 วินาทีมีจุดความถี่หยาบกว่า', '', 'ผลจากสามคนและหน้าต่างซ้อนกัน เป็นการสำรวจ ไม่ใช่การประเมินประชากรหรือหลักฐาน real-time สด; ไฟล์ windows.csv รวมทั้งค่า BPM ที่ไฟล์เซนเซอร์ให้มา', '', f"ช่วงที่คำนวณไม่ได้: {len(failures)}; ดูเหตุผลใน manifest.json"]
    (out/'RESULT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print('Saved:',out,flush=True)
if __name__=='__main__':main()
