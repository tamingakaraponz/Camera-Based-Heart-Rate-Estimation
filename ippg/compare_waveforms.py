"""Visual comparison of POS, CHROM and PPG on predetermined development windows."""
import csv, json, hashlib
from datetime import datetime
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import ippg_v2_refactored as core
from benchmark_new_dataset import ROOT, development_inputs
from replay_realtime import TimeBuffer, replay_video, evaluate, load_reference, reference_window
from compare_dev_methods import chrom, reference_hr

END = 30.0
DURATIONS = (5, 10, 15)

def zscore(x):
    sd = np.std(x)
    if sd < 1e-12: raise ValueError('flat signal')
    return (x - np.mean(x)) / sd

def window_result(buf, fps, cfg, ref, device, duration, out, subject):
    row, stage = evaluate(buf, END, duration, fps, cfg)
    if row['status'] != 'estimated':
        raise ValueError(f"POS={row['reason']}")
    t, rgb = stage['time_seconds'], stage['rgb']
    c = chrom(rgb, fps)
    c = core.wavelet_denoise(c, cfg, fps)[0]
    c = core.bandpass_filter(c, fps, cfg.forehead_low_hz, cfg.forehead_high_hz, cfg.forehead_filter_order)
    # CHROM overlap-add omits its trailing partial segment. Restrict all plotted
    # signals and FFTs to the same observed samples, preserving nominal times.
    available = ref[0][ref[0] <= END]
    if not len(available): raise ValueError('no reference samples')
    n = min(len(t), len(c), len(stage['filtered']), int(np.searchsorted(t, available[-1], side='right')))
    t = t[:n]
    if len(t) < 2: raise ValueError('insufficient common samples')
    _, ppg_raw, ppg_filtered, _ = reference_window(ref, END, t, fps, cfg, .25)
    sig = {'PPG':ppg_filtered, 'POS':stage['filtered'][:n], 'CHROM':c[:n]}
    normalized = {k:zscore(v) for k,v in sig.items()}
    spectra = {}; bpms = {}
    for name,x in sig.items():
        f, spectrum = core.calculate_spectrum(t, x, fps)
        spectra[name] = (f*60, np.abs(spectrum)**2)
        bpms[name] = core.estimate_heart_rate(f, spectrum, cfg.forehead_low_hz, cfg.forehead_high_hz)
    device_bpm = reference_hr(device, float(t[0]), float(t[-1]))
    fig,axes=plt.subplots(2,1,figsize=(14,9),layout='constrained',gridspec_kw={'height_ratios':[2,1]})
    colors={'PPG':'#333333','POS':'#1575bd','CHROM':'#e36f12'}
    for name in ('PPG','POS','CHROM'):
        axes[0].plot(t,normalized[name],label=name,lw=1.2,color=colors[name],alpha=.85)
        f,p=spectra[name]; mask=(f>=36)&(f<=150); maximum=np.max(p[mask])
        axes[1].plot(f[mask],p[mask]/maximum,label=name,color=colors[name],lw=1.5)
    axes[0].set(title=f'{subject} | requested {duration} s ending at {END:.0f} s | common samples {t[0]:.2f}–{t[-1]:.2f} s',xlabel='Time (s)',ylabel='Standardized amplitude')
    dev_label = f'{device_bpm:.1f}' if device_bpm is not None else 'unavailable'
    axes[1].set(title=f"FFT BPM on common samples: PPG {bpms['PPG']:.1f} | POS {bpms['POS']:.1f} | CHROM {bpms['CHROM']:.1f} | supplied HR mean {dev_label}",xlabel='Frequency (cycles/min)',ylabel='Relative power')
    for ax in axes: ax.grid(alpha=.25); ax.legend(ncol=3)
    name=f'{subject}_{duration}s.png'; fig.savefig(out/name,dpi=130); plt.close(fig)
    np.savez_compressed(out/f'{subject}_{duration}s.npz',time_s=t,**{f'{k.lower()}_filtered':v for k,v in sig.items()},ppg_raw=ppg_raw)
    return dict(subject=subject,window_seconds=duration,requested_end_s=END,common_start_s=float(t[0]),common_end_s=float(t[-1]),common_samples=n,ppg_fft_bpm=bpms['PPG'],pos_fft_bpm=bpms['POS'],chrom_fft_bpm=bpms['CHROM'],supplied_hr_mean_bpm=device_bpm,plot=name)

def main():
    inputs=development_inputs(ROOT/'new_dataset_split.json')
    out=ROOT/'outputs/waveform_comparison'/datetime.now().strftime('comparison_%Y%m%d_%H%M%S_%f');out.mkdir(parents=True)
    cfg=core.IPPGConfig(max_duration_seconds=None,show_preview=False,pulse_method='pos',roi_method='legacy',denoise='dwt',background_method='none')
    rows=[]; failures=[]
    for item in inputs:
        print('Processing', item['subject'],flush=True)
        ref=load_reference(Path(item['reference'])); data=np.loadtxt(item['reference'],ndmin=2); device=(data[2]-data[2,0],data[1])
        meta={}; buf=TimeBuffer(15); stream=replay_video(Path(item['video']),cfg,meta)
        try:
            for obs in stream:
                if obs.time_s > END:
                    for seconds in DURATIONS:
                        try: rows.append(window_result(buf,meta['fps'],cfg,ref,device,seconds,out,item['subject']))
                        except (ValueError,RuntimeError) as e: failures.append(dict(subject=item['subject'],window_seconds=seconds,reason=str(e)))
                    break
                buf.append(obs)
            else: failures.append(dict(subject=item['subject'],window_seconds='all',reason='video_ended_before_30s'))
        finally: stream.close()
    keys=['subject','window_seconds','requested_end_s','common_start_s','common_end_s','common_samples','ppg_fft_bpm','pos_fft_bpm','chrom_fft_bpm','supplied_hr_mean_bpm','plot']
    with (out/'results.csv').open('w',encoding='utf-8-sig',newline='') as fp:
        w=csv.DictWriter(fp,fieldnames=keys);w.writeheader();w.writerows(rows)
    info=dict(status='complete' if len(rows)==9 else 'incomplete',subjects=[i['subject'] for i in inputs],held_out_opened=False,requested_end_s=END,windows_s=DURATIONS,failures=failures,code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),caveats=['Nominal clock alignment not physically verified','CHROM drops trailing partial section; all plotted signals use common samples','Filtered separately on original method windows before truncation; edge effects differ','No lag alignment or sign inversion; each waveform standardized','Individual spectrum normalized independently; frequencies from same common samples','Exploratory fixed windows, not accuracy estimates or representative assessment'])
    (out/'manifest.json').write_text(json.dumps(info,indent=2,ensure_ascii=False),encoding='utf-8')
    lines=['# กราฟเปรียบเทียบคลื่น iPPG และ PPG','','เลือกช่วงสิ้นสุดที่ 30 วินาทีล่วงหน้าเหมือนกันทุกคน ไม่ได้เลือกจากผล; แกนบนสำคัญที่สุด: PPG สีดำ, POS สีฟ้า, CHROM สีส้ม','', '| คน | ช่วงที่ขอ | ช่วงข้อมูลร่วมจริง (s) | PPG FFT | POS FFT | CHROM FFT | BPM ในไฟล์เฉลี่ย | กราฟ |','|---|---:|---:|---:|---:|---:|---:|---|']
    for r in rows:
        d='ไม่พร้อม' if r['supplied_hr_mean_bpm'] is None else f"{r['supplied_hr_mean_bpm']:.1f}"
        lines.append(f"| {r['subject']} | {r['window_seconds']} s | {r['common_start_s']:.2f}–{r['common_end_s']:.2f} | {r['ppg_fft_bpm']:.1f} | {r['pos_fft_bpm']:.1f} | {r['chrom_fft_bpm']:.1f} | {d} | [{r['plot']}]({r['plot']}) |")
    lines += ['', 'กราฟบนปรับขนาดแต่ละคลื่นเพื่อดูจังหวะ ไม่ได้บ่งบอกขนาดสัญญาณจริง ไม่เลื่อนเวลาหรือกลับขั้ว; กราฟล่างปรับความสูงแต่ละเส้นแยกกัน ไม่เปรียบเทียบกำลังจริง', '', 'CHROM ตัดท้ายช่วงที่ไม่ครบ ทำให้ทุกวิธีแสดงเฉพาะเวลาที่ซ้อนกันจริง; แต่ละวิธีถูกกรองก่อนตัด จึงอาจมีผลขอบต่างกัน ค่า BPM คำนวณใหม่บนช่วงร่วม จึงไม่จำเป็นต้องเท่าผล benchmark เดิม', '', 'เวลาจากไฟล์ยังไม่ยืนยันว่าตรงกันทางกายภาพ; ตัวกรองร่วมอาจทำให้คลื่นดูคล้ายกัน และกราฟตัวอย่างไม่ใช่หลักฐานความแม่นยำโดยรวม','',f'ช่วงที่ทำไม่ได้: {json.dumps(failures,ensure_ascii=False)}']
    (out/'RESULT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print('Saved:',out,flush=True)
if __name__=='__main__': main()
