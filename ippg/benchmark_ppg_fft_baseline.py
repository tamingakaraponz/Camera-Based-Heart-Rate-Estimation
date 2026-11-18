"""Re-evaluate saved no-DWT signals using filtered PPG FFT as reference."""
import argparse,csv,hashlib,json
from datetime import datetime
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import ippg_v2_refactored as core
from replay_realtime import load_reference,reference_window

ROOT=Path(__file__).resolve().parents[1]
def table(path,rows):
    if not rows:return
    with path.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
def summarize(rows):
    errors=np.array([r['difference_bpm'] for r in rows])
    return dict(valid=len(rows),mae_bpm=float(np.mean(abs(errors))),rmse_bpm=float(np.sqrt(np.mean(errors**2))),bias_bpm=float(np.mean(errors)),mean_waveform_r=float(np.mean([r['waveform_r'] for r in rows])),mean_spectral_overlap=float(np.mean([r['spectral_overlap'] for r in rows])))
def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--source',type=Path,default=ROOT/'outputs/band_ab_20261116_150137_572560');args=parser.parse_args();src=args.source.resolve();manifest=json.loads((src/'manifest.json').read_text(encoding='utf-8'));cfg=core.IPPGConfig(**manifest['configs']['new'])
    if cfg.denoise!='none' or cfg.forehead_low_hz!=.6 or cfg.forehead_high_hz!=3:raise ValueError('Expected no DWT and 0.6–3 Hz source')
    out=ROOT/'outputs'/('ppg_fft_baseline_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'));out.mkdir();rows=[];skips=[];hashes={};summaries=[]
    oldfail={(r['subject'],int(r.get('end_s',-1)),int(r.get('window_s',-1))):r['reason'] for r in manifest['failures']}
    for clip in manifest['clips']:
        subject=clip['subject'];fps=clip['fps'];video=Path(clip['video']);reference=load_reference(video.parent/'ground_truth.txt');figw,axw=plt.subplots(4,3,figsize=(16,11),layout='constrained');figs,axs=plt.subplots(4,3,figsize=(16,11),layout='constrained')
        for i,end in enumerate((20,30,40,50)):
            for j,seconds in enumerate((5,10,15)):
                wave=axw[i,j];spec=axs[i,j];path=src/f'{subject}_{end}_{seconds}.npz'
                try:
                    if not path.exists():raise ValueError(oldfail.get((subject,end,seconds),'source_window_missing'))
                    hashes[path.name]=hashlib.sha256(path.read_bytes()).hexdigest()
                    with np.load(path,allow_pickle=False) as a:t=a['time'].copy();x=a['new'].copy();stored_ppg=a['ppg_filtered'].copy()
                    if len(t)<3 or np.any(np.diff(t)<=0) or not np.all(np.isfinite(x)):raise ValueError('invalid input')
                    _,raw,y,last=reference_window(reference,end,t,fps,cfg,.25)
                    np.testing.assert_allclose(y,stored_ppg,rtol=1e-10,atol=1e-10)
                    f,ix=core.calculate_spectrum(t,x,fps);rf,ry=core.calculate_spectrum(t,y,fps);np.testing.assert_array_equal(f,rf)
                    ib=core.estimate_heart_rate(f,ix,.6,3);pb=core.estimate_heart_rate(f,ry,.6,3)
                    grid=np.arange(t[0],t[-1],1/fps);xx=np.interp(grid,t,x);yy=np.interp(grid,t,y)
                    if min(np.std(xx),np.std(yy))<1e-12:raise ValueError('flat waveform')
                    zx=(xx-xx.mean())/xx.std();zy=(yy-yy.mean())/yy.std();band=(f>=.6)&(f<=3);px=abs(ix[band])**2;py=abs(ry[band])**2
                    if min(px.sum(),py.sum())<=1e-20:raise ValueError('zero band power')
                    row=dict(subject=subject,end_s=end,window_s=seconds,actual_start_s=float(grid[0]),actual_end_s=float(grid[-1]),fft_samples=len(grid),fps=fps,ippg_fft_bpm=ib,ppg_fft_bpm=pb,difference_bpm=ib-pb,abs_difference_bpm=abs(ib-pb),fft_bin_spacing_bpm=float((f[1]-f[0])*60),waveform_r=float(np.mean(zx*zy)),waveform_z_rmse=float(np.sqrt(np.mean((zx-zy)**2))),spectral_overlap=float(np.minimum(px/px.sum(),py/py.sum()).sum()),ppg_last_available_s=last)
                    rows.append(row);np.savez_compressed(out/path.name,time=grid,ippg=xx,ppg=yy,frequency_hz=f,ippg_spectrum=ix,ppg_spectrum=ry)
                    wave.plot(grid,zx,label='iPPG',color='#367bbb',lw=.85);wave.plot(grid,zy,label='PPG',color='#de9030',lw=.85,alpha=.8);wave.set_title(f'{end-seconds}–{end}s | r={row["waveform_r"]:.2f}',fontsize=9)
                    spec.plot(f[band]*60,px/px.max(),color='#367bbb',label='iPPG');spec.plot(f[band]*60,py/py.max(),color='#de9030',label='PPG');spec.set_title(f'{end-seconds}–{end}s | iPPG {ib:.1f}, PPG {pb:.1f}',fontsize=9)
                except ValueError as error:
                    reason=str(error);skips.append(dict(subject=subject,end_s=end,window_s=seconds,reason=reason))
                    for ax in (wave,spec):ax.text(.5,.5,'SKIPPED: '+reason,ha='center',va='center',fontsize=8,transform=ax.transAxes);ax.set_title(f'End {end}s / {seconds}s',fontsize=9)
                for ax in (wave,spec):ax.grid(alpha=.2)
                if i==3:wave.set_xlabel('Nominal time (s)');spec.set_xlabel('Frequency (BPM)')
        axw[0,0].legend(fontsize=7);axs[0,0].legend(fontsize=7);figw.suptitle(subject+': waveform comparison; no time/sign adjustment');figs.suptitle(subject+': standard FFT on identical common grid; each spectrum normalized independently');figw.savefig(out/f'{subject}_waveforms.png',dpi=130);figs.savefig(out/f'{subject}_spectra.png',dpi=130);plt.close(figw);plt.close(figs)
        own=[r for r in rows if r['subject']==subject]
        if own:summaries.append(dict(subject=subject,attempted=12,**summarize(own)))
        print(subject,'valid',len(own),flush=True)
    groups=[]
    for seconds in (5,10,15):
        own=[r for r in rows if r['window_s']==seconds]
        if own:groups.append(dict(window_s=seconds,attempted=24,**summarize(own)))
    table(out/'windows.csv',rows);table(out/'skipped.csv',skips);table(out/'summary_subjects.csv',summaries);table(out/'summary_durations.csv',groups)
    report=dict(status='complete',source=str(src),source_manifest_sha256=hashlib.sha256((src/'manifest.json').read_bytes()).hexdigest(),source_arrays_sha256=hashes,script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),config=vars(cfg),reference='PPG: bandpass .6–3 Hz order3 and standard FFT; no peak detection or supplied HR in BPM comparison',ippg='Saved POS no DWT, bandpass .6–3 Hz order6; standard FFT',timing='Nominal frame/FPS vs sensor timestamps; dataset-declared synchronized, residual offset unmeasured; crop common support only; gaps >.25s skipped',limitations=['Source reference windows were previously retained using a peak-based pipeline: inherited missing windows reported; current validity uses timestamp support and gaps, not peak-count or peak agreement.','Overlapping development windows; not independent validation.','No new video/ROI extraction; isolates reference change.','Standard FFT frequency spacing varies with duration; bin agreement does not prove waveform accuracy.'],attempted=72,skipped=skips,overall_window_weighted=summarize(rows),macro_subject_mae_bpm=float(np.mean([r['mae_bpm'] for r in summaries])))
    (out/'manifest.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    lines=['# ผลตั้งต้นใหม่: ใช้ FFT ของ PPG เป็นค่าอ้างอิง','','ใช้ iPPG POS ไม่ผ่าน DWT, bandpass/ค้นหา 0.6–3 Hz; PPG bandpass 0.6–3 Hz แล้ว FFT ปกติบนแกนเวลาร่วมเดียวกัน ไม่ใช้การนับยอดหรือแถว BPM ในไฟล์เป็นคำตอบ','','| คน | ใช้ได้/12 | MAE BPM | RMSE BPM | correlation คลื่น | ความถี่ซ้อนทับ |','|---|---:|---:|---:|---:|---:|']
    for r in summaries:lines.append(f"| {r['subject']} | {r['valid']}/12 | {r['mae_bpm']:.2f} | {r['rmse_bpm']:.2f} | {r['mean_waveform_r']:.3f} | {r['mean_spectral_overlap']:.3f} |")
    lines+=['','| ช่วง | ใช้ได้/24 | MAE BPM | RMSE BPM |','|---|---:|---:|---:|']
    for r in groups:lines.append(f"| {r['window_s']} s | {r['valid']}/24 | {r['mae_bpm']:.2f} | {r['rmse_bpm']:.2f} |")
    total=report['overall_window_weighted'];lines+=['',f"รวม {len(rows)}/72 ช่วง: MAE {total['mae_bpm']:.2f} BPM, RMSE {total['rmse_bpm']:.2f} BPM; ค่าเฉลี่ย MAE ให้น้ำหนักแต่ละคนเท่ากัน {report['macro_subject_mae_bpm']:.2f} BPM",'','MAE = ส่วนต่างสัมบูรณ์เฉลี่ย; RMSE ให้น้ำหนักข้อผิดพลาดสูงมากขึ้น correlation มีเครื่องหมายและไม่ได้กลับขั้ว/เลื่อนเวลา ความถี่ซ้อนทับอยู่ในช่วง 0–1 ไม่ใช่เปอร์เซ็นต์ความแม่นยำ','', '## ช่วงที่ข้าม']
    for r in skips:lines.append(f"- {r['subject']} จบ {r['end_s']}s หน้าต่าง {r['window_s']}s: {r['reason']}")
    lines+=['','## กราฟคลื่นและความถี่']
    for r in summaries:
        s=r['subject'];lines.append(f'- {s}: [คลื่น]({s}_waveforms.png), [ความถี่]({s}_spectra.png)')
    lines+=['','## ข้อจำกัดและวิธีตรวจซ้ำ','', 'ใช้สัญญาณที่บันทึกไว้จากรอบเปรียบเทียบช่วงกรองเพื่อเปลี่ยนเฉพาะ reference ไม่ได้อ่านวิดีโอหรือเลือก ROI ใหม่ ตรวจซ้ำ PPG จากไฟล์ต้นฉบับและตรวจว่าตรงกับสัญญาณที่บันทึกไว้; FFT ทั้งสองฝั่งใช้ frequency grid เดียวกัน ช่วงกรอง iPPG order6 และ PPG order3 จึงไม่ได้อ้างว่าใช้ filter เดียวกันทั้งหมด','', 'เวลาจับคู่ตามไฟล์และ frame/FPS; dataset ระบุ synchronized แต่ยังไม่ได้วัด residual offset ห้ามตีความ correlation ต่ำว่า iPPG ไม่พบชีพจรทันที หน้าต่างซ้อนกันและข้อมูลเคยใช้พัฒนาแล้ว ผลนี้เป็น baseline ของชุดนี้ ไม่ใช่การรับรองทั่วไป','', 'windows.csv เก็บ BPM เวลา จำนวน sample และความละเอียด FFT ต่อช่วง; NPZ เก็บสัญญาณและ spectrum ที่ใช้คำนวณจริง; manifest.json เก็บ config, hashes และวิธีคำนวณ ผลนี้ไม่ควรเทียบตรงกับ error รอบที่ใช้ PPG peaks เป็น reference']
    (out/'RESULT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8');print('OUTPUT',out);print(json.dumps(dict(subjects=summaries,durations=groups,overall=total,skipped=skips),indent=2))
if __name__=='__main__':main()
