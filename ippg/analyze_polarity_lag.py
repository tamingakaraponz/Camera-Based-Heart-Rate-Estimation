"""Exploratory bounded shift/sign sensitivity, not validation or calibration."""
import csv, json
from datetime import datetime
from pathlib import Path
import numpy as np
from scipy.stats import pearsonr
ROOT=Path(__file__).resolve().parents[1]
LAG_MAX_SECONDS=.3

def correlations(ref,x,t):
    dt=float(np.median(np.diff(t)))
    radius=int(np.floor((LAG_MAX_SECONDS+1e-9)/dt))
    if radius<1: raise ValueError('sampling interval too long')
    result=[]
    for k in range(-radius,radius+1):
        # Compare reference[i] with input[i+k] on overlap only. Positive k
        # means the input occurs later in the stored sample sequence.
        if k>0: a,b=ref[:-k],x[k:]
        elif k<0: a,b=ref[-k:],x[:k]
        else: a,b=ref,x
        if min(np.std(a),np.std(b))<1e-12: raise ValueError('flat waveform')
        result.append((k*dt,float(pearsonr(a,b).statistic)))
    base=result[radius][1]
    no_flip=max(result,key=lambda a:a[1])
    flip=max(((lag,-r) for lag,r in result),key=lambda a:a[1])
    return dict(original=base,inverted=-base,lag_best=no_flip[0],lag_best_r=no_flip[1],inverted_lag_best=flip[0],inverted_lag_best_r=flip[1],lag_limit_actual=radius*dt)

def main():
    dirs=sorted((ROOT/'outputs/waveform_metrics').glob('dev_*'),key=lambda p:p.name)
    src=dirs[-1]; manifest=json.loads((src/'manifest.json').read_text(encoding='utf-8'))
    if manifest['status']!='complete' or manifest['held_out_opened']:raise ValueError('invalid source')
    with (src/'windows.csv').open(encoding='utf-8-sig',newline='') as f: source=list(csv.DictReader(f))
    out=ROOT/'outputs/polarity_lag_analysis'/datetime.now().strftime('dev_%Y%m%d_%H%M%S_%f');out.mkdir(parents=True)
    records=[]
    for r in source:
        path=src/f"{r['subject']}_t{int(float(r['end_s']))}_{r['requested_window_s']}s.npz"
        with np.load(path) as z:
            t=z['time_s'];res=correlations(z['ppg'],z[r['method'].lower()],t)
        if abs(res['original']-float(r['correlation']))>1e-10:raise ValueError('original correlation mismatch')
        records.append(dict(subject=r['subject'],end_s=r['end_s'],window_s=r['requested_window_s'],method=r['method'],**res))
    with (out/'windows.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=records[0]);w.writeheader();w.writerows(records)
    groups=[]
    for subject in sorted({r['subject'] for r in records}):
      for duration in ('5','10','15'):
       for method in ('POS','CHROM'):
        group=[r for r in records if r['subject']==subject and r['window_s']==duration and r['method']==method]
        if not group:continue
        avg=lambda key:float(np.mean([v[key] for v in group]))
        groups.append(dict(subject=subject,window_s=duration,method=method,n=len(group),original=avg('original'),inverted=avg('inverted'),lag_best_r=avg('lag_best_r'),inverted_lag_best_r=avg('inverted_lag_best_r'),median_lag_no_flip=float(np.median([v['lag_best'] for v in group])),median_lag_flip=float(np.median([v['inverted_lag_best'] for v in group])),edge_no_flip=sum(abs(v['lag_best'])>=v['lag_limit_actual']-1e-9 for v in group),edge_flip=sum(abs(v['inverted_lag_best'])>=v['lag_limit_actual']-1e-9 for v in group)))
    with (out/'summary.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=groups[0]);w.writeheader();w.writerows(groups)
    info=dict(status='complete',source=str(src),held_out_opened=False,lag_bound_seconds=LAG_MAX_SECONDS,lag_step='one frame sample',definition='positive lag compares PPG[i] with iPPG[i+k] on overlap; iPPG appears later',warning='Best shift selected using same reference and per window; optimistic, not physically verified or deployable. Signed and inverted treated separately. No change to signal algorithm.',overlap='only overlapping samples; no extrapolation',results=len(records))
    (out/'manifest.json').write_text(json.dumps(info,indent=2,ensure_ascii=False),encoding='utf-8')
    lines=['# ตรวจความไวต่อการกลับขั้วและเวลา','','กลุ่มพัฒนา 3 คน ใช้คลื่นชุดเดิม; เลื่อนเป็นขั้นเท่าหนึ่งตัวอย่าง ภายใน ±0.3 วินาที เทียบเฉพาะส่วนที่ซ้อนกัน','', '| คน | ช่วง | วิธี | n | เดิม | กลับขั้ว | เลื่อนอย่างเดียว (ดีที่สุด) | กลับขั้ว + เลื่อน (ดีที่สุด) |','|---|---:|---|---:|---:|---:|---:|---:|']
    for r in groups:lines.append(f"| {r['subject']} | {r['window_s']} s | {r['method']} | {r['n']} | {r['original']:.3f} | {r['inverted']:.3f} | {r['lag_best_r']:.3f} | {r['inverted_lag_best_r']:.3f} |")
    lines += ['', 'ค่าเป็น correlation เฉลี่ยรายคน: กลับขั้วอย่างเดียวเปลี่ยนเครื่องหมายพอดี; การหาจุดเลื่อนที่ดีที่สุดใช้ PPG ของช่วงเดียวกันจึงมี bias ให้คะแนนดูดีขึ้น ไม่ใช่การวัดความแม่นยำจริงหรือหลักฐานว่ารู้ delay ที่แท้จริง', '', 'ค่าเฉลี่ย delay และจำนวนที่ติดขอบเขตอยู่ใน summary.csv; lag บวกหมายถึงเปรียบเทียบ PPG[i] กับ iPPG[i+k] ซึ่ง iPPG อยู่ภายหลังตามเวลาในไฟล์ ไม่ยืนยันสาเหตุทางสรีรวิทยา', '', 'ขั้วและการเลื่อนอาจไม่คงที่ รายงานแบบไม่ปรับเสมอ เวลาจริงยังไม่ยืนยัน ตัวกรองและช่วงที่ซ้อนกันมีผลต่อการตีความ; กลุ่มทดสอบที่กันไว้ยังไม่เปิด']
    (out/'RESULT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print('Saved:',out)
if __name__=='__main__':main()
