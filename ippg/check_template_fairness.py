"""Common-band periodic-template comparison with explicit phase fitting."""
from pathlib import Path
import csv,json
import numpy as np
from test_template_candidates import bank,waves,subjects,bpm_fft
OUT=Path(__file__).resolve().parents[1]/'outputs/template_fairness';OUT.mkdir(parents=True,exist_ok=True)

def coefficients(shape,rate):
    # Treat each stored shape as an observed output-domain periodic shape.
    # Apply identical ideal support projection, never another Butterworth pass.
    c=np.fft.rfft(shape)/len(shape)
    harmonic=np.arange(len(c))
    keep=(harmonic*rate/60>=.6)&(harmonic*rate/60<=2.)
    indices=np.flatnonzero(keep)
    if not len(indices) or np.sum(np.abs(c[indices])**2)<1e-16:
        raise ValueError('no in-band template energy')
    return indices,c[indices]

def render(shape,rate,t,shift=0):
    h,c=coefficients(shape,rate)
    return 2*np.real(np.exp(2j*np.pi*np.outer(t*rate/60+shift,h))@c)

def score(shape,rate,t,x):
    h,c=coefficients(shape,rate);x=x-x.mean();energy=np.dot(x,x)
    if energy<1e-16:raise ValueError('flat input')
    if len(h)==1:
        # Continuous phase and signed amplitude fit for any single harmonic.
        design=np.column_stack((np.sin(2*np.pi*h[0]*rate/60*t),np.cos(2*np.pi*h[0]*rate/60*t),np.ones(len(t))))
        y=design@np.linalg.lstsq(design,x,rcond=None)[0]
        return float(1-np.sum((x-y)**2)/energy)
    shifts=np.arange(128)/128
    y=2*np.real(np.exp(2j*np.pi*np.outer(t*rate/60,h))@(c[:,None]*np.exp(2j*np.pi*np.outer(h,shifts))))
    y-=y.mean(axis=0);den=np.sum(y*y,axis=0);numerator=x@y
    return float(np.max(numerator*numerator/(energy*den)))

def estimate(shape,t,x):
    values=[score(shape,b,t,x) for b in range(36,121)]
    best=int(np.argmax(values));return 36+best,values[best]

def save(name,rows):
    with (OUT/name).open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)

def main():
    templates,_=bank('subject3');rng=np.random.default_rng(42);checks=[]
    for duration in (5,10,15):
        t=np.arange(0,duration,1/30)
        for name,shape in templates.items():
            for true in (48,80,110):
                clean=render(shape,true,t,.173)
                for case in ('clean','inverted','noise'):
                    x=-clean if case=='inverted' else clean.copy()
                    if case=='noise':x+=rng.normal(0,.15*np.std(clean),len(t))
                    found,r2=estimate(shape,t,x)
                    checks.append(dict(duration=duration,candidate=name,true_bpm=true,case=case,estimated=found,error=abs(found-true),r2=r2))
    # All five families must be equivalent at 80 BPM when only fundamental survives.
    t=np.arange(0,10,1/30);x=np.sin(2*np.pi*80/60*t+.45)
    equivalent=[score(shape,80,t,x) for shape in templates.values()]
    assert np.ptp(equivalent)<1e-10
    assert max(r['error'] for r in checks if r['case']!='noise')==0
    results=[]
    for held in subjects:
        templates,_=bank(held);z=waves[held];t=z['time_s']-z['time_s'][0];fs=1/np.median(np.diff(t))
        for method in ('pos','chrom'):
            for name,shape in templates.items():
                rate,r2=estimate(shape,t,z[method]);results.append(dict(subject=held,method=method,candidate=name,bpm=rate,r2=r2,ppg_fft=bpm_fft(z['ppg'],fs)))
    save('synthetic.csv',checks);save('real_windows.csv',results)
    report=dict(synthetic_cases=len(checks),clean_worst=max(r['error'] for r in checks if r['case']=='clean'),inverted_worst=max(r['error'] for r in checks if r['case']=='inverted'),noise_worst=max(r['error'] for r in checks if r['case']=='noise'),equivalence_spread=float(np.ptp(equivalent)),real_cases=len(results),protocol='output-domain templates; ideal 0.6–2Hz harmonic support; same rate/time; continuous single-harmonic phase, 128 phase trials otherwise',limits='Ideal projection differs from actual Butterworth rolloff; empirical source already filtered, morphology and rescaling remain approximate; synthetic inputs generated from candidate model; no independent physiology validation')
    (OUT/'checks.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps(report,indent=2),flush=True)
if __name__=='__main__':main()
