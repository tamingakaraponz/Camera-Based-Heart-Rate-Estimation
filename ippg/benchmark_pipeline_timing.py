"""Measure the existing unpaced POS replay, without changing its calculations."""
import json
import platform
import time
from collections import defaultdict, Counter
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import ippg_v2_refactored as core
import replay_realtime as replay

DATA = Path(r'C:\Users\Akkarapon\Downloads\Project scius\dataset for ipph')
OUT = Path(__file__).resolve().parents[1] / 'outputs' / ('pipeline_timing_' + datetime.now().strftime('%Y%m%d_%H%M%S'))
records = defaultdict(list)

def wrap(obj, name, label):
    original = getattr(obj, name)
    def timed(*args, **kwargs):
        start = time.perf_counter()
        try:
            return original(*args, **kwargs)
        finally:
            records[label].append((time.perf_counter() - start) * 1000)
    setattr(obj, name, timed)

def install_instrumentation():
    for name in ('resample_to_fps', 'calculate_pos', 'wavelet_denoise', 'bandpass_filter', 'calculate_spectrum', 'estimate_heart_rate', 'mean_skin_rgb'):
        wrap(core, name, name)
    wrap(core.AdaptiveSkinDetector, 'calibrate', 'calibrate')
    original_tracker = core._create_region_tracker
    def tracker_factory(config):
        tracker = original_tracker(config)
        wrap(tracker, 'detect', 'face_roi')
        return tracker
    core._create_region_tracker = tracker_factory
    original_capture = cv2.VideoCapture
    class TimedCapture:
        def __init__(self, *args): self.capture = original_capture(*args)
        def read(self):
            start = time.perf_counter()
            try: return self.capture.read()
            finally: records['decode'].append((time.perf_counter() - start) * 1000)
        def __getattr__(self, name): return getattr(self.capture, name)
    cv2.VideoCapture = TimedCapture
    original_evaluate = replay.evaluate
    def timed_evaluate(*args, **kwargs):
        start = time.perf_counter()
        try:
            row, stages = original_evaluate(*args, **kwargs)
            records[f"window_{args[2]}_{row['status']}"] .append((time.perf_counter()-start)*1000)
            return row, stages
        except Exception:
            records['window_error'].append((time.perf_counter()-start)*1000)
            raise
    replay.evaluate = timed_evaluate

def summary(values):
    return {'count':len(values), 'total_ms':float(sum(values)), 'mean_ms':float(np.mean(values)) if values else None, 'p95_ms':float(np.percentile(values,95)) if values else None}

def main():
    install_instrumentation()
    OUT.mkdir(parents=True, exist_ok=False)
    results = []
    cfg = core.IPPGConfig(roi_method='legacy', pulse_method='pos', denoise='dwt', background_method='none', show_preview=False)
    for n in (3,5,8,9,10):
        records.clear()
        matches = list(DATA.glob(f'subject{n}-*/subject{n}/vid.avi'))
        if len(matches) != 1:
            results.append({'subject':n,'error':f'Expected one video, found {len(matches)}'})
            continue
        meta = {}; engine = None; counts = Counter(); frame_ms = []
        start = time.perf_counter(); first = None
        stream = replay.replay_video(matches[0], cfg, meta)
        try:
            for obs in stream:
                if first is None: first = time.perf_counter() - start
                if engine is None: engine = replay.ReplayEngine(meta['fps'],cfg)
                if obs.time_s > 22: break
                counts[obs.status.split(':')[0]] += 1
                frame_ms.append(obs.extraction_ms)
                engine.push(obs)
        except Exception as e:
            results.append({'subject':n,'error':repr(e), 'statuses':dict(counts)})
            continue
        finally: stream.close()
        elapsed = time.perf_counter()-start
        rows = engine.rows if engine else []
        result = {'subject':n,'video':str(matches[0]),'fps':meta.get('fps'),'tracker':meta.get('tracker'),
                  'video_seconds':len(frame_ms)/meta['fps'],'frames':len(frame_ms),'statuses':dict(counts),
                  'elapsed_s':elapsed,'first_frame_s':first,'throughput_fps':len(frame_ms)/elapsed,
                  'frame_extraction':summary(frame_ms),'stages':{k:summary(v) for k,v in records.items()},
                  'window_status':dict(Counter(f"{r['window_seconds']}s_{r['status']}" for r in rows)),
                  'first_estimate_s':{str(w):next((r['update_time_s'] for r in rows if r['window_seconds']==w and r['status']=='estimated'),None) for w in (5,10,15)}}
        results.append(result)
        print(json.dumps(result,ensure_ascii=False),flush=True)
    report={'scope':'Unpaced video replay, first 22 seconds; POS DWT standard FFT, no reference, display or export; one run per subject. Instrumentation included.', 'python':platform.python_version(),'platform':platform.platform(),'opencv':cv2.__version__,'results':results}
    (OUT/'timings.json').write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding='utf-8')
    print('OUTPUT',OUT,flush=True)

if __name__ == '__main__': main()
