"""Tests for replay causality, bounded memory and failures, not live accuracy."""
import sys
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import replay_realtime as replay


def config():
    return replay.core.IPPGConfig(show_preview=False,pulse_method="pos",roi_method="legacy",denoise="dwt",background_method="none")


def rgb(t):
    return np.array([160.,120.,90.])*(1+.01*np.sin(2*np.pi*1.2*t)*np.array([.2,1.,.1]))


class ReplayTests(unittest.TestCase):
    def test_buffer_is_time_bounded_and_rejects_future_reads(self):
        b=replay.TimeBuffer(15)
        for t in np.arange(0,100,.1):
            b.append(replay.Observation(t,rgb(t)))
            self.assertLessEqual(len(b.items),151)
            self.assertTrue(all(t-15<o.time_s<=t for o in b.items))
        with self.assertRaisesRegex(ValueError,"Future"):
            b.select(99,5)
        with self.assertRaises(ValueError):
            b.append(replay.Observation(1,rgb(1)))

    def test_future_frame_not_passed_to_scheduled_tick(self):
        seen=[]
        def spy(buffer,now,seconds,*args):
            seen.append((now,buffer.last_time))
            return {"time":now},{}
        e=replay.ReplayEngine(30,config(),windows=(5,))
        with patch.object(replay,"evaluate",side_effect=spy):
            e.push(replay.Observation(0,rgb(0)))
            e.push(replay.Observation(4.5,rgb(4.5)))
        self.assertEqual([a for a,b in seen],[1,2,3,4])
        self.assertTrue(all(last<=now for now,last in seen))

    def test_future_prefix_does_not_change_prior_results(self):
        def run(change):
            e=replay.ReplayEngine(30,config(),windows=(5,))
            for t in np.arange(0,9,1/30):
                v=rgb(t) if t<7 or not change else np.array([1e5,3.,500.])
                e.push(replay.Observation(t,v))
            return [(r["update_time_s"],r["status"],r["ippg_bpm"]) for r in e.rows if r["update_time_s"]<7]
        self.assertEqual(run(False),run(True))

    def test_known_pulse_is_recovered_after_history_exists(self):
        e=replay.ReplayEngine(30,config(),windows=(5,10,15))
        for t in np.arange(0,17,1/30):
            e.push(replay.Observation(t,rgb(t)))
        for w in (5,10,15):
            ready=[r for r in e.rows if r["window_seconds"]==w and r["status"]=="estimated"]
            self.assertTrue(ready)
            self.assertAlmostEqual(ready[0]["update_time_s"],w)
            self.assertLessEqual(abs(ready[-1]["ippg_bpm"]-72),60/w)
        self.assertLessEqual(e.max_buffer_samples,451)

    def test_missing_interval_and_expired_signal_are_not_published(self):
        e=replay.ReplayEngine(30,config(),windows=(5,))
        for t in np.arange(0,14,1/30):
            e.push(replay.Observation(t,rgb(t) if t<6 else None,"valid" if t<6 else "face_missing"))
        late=[r for r in e.rows if r["update_time_s"]>=7]
        self.assertTrue(late)
        self.assertTrue(all(r["status"]=="unavailable" and r["ippg_bpm"] is None for r in late))

    def test_flat_rgb_is_rejected(self):
        e=replay.ReplayEngine(30,config(),windows=(5,))
        for t in np.arange(0,7,1/30):
            e.push(replay.Observation(t,np.array([160.,120.,90.])))
        self.assertTrue(all(r["status"]!="estimated" for r in e.rows))

    def test_reference_rejects_future_only_support(self):
        t=np.arange(0,10,.1)
        ref=(t,np.sin(2*np.pi*1.2*t))
        with self.assertRaisesRegex(ValueError,"support"):
            replay.reference_window(ref,5,np.arange(1,6,.1),30,config(),.25)
        grid=np.arange(1,4.9,1/30)
        a=replay.reference_window(ref,5,grid,30,config(),.25)
        changed=ref[1].copy()
        changed[t>5]=1e6
        b=replay.reference_window((t,changed),5,grid,30,config(),.25)
        self.assertEqual(a[0],b[0])
        np.testing.assert_array_equal(a[2],b[2])

    def test_shared_summary_uses_identical_paired_times(self):
        rows=[]
        for w in (5,10,15):
            for tick in (16,17,18):
                rows.append(dict(window_seconds=w,update_time_s=tick,status="estimated",ippg_bpm=72.,
                    difference_bpm=None if (w==10 and tick==17) else float(w),compute_ms=2.))
        result=replay.summarize(rows,(5,10,15))
        self.assertTrue(all(r["shared_paired_ticks"]==2 for r in result))
        self.assertEqual([r["shared_MAE_bpm"] for r in result],[5,10,15])


if __name__=="__main__":
    unittest.main()
