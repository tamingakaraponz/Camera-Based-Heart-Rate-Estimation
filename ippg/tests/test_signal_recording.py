"""Round-trip and failure-state checks for signal archives."""
from pathlib import Path
from hashlib import sha256
import csv
import json
import sys
import tempfile
import unittest

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import ippg_v2_refactored as core
from analyze_ground_truth import analyze_file, plot_results
from signal_recording import save_ippg_run, save_reference_run, save_bundle


class RecordingTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.root=Path(self.tmp.name)

    def tearDown(self):
        plt.close("all")
        self.tmp.cleanup()

    def reference(self):
        t=np.arange(900)/30+1000
        ppg=5+np.sin(2*np.pi*1.2*(t-1000))
        rows=np.vstack((ppg,np.full_like(t,72),t))
        rows=np.insert(rows,100,rows[:,100],axis=1)
        path=self.root/"ground_truth.txt"
        np.savetxt(path,rows)
        return path,rows,analyze_file(path)

    def verify_manifest(self,folder):
        m=json.loads((folder/"manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(m["status"],"complete")
        self.assertFalse(m["alignment_applied"])
        self.assertFalse(m["polarity_flip_applied"])
        self.assertEqual(m["hr_multiplier"],1)
        for name,details in m["files"].items():
            self.assertEqual(sha256((folder/name).read_bytes()).hexdigest(),details["sha256"])
        return m

    def test_reference_preserves_original_time_duplicates_and_csv_values(self):
        path,rows,result=self.reference()
        before=path.read_bytes()
        fig=plot_results(result,"test")
        folder=save_reference_run(result,path,fig,3,output_root=self.root/"runs")
        m=self.verify_manifest(folder)
        self.assertEqual(m["metadata"]["duplicate_timestamps_combined"],1)
        with np.load(folder/"signals.npz") as z:
            np.testing.assert_array_equal(z["source_time_seconds"],rows[2])
            np.testing.assert_array_equal(z["original_signal"],rows[0])
            np.testing.assert_array_equal(z["filtered_signal"],result.filtered_ppg)
        with (folder/"original_signal.csv").open(encoding="utf-8-sig",newline="") as stream:
            entries=list(csv.DictReader(stream))
        self.assertEqual(len(entries),len(rows[0]))
        self.assertEqual(float(entries[0]["source_time_s"]),1000)
        self.assertEqual(float(entries[0]["relative_time_s"]),0)
        self.assertEqual(path.read_bytes(),before)

    def test_repeated_runs_do_not_overwrite(self):
        path,_,result=self.reference()
        fig=plot_results(result,"test")
        a=save_reference_run(result,path,fig,3,output_root=self.root/"runs")
        b=save_reference_run(result,path,fig,3,output_root=self.root/"runs")
        self.assertNotEqual(a,b)
        self.verify_manifest(a)
        self.verify_manifest(b)

    def test_ippg_keeps_each_stage_and_spectrum_on_its_own_grid(self):
        t=np.arange(900)/30
        pulse=np.sin(2*np.pi*1.2*t)
        rgb=np.column_stack((100+pulse,120+pulse,90+pulse))
        signals=core.ExtractedSignals(t,rgb,rgb,rgb,pulse,None,30,"pos",None)
        config=core.IPPGConfig(show_preview=False,background_method="none")
        result=core.analyze_signals(signals,config)
        fig=core.plot_result(result,show=False)
        source=self.root/"vid.avi"
        source.write_bytes(b"fixture: source fingerprint only; no video decoding in this test")
        folder=save_ippg_run(result,config,source,fig,self.root/"runs")
        m=self.verify_manifest(folder)
        self.assertEqual(m["metadata"]["config"]["forehead_filter_order"],6)
        with np.load(folder/"signals.npz") as z:
            np.testing.assert_array_equal(z["raw_signal"],pulse)
            np.testing.assert_array_equal(z["filtered_signal"],result.filtered_pulse_signal)
            np.testing.assert_array_equal(z["fft_power"],result.compensated_spectrum)
            self.assertNotEqual(len(z["time_seconds"]),len(z["frequencies_hz"]))

    def test_failed_export_is_not_marked_complete(self):
        path,_,_=self.reference()
        fig=plt.figure()
        with self.assertRaisesRegex(ValueError,"matching"):
            save_bundle("test",path,{"x":np.arange(3)},
                        {"bad.csv":{"a":np.arange(3),"b":np.arange(2)}},
                        {"heart_rate_bpm":72},fig,self.root/"runs")
        manifests=list((self.root/"runs").glob("*/manifest.json"))
        self.assertEqual(len(manifests),1)
        self.assertEqual(json.loads(manifests[0].read_text(encoding="utf-8"))["status"],"failed")


if __name__=="__main__":
    unittest.main()
