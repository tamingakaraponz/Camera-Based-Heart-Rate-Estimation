import sys
from pathlib import Path
import unittest
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from compare_saved_signals import agreement, estimate_lag, fft_hr


class ComparisonTests(unittest.TestCase):
    def test_scale_offset_and_polarity(self):
        x=np.array([0.,2.,1.,-3.,5.])
        self.assertAlmostEqual(agreement(x,7*x+21)["pearson_r"],1.)
        self.assertAlmostEqual(agreement(x,-x)["pearson_r"],-1.)
        self.assertAlmostEqual(agreement(x,-x)["normalized_rmse"],2.)
        with self.assertRaises(ValueError):
            agreement(x,np.ones_like(x))

    def test_lag_sign_recovers_known_delay(self):
        fs=50.
        t=np.arange(0,30,1/fs)
        def pulse(t):
            return np.sin(2*np.pi*(.8*t+.025*t*t))
        x,y=pulse(t),pulse(t-.2)
        grid,a,lag,before,after,offsets,scores=estimate_lag(t,x,t,y,fs,1.)
        self.assertAlmostEqual(lag,.2)
        self.assertGreater(after["pearson_r"],.99999)
        self.assertLess(before["pearson_r"],.5)
        self.assertGreaterEqual(grid[0]+min(offsets),t[0])
        self.assertLessEqual(grid[-1]+max(offsets),t[-1])

    def test_known_frequency(self):
        fs=30.
        t=np.arange(600)/fs
        self.assertAlmostEqual(fft_hr(np.sin(2*np.pi*1.5*t),fs),90.)


if __name__ == "__main__":
    unittest.main()
