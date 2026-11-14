r"""Set VIDEO_FILE below and press Run to estimate heart rate from face video.

Run from the project directory:
    & "..\.venv\Scripts\python.exe" "ippg\run_ippg.py"

Keep this file beside ippg_v2_refactored.py in the ippg folder.
"""
from pathlib import Path
import subprocess
import sys
import argparse

# Code Runner may invoke a system Python without this project's dependencies.
# Switch before importing OpenCV/NumPy through the processing module.
PROJECT_PYTHON = Path(__file__).resolve().parents[2] / ".venv" / "Scripts" / "python.exe"
if (
    __name__ == "__main__"
    and PROJECT_PYTHON.is_file()
    and str(Path(sys.executable).resolve()).casefold()
    != str(PROJECT_PYTHON.resolve()).casefold()
):
    print(f"Using project Python: {PROJECT_PYTHON}", flush=True)
    raise SystemExit(subprocess.run([str(PROJECT_PYTHON), *sys.argv]).returncode)

from ippg_v2_refactored import IPPGConfig, analyze_video, plot_result
from signal_recording import DEFAULT_OUTPUT_ROOT, save_ippg_run


# ใส่ที่อยู่ไฟล์วิดีโอตรงนี้
VIDEO_FILE = Path(
    r"C:\Users\Akkarapon\Downloads\subject1-20261104T091024Z-1-001\subject1\vid.avi"
)

# None = วิเคราะห์ทั้งคลิป; ใส่ 34.333 หากต้องการจำกัดช่วงแรก
MAX_DURATION_SECONDS = None

# True = เปิดภาพแสดงบริเวณผิวระหว่างประมวลผล; False = แสดงเฉพาะกราฟตอนจบ
SHOW_PREVIEW = False

# บันทึกผลทุกครั้งในโฟลเดอร์ใหม่ ไม่ทับผลรอบก่อน
OUTPUT_ROOT = DEFAULT_OUTPUT_ROOT


def main():
    parser = argparse.ArgumentParser(description="Run iPPG and automatically save graphs and signal values.")
    parser.add_argument("video", nargs="?", type=Path, default=VIDEO_FILE)
    parser.add_argument("--duration", type=float, default=MAX_DURATION_SECONDS)
    parser.add_argument("--preview", action="store_true", default=SHOW_PREVIEW)
    parser.add_argument("--no-show", action="store_true", help="Save results without opening graph window")
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    args = parser.parse_args()
    if args.no_show:
        import matplotlib
        matplotlib.use("Agg")
    config = IPPGConfig(
        max_duration_seconds=args.duration,
        show_preview=args.preview,
        pulse_method="pos",
        roi_method="legacy",
        denoise="dwt",
        background_method="none",
    )

    try:
        # วิดีโอ -> ใบหน้า/ผิว -> RGB -> POS -> DWT -> band-pass -> FFT -> BPM
        result = analyze_video(args.video.expanduser(), config)
    except (OSError, RuntimeError, ValueError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1

    print(f"Video: {args.video}")
    print(f"Video FPS: {result.signals.fps:.4f}")
    print(
        f"Analysis interval: {result.signals.time_seconds[0]:.2f}"
        f" - {result.signals.time_seconds[-1]:.2f} s"
    )
    print(f"Heart Rate: {result.heart_rate_bpm:.2f} BPM")
    import matplotlib.pyplot as plt
    figure = plot_result(result, show=False)
    try:
        folder = save_ippg_run(result, config, args.video, figure, args.output_root)
        print(f"Saved results: {folder}")
        if not args.no_show:
            plt.show()
    except (OSError, ValueError, RuntimeError) as error:
        print(f"ERROR saving results: {error}", file=sys.stderr)
        return 1
    finally:
        plt.close(figure)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
