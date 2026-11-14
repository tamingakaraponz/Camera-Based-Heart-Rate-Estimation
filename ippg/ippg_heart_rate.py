r"""
iPPG heart-rate estimation from a face video.

Run from the project directory with the existing virtual environment:
    & "..\.venv\Scripts\python.exe" "ippg\ippg_heart_rate.py"
"""

from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d
from scipy.signal import butter, filtfilt


# แก้ path ตรงนี้หากย้ายไฟล์วิดีโอ
VIDEO_FILE = Path(r"C:\Users\Akkarapon\Downloads\Project scius\SCiUS-Project\DATA\temp\52.avi")

# ช่วงความถี่ที่ใช้ค้นหาชีพจร (Hz)
LOWCUT_HZ = 0.6       # 36 BPM
HIGHCUT_HZ = 2.0      # 120 BPM
MAX_DURATION_S = 34.333
SHOW_VIDEO = True


def bandpass_filter(signal, fs, lowcut=LOWCUT_HZ, highcut=HIGHCUT_HZ, order=4):
    """กรองสัญญาณให้อยู่ในช่วงความถี่ชีพจร"""
    signal = np.asarray(signal, dtype=float)
    nyquist = 0.5 * fs

    if fs <= 0:
        raise ValueError("FPS ต้องมากกว่า 0")
    if not 0 < lowcut < highcut < nyquist:
        raise ValueError(
            f"ช่วง filter ไม่ถูกต้อง: ต้องมี 0 < {lowcut} < {highcut} < {nyquist:.2f} Hz"
        )

    b, a = butter(order, [lowcut / nyquist, highcut / nyquist], btype="band")
    min_samples = 3 * (max(len(a), len(b)) - 1) + 1
    if len(signal) < min_samples:
        raise ValueError(
            f"ข้อมูลสั้นเกินไปสำหรับ filter: มี {len(signal)} จุด แต่ต้องมีอย่างน้อย {min_samples} จุด"
        )

    return filtfilt(b, a, signal)


def make_uniform_signal(times, values, fs):
    """แปลงข้อมูลที่อาจขาดบางเฟรมให้มีช่วงเวลาสม่ำเสมอก่อนทำ FFT"""
    times = np.asarray(times, dtype=float)
    values = np.asarray(values, dtype=float)

    if len(times) < 2:
        raise ValueError("มีข้อมูลน้อยเกินไป")

    # interp1d ต้องการค่าเวลาไม่ซ้ำกัน
    times, unique_indices = np.unique(times, return_index=True)
    values = values[unique_indices]
    if len(times) < 2:
        raise ValueError("ค่าเวลาที่อ่านได้ไม่เพียงพอ")

    uniform_times = np.arange(times[0], times[-1], 1.0 / fs)
    interpolate = interp1d(
        times,
        values,
        kind="linear",
        bounds_error=False,
        fill_value="extrapolate",
    )
    return uniform_times, interpolate(uniform_times)


def estimate_heart_rate(times, face_signal, background_signal, fs):
    """กรองสัญญาณ ลบสัญญาณรบกวนฉากหลัง และหา peak ใน frequency domain"""
    uniform_times, uniform_face = make_uniform_signal(times, face_signal, fs)
    _, uniform_background = make_uniform_signal(times, background_signal, fs)

    filtered_face = bandpass_filter(uniform_face, fs)
    filtered_background = bandpass_filter(uniform_background, fs)

    # คงแนวคิดเดิมของโค้ด: ลดสัญญาณรบกวนจากแสงฉากหลัง
    effective_signal = filtered_face - 1.5 * filtered_background
    effective_signal -= np.mean(effective_signal)

    # ใช้ window ลด spectral leakage
    windowed_signal = effective_signal * np.hanning(len(effective_signal))
    fft_output = np.fft.rfft(windowed_signal)
    frequencies = np.fft.rfftfreq(len(windowed_signal), d=1.0 / fs)
    magnitudes = np.abs(fft_output)

    valid = (frequencies >= LOWCUT_HZ) & (frequencies <= HIGHCUT_HZ)
    if not np.any(valid):
        raise ValueError("ไม่พบช่วงความถี่สำหรับคำนวณ Heart Rate")

    valid_indices = np.flatnonzero(valid)
    peak_index = valid_indices[np.argmax(magnitudes[valid])]
    heart_rate_bpm = frequencies[peak_index] * 60.0

    return {
        "heart_rate_bpm": heart_rate_bpm,
        "times": uniform_times,
        "raw_face": uniform_face,
        "filtered_face": filtered_face,
        "effective_signal": effective_signal,
        "frequencies": frequencies,
        "magnitudes": magnitudes,
    }


def choose_background_region(frame, face_box):
    """เลือกบริเวณฉากหลังที่ไม่ทับใบหน้า และป้องกัน ROI ว่าง"""
    frame_height, frame_width = frame.shape[:2]
    x, _, w, _ = face_box
    gap = 10

    left_end = max(0, x - gap)
    right_start = min(frame_width, x + w + gap)

    left_width = left_end
    right_width = frame_width - right_start

    if left_width >= right_width and left_width >= 10:
        return frame[:, :left_end]
    if right_width >= 10:
        return frame[:, right_start:]

    # fallback: มุมซ้ายบนขนาดเล็ก เมื่อใบหน้ากินพื้นที่เกือบทั้งภาพ
    roi_h = max(1, frame_height // 5)
    roi_w = max(1, frame_width // 5)
    return frame[:roi_h, :roi_w]


def extract_signals(video_file, max_duration=MAX_DURATION_S, show_video=SHOW_VIDEO):
    video_file = Path(video_file)
    if not video_file.is_file():
        raise FileNotFoundError(f"ไม่พบไฟล์วิดีโอ:\n{video_file}")

    cascade_file = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
    if not cascade_file.is_file():
        cascade_file = Path(__file__).resolve().parent / "assets" / cascade_file.name

    face_cascade = cv2.CascadeClassifier(str(cascade_file))
    if face_cascade.empty():
        raise RuntimeError("OpenCV โหลด Haar Cascade ไม่สำเร็จ")

    cap = cv2.VideoCapture(str(video_file))
    if not cap.isOpened():
        raise RuntimeError(f"เปิดวิดีโอไม่ได้: {video_file}")

    fps = float(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if not np.isfinite(fps) or fps <= 0:
        cap.release()
        raise RuntimeError("อ่าน FPS จากวิดีโอไม่ได้")

    duration = total_frames / fps if total_frames > 0 else 0.0
    times = []
    face_values = []
    background_values = []
    last_face = None
    frame_number = 0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            frame_number += 1
            current_time = frame_number / fps
            if max_duration is not None and current_time > max_duration:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # ตรวจจับใหม่ทุก 10 เฟรม; เฟรมอื่นใช้ตำแหน่งล่าสุด ลดอาการข้อมูลขาด
            if last_face is None or frame_number % 10 == 0:
                faces = face_cascade.detectMultiScale(
                    gray,
                    scaleFactor=1.2,
                    minNeighbors=6,
                    minSize=(80, 80),
                )
                if len(faces) > 0:
                    last_face = max(faces, key=lambda box: box[2] * box[3])

            if last_face is None:
                continue

            x, y, w, h = map(int, last_face)
            frame_h, frame_w = frame.shape[:2]

            # ROI หน้าผาก: กึ่งกลางด้านบนของกรอบใบหน้า
            x1 = max(0, x + int(0.30 * w))
            x2 = min(frame_w, x + int(0.70 * w))
            y1 = max(0, y + int(0.08 * h))
            y2 = min(frame_h, y + int(0.25 * h))
            forehead = frame[y1:y2, x1:x2]

            if forehead.size == 0:
                last_face = None
                continue

            background = choose_background_region(frame, (x, y, w, h))
            if background.size == 0:
                continue

            # OpenCV เรียง channel เป็น BGR: index 1 = Green, index 2 = Red
            face_value = float(np.mean(forehead[:, :, 1]) - np.mean(forehead[:, :, 2]))
            background_value = float(
                np.mean(background[:, :, 1]) - np.mean(background[:, :, 2])
            )

            times.append(current_time)
            face_values.append(face_value)
            background_values.append(background_value)

            if show_video:
                display = frame.copy()
                cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 255), 2)
                cv2.putText(
                    display,
                    "Forehead ROI | Q: quit",
                    (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 255),
                    2,
                )
                cv2.imshow("iPPG", display)
                if cv2.waitKey(1) & 0xFF in (ord("q"), ord("Q")):
                    break
    finally:
        cap.release()
        cv2.destroyAllWindows()

    if not times:
        raise RuntimeError(
            "ไม่พบใบหน้าในวิดีโอ ลองใช้วิดีโอที่ใบหน้าตรง แสงสว่าง และไม่ขยับมาก"
        )

    print(f"Video FPS: {fps:.2f}")
    print(f"Video duration: {duration:.2f} s")
    print(f"Samples used: {len(times)}")
    return np.asarray(times), np.asarray(face_values), np.asarray(background_values), fps


def plot_results(result):
    times = result["times"]

    fig, axes = plt.subplots(3, 1, figsize=(11, 12))

    axes[0].plot(times, result["raw_face"], color="green", label="Raw G-R signal")
    axes[0].set_ylabel("Amplitude (a.u.)")
    axes[0].set_title("Raw forehead signal")
    axes[0].grid(True)
    axes[0].legend()

    axes[1].plot(times, result["filtered_face"], color="blue", label="Band-pass filtered")
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("Amplitude (a.u.)")
    axes[1].set_title("Filtered forehead signal")
    axes[1].grid(True)
    axes[1].legend()

    bpm_axis = result["frequencies"] * 60.0
    valid = (bpm_axis >= LOWCUT_HZ * 60) & (bpm_axis <= HIGHCUT_HZ * 60)
    axes[2].plot(bpm_axis[valid], result["magnitudes"][valid], color="red")
    axes[2].axvline(result["heart_rate_bpm"], color="black", linestyle="--")
    axes[2].set_xlabel("Heart rate (BPM)")
    axes[2].set_ylabel("Magnitude (a.u.)")
    axes[2].set_title(f"FFT peak = {result['heart_rate_bpm']:.1f} BPM")
    axes[2].grid(True)

    plt.tight_layout()
    plt.show()


def main():
    try:
        times, face_values, background_values, fps = extract_signals(VIDEO_FILE)
        result = estimate_heart_rate(times, face_values, background_values, fps)
        print(f"Heart Rate: {result['heart_rate_bpm']:.1f} BPM")
        plot_results(result)
    except (FileNotFoundError, RuntimeError, ValueError) as error:
        print(f"ERROR: {error}")


if __name__ == "__main__":
    main()
