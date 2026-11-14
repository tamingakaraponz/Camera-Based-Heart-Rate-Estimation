"""Lossless signal archives plus readable CSV exports for later comparisons.

This module saves results only: it never aligns, scales, filters, or fits them.
"""
from dataclasses import asdict
from datetime import datetime, timedelta, timezone
from hashlib import file_digest, sha256
import csv
import importlib.metadata
import json
from pathlib import Path
import re
import uuid

import numpy as np

DEFAULT_OUTPUT_ROOT = Path(__file__).resolve().parents[1] / "outputs" / "signal_runs"


def _json_default(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def _write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=_json_default, allow_nan=False), encoding="utf-8")


def _source_info(path):
    path = Path(path).expanduser().resolve()
    stat = path.stat()
    with path.open("rb") as stream:
        fingerprint = file_digest(stream, "sha256").hexdigest()
    return {"path":str(path),"sha256":fingerprint,"size_bytes":stat.st_size,"mtime_ns":stat.st_mtime_ns}


def _code_info(names):
    base = Path(__file__).resolve().parent
    return {name:sha256((base/name).read_bytes()).hexdigest() for name in names}


def _versions():
    versions={}
    for name in ("numpy","scipy","matplotlib","opencv-python","opencv-contrib-python","mediapipe","PyWavelets"):
        try:
            versions[name]=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name]=None
    return versions


def _write_table(path, columns):
    arrays = [np.asarray(v) for v in columns.values()]
    if not arrays or any(a.ndim != 1 for a in arrays) or len({len(a) for a in arrays}) != 1:
        raise ValueError(f"CSV columns must be matching 1-D arrays: {path.name}")
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer=csv.writer(stream)
        writer.writerow(columns)
        writer.writerows(zip(*arrays))


def save_bundle(kind, source, arrays, tables, metadata, figure, output_root=DEFAULT_OUTPUT_ROOT):
    """Create one never-overwritten run directory, finalized by manifest.json."""
    source_info = _source_info(source)
    clean_arrays = {}
    for name, value in arrays.items():
        if value is None:
            continue
        a = np.asarray(value)
        if a.dtype.kind not in "fiu" or a.size == 0 or not np.all(np.isfinite(a)):
            raise ValueError(f"Cannot archive invalid numeric array: {name}")
        clean_arrays[name] = a
    now=datetime.now(timezone(timedelta(hours=7)))
    label=re.sub(r"[^A-Za-z0-9_-]+", "_", Path(source).parent.name).strip("_") or "recording"
    run_id=f"{label}_{kind}_{now.strftime('%Y%m%d_%H%M%S_%f')}_{uuid.uuid4().hex[:6]}"
    folder=Path(output_root).expanduser().resolve()/run_id
    folder.mkdir(parents=True,exist_ok=False)
    manifest={"schema_version":1,"status":"writing","kind":kind,"recording_label":Path(source).parent.name,
              "created_at":now.isoformat(),"timezone":"Asia/Bangkok","source":source_info,
              "signal_processing_changed_by_export":False,"alignment_applied":False,
              "polarity_flip_applied":False,"hr_multiplier":1.0,"synchronization":"not verified by this exporter",
              "metadata":metadata,"arrays":{k:{"shape":list(v.shape),"dtype":str(v.dtype)} for k,v in clean_arrays.items()}}
    _write_json(folder/"manifest.json",manifest)
    try:
        np.savez_compressed(folder/"signals.npz",**clean_arrays)
        for filename,columns in tables.items():
            if Path(filename).name!=filename or not filename.endswith(".csv"):
                raise ValueError("Invalid export table name")
            _write_table(folder/filename,columns)
        figure.savefig(folder/"graph.png",dpi=150)
        readme=(
            f"# ผลการรัน {kind}\n\n"
            f"ไฟล์ต้นทาง: {source_info['path']}\n\n"
            f"Heart Rate ที่คำนวณ: **{metadata['heart_rate_bpm']:.4f} BPM**\n\n"
            "- graph.png: กราฟจากการรันนี้\n"
            "- signals.npz: ค่าสัญญาณเต็มความละเอียดสำหรับโหลดกลับด้วย NumPy\n"
            "- *.csv: ตัวเลขรายจุด แยกตารางตามแกนเวลา เปิดใน Excel ได้\n"
            "- manifest.json: ที่มา เวลา การตั้งค่า รุ่นโค้ด และ hash ของไฟล์ผลลัพธ์\n\n"
            "กราฟและ CSV ไม่ได้เลื่อนเวลา กลับขั้ว ปรับตัวคูณ BPM หรือ normalize เพื่อให้สองแหล่งตรงกัน\n"
            "ค่าก่อนกรองของ iPPG คือสัญญาณหลัง POS และ RGB ที่จัดเวลาแล้ว ไม่ใช่สัญญาณเซ็นเซอร์ดิบ\n"
            "PPG ต้นฉบับเก็บแถวข้อมูลเดิมและเวลาเดิมไว้แยกจากค่าที่ interpolate และกรองแล้ว\n"
            "หน่วย amplitude เป็น a.u.; ความสูงของ iPPG/PPG คนละแหล่งไม่จำเป็นต้องเท่ากัน\n"
            "การมีเวลาเป็นตัวเลขไม่ยืนยันว่าบันทึกพร้อมกัน ต้องตรวจ time basis ก่อนเปรียบเทียบ\n"
        )
        (folder/"RESULT.md").write_text(readme,encoding="utf-8")
        stat=Path(source_info["path"]).stat()
        manifest["source_stat_unchanged_during_export"]=(stat.st_size==source_info["size_bytes"] and stat.st_mtime_ns==source_info["mtime_ns"])
        if not manifest["source_stat_unchanged_during_export"]:
            raise RuntimeError("Source changed during export")
        manifest["files"]={p.name:{"size_bytes":p.stat().st_size,"sha256":sha256(p.read_bytes()).hexdigest()} for p in folder.iterdir() if p.name!="manifest.json"}
        manifest["status"]="complete"
        _write_json(folder/"manifest.json",manifest)
    except Exception as error:
        manifest["status"]="failed"
        manifest["error"]=str(error)
        _write_json(folder/"manifest.json",manifest)
        raise
    return folder


def save_ippg_run(result, config, source, figure, output_root=DEFAULT_OUTPUT_ROOT):
    s=result.signals
    if s.pulse_signal is None or s.pulse_method != "pos":
        raise ValueError("This exporter currently expects a POS single-pulse result, not G-R or fusion.")
    frequency,fft=np.asarray(result.frequencies_hz),np.asarray(result.compensated_spectrum)
    if result.background_method!="none":
        raise ValueError("This exporter expects background_method='none' for unambiguous FFT power.")
    arrays={"time_seconds":s.time_seconds,"rgb":s.combined_rgb,"forehead_rgb":s.forehead_rgb,"cheeks_rgb":s.cheeks_rgb,
            "raw_signal":s.pulse_signal,"reconstructed_signal":result.reconstructed_pulse_signal,
            "filtered_signal":result.filtered_pulse_signal,"frequencies_hz":frequency,
            "fft_power":fft,"roi_quality":s.roi_quality,"roi_weights":s.roi_weights}
    tables={"signal.csv":{"video_time_s":s.time_seconds,"R":s.combined_rgb[:,0],"G":s.combined_rgb[:,1],"B":s.combined_rgb[:,2],
                          "raw_pos":s.pulse_signal,"after_dwt":result.reconstructed_pulse_signal,"filtered_pos":result.filtered_pulse_signal},
            "spectrum.csv":{"frequency_hz":frequency,"frequency_bpm":frequency*60,"fft_power":fft}}
    model=config.face_landmarker_model_path or Path(__file__).resolve().parents[1]/"models/face_landmarker.task"
    model_info=_source_info(model) if s.face_tracker_name and "MediaPipe" in s.face_tracker_name and Path(model).is_file() else None
    metadata={"heart_rate_bpm":float(result.heart_rate_bpm),"sampling_rate_hz":float(s.fps),
              "face_tracker_actual":s.face_tracker_name,"face_landmarker_model":model_info,
              "roi_quality_meaning":"Legacy ROI uses placeholder ones; not a confidence score" if config.roi_method=="legacy" else "Patch selection score, not calibrated confidence",
              "time_basis":"video frame number / metadata FPS; calibration frames excluded; RGB interpolated",
              "analysis_start_s":float(s.time_seconds[0]),"analysis_end_s":float(s.time_seconds[-1]),
              "config":asdict(config),"denoise":result.denoise_metadata,
              "stage_definitions":{"raw_signal":"POS before DWT and band-pass","reconstructed_signal":"after optional DWT, before band-pass","filtered_signal":"after band-pass"},
              "filter_implementation":"Butterworth b/a, filtfilt","fft_window":"rectangular; mean-centered; core may rebuild FFT grid",
              "core_source_sha256":_code_info(["ippg_v2_refactored.py","signal_recording.py"]),
              "versions":_versions()}
    return save_bundle("ippg",source,arrays,tables,metadata,figure,output_root)


def save_reference_run(result, source, figure, order, requested_fs=None, output_root=DEFAULT_OUTPUT_ROOT):
    source_time=result.source_time if result.source_time is not None else result.original_time
    arrays={"source_time_seconds":source_time,"original_time_seconds":result.original_time,"original_signal":result.original_ppg,
            "provided_hr_bpm":result.provided_hr,"time_seconds":result.time,"raw_signal":result.uniform_ppg,
            "filtered_signal":result.filtered_ppg,"frequencies_hz":result.frequency,"fft_magnitude":result.magnitude,"fft_power":result.magnitude**2}
    tables={"original_signal.csv":{"source_time_s":source_time,"relative_time_s":result.original_time,"original_ppg":result.original_ppg,"provided_hr_bpm":result.provided_hr},
            "signal.csv":{"relative_time_s":result.time,"interpolated_ppg":result.uniform_ppg,"filtered_ppg":result.filtered_ppg},
            "spectrum.csv":{"frequency_hz":result.frequency,"frequency_bpm":result.frequency*60,"fft_magnitude":result.magnitude,"fft_power":result.magnitude**2}}
    metadata={"heart_rate_bpm":float(result.heart_rate_bpm),"sampling_rate_hz":float(result.fps),
              "time_basis":"stored timestamp minus first stored timestamp; duplicate times averaged; linear interpolation",
              "source_time_origin_s":float(source_time[0]),"duplicate_timestamps_combined":result.duplicate_count,
              "largest_timestamp_gap_s":result.largest_gap_s,"analysis_start_s":float(result.time[0]),"analysis_end_s":float(result.time[-1]),
              "low_hz":result.low_hz,"high_hz":result.high_hz,"filter_order":order,"requested_fs":requested_fs,
              "filter_implementation":"Butterworth SOS, sosfiltfilt; input mean removed; explicit padlen",
              "fft_window":"rectangular; mean-centered; full filtered signal",
              "provided_hr_mean_bpm":float(np.mean(result.provided_hr)),"provided_hr_used_in_fft":False,
              "core_source_sha256":_code_info(["analyze_ground_truth.py","signal_recording.py"]),
              "versions":_versions()}
    return save_bundle("reference_ppg",source,arrays,tables,metadata,figure,output_root)
