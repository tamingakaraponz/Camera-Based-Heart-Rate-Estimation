"""Read-only inventory and sampled integrity checks; no HR estimation."""
import argparse
import csv
from datetime import datetime
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

DEFAULT_ROOT=Path(r"C:\Users\Akkarapon\Downloads\Project scius\dataset for ipph")


def inspect(root):
    records=[]
    folders=sorted({p.parent for name in ("vid.avi","ground_truth.txt") for p in root.rglob(name)})
    for folder in folders:
        video,reference=folder/"vid.avi",folder/"ground_truth.txt"
        row=dict(subject=folder.name,folder=str(folder),video=str(video),reference=str(reference),
                 pair_present=video.is_file() and reference.is_file())
        for path,label in ((video,"video"),(reference,"reference")):
            if path.is_file():
                row[label+"_bytes"]=path.stat().st_size
                row[label+"_mtime_ns"]=path.stat().st_mtime_ns
        if row["pair_present"]:
            cap=cv2.VideoCapture(str(video))
            try:
                fs=float(cap.get(cv2.CAP_PROP_FPS))
                n=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                ok,_=cap.read()
                cap.set(cv2.CAP_PROP_POS_FRAMES,max(0,n-1))
                end_ok,_=cap.read()
                row.update(video_fps=fs,video_frames=n,video_duration_s=n/fs if fs>0 else None,
                           first_frame_decoded=bool(ok),last_frame_decoded=bool(end_ok))
            finally:
                cap.release()
            try:
                data=np.loadtxt(reference,ndmin=2)
                row["reference_sha256"]=hashlib.sha256(reference.read_bytes()).hexdigest()
                row.update(reference_shape=list(data.shape),reference_finite=bool(np.all(np.isfinite(data))))
                if data.shape[0]==3 and data.shape[1]>=2 and row["reference_finite"]:
                    t=data[2]-data[2,0]
                    differences=np.diff(t)
                    row.update(reference_samples=len(t),reference_duration_s=float(t[-1]),
                        duplicate_timestamps=int(sum(differences==0)),backward_timestamps=int(sum(differences<0)),
                        largest_timestamp_gap_s=float(differences.max()),
                        samples_equal_video_frames=bool(len(t)==n),
                        video_vs_reference_duration_s=float(n/fs-t[-1]) if fs>0 else None)
                    row["metadata_checks_passed"]=bool(fs>0 and n>0 and ok and end_ok and t[-1]>0 and not np.any(differences<0))
                else:
                    row["metadata_checks_passed"]=False
            except (ValueError,OSError) as error:
                row.update(metadata_checks_passed=False,error=str(error))
            row["source_stats_unchanged_during_check"]=all(
                path.stat().st_size==row[label+"_bytes"] and path.stat().st_mtime_ns==row[label+"_mtime_ns"]
                for path,label in ((video,"video"),(reference,"reference")))
        records.append(row)
    return dict(dataset_root=str(root),checked_at=datetime.now().isoformat(),pairs=sum(r["pair_present"] for r in records),
        unique_subject_labels=sorted({r["subject"] for r in records}),records=records,
        checks="File presence, metadata, first/last decoded frame, reference shape/timestamps; NO HR estimation or full-frame decode",
        limits="Matching sample/frame counts do not prove synchronization. No video content hash or complete decode performed. Download completeness not guaranteed.")


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root",nargs="?",type=Path,default=DEFAULT_ROOT)
    parser.add_argument("--output-root",type=Path,default=Path(__file__).resolve().parents[1]/"outputs/new_dataset_inventory")
    args=parser.parse_args()
    report=inspect(args.root.resolve())
    output=args.output_root.resolve()/datetime.now().strftime("inventory_%Y%m%d_%H%M%S_%f")
    output.mkdir(parents=True,exist_ok=False)
    (output/"inventory.json").write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False),encoding="utf-8")
    fields=["subject","pair_present","metadata_checks_passed","video_frames","video_fps","video_duration_s",
            "reference_samples","reference_duration_s","duplicate_timestamps","largest_timestamp_gap_s"]
    with (output/"inventory.csv").open("w",newline="",encoding="utf-8-sig") as stream:
        writer=csv.DictWriter(stream,fieldnames=fields,extrasaction="ignore")
        writer.writeheader();writer.writerows(report["records"])
    print(json.dumps(report,ensure_ascii=False,indent=2))
    print(f"Saved inventory: {output}")
