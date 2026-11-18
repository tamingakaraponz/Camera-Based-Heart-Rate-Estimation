"""Prepare matching source frames and saved ROI landmarks for visual review."""
import hashlib
import json
import math
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT.parent / "dataset for ipph"
GEOMETRY = ROOT / "outputs/roi_overlay_abc_20261116_171220"
OUT = ROOT / "outputs/roi_review_frames"


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    records = []
    for number in (3, 5, 8, 9, 10, 11):
        subject = f"subject{number}"
        videos = list(DATA.glob(f"{subject}-*/{subject}/vid.avi"))
        if len(videos) != 1:
            raise ValueError(f"{subject}: expected one video, found {len(videos)}")
        video = videos[0]
        video_hash = sha256(video)
        cap = cv2.VideoCapture(str(video))
        fps = cap.get(cv2.CAP_PROP_FPS)
        if not cap.isOpened() or not math.isfinite(fps) or fps <= 0:
            raise ValueError(f"Cannot open video or invalid FPS: {video}")
        for second in (2, 10, 20, 30, 40, 50):
            index = math.ceil(second * fps)
            record = {"subject": subject, "video": str(video), "video_sha256": video_hash,
                      "fps": fps, "requested_seconds": second, "frame_index": index,
                      "actual_seconds": index / fps, "status": "missing"}
            source = GEOMETRY / f"{subject}_{second}s_points.npz"
            cap.set(cv2.CAP_PROP_POS_FRAMES, index)
            ok, frame = cap.read()
            if ok and source.exists():
                with np.load(source) as data:
                    points = data["landmarks"]
                    regions = {key: data[key] for key in "ABC"}
                    axis = points[263] - points[33]
                    scale = np.linalg.norm(axis)
                    if not np.isfinite(scale) or scale <= 0:
                        raise ValueError(f"Invalid landmarks in {source}")
                    axis = axis / scale
                    vertical = np.array([-axis[1], axis[0]])
                    if np.dot(vertical, points[152] - points[10]) < 0:
                        vertical = -vertical
                    center = (points[9] + points[168]) / 2
                    regions["D"] = np.rint(np.array([
                        center - axis * scale * .07 - vertical * scale * .04,
                        center + axis * scale * .07 - vertical * scale * .04,
                        center + axis * scale * .07 + vertical * scale * .04,
                        center - axis * scale * .07 + vertical * scale * .04,
                    ])).astype(np.int32)
                height, width = frame.shape[:2]
                valid = all(cv2.contourArea(poly) > 0 and np.all(poly[:, 0] >= 0)
                            and np.all(poly[:, 0] < width) and np.all(poly[:, 1] >= 0)
                            and np.all(poly[:, 1] < height) for poly in regions.values())
                image = OUT / f"{subject}_{second}s.png"
                if not cv2.imwrite(str(image), frame):
                    raise OSError(f"Failed to write {image}")
                overlay = frame.copy()
                for label, poly in regions.items():
                    cv2.polylines(overlay, [poly], True, (0, 255, 255), 1)
                    cv2.putText(overlay, label, tuple(poly[0]), cv2.FONT_HERSHEY_SIMPLEX,
                                .45, (0, 0, 255), 1)
                cv2.imwrite(str(OUT / f"{subject}_{second}s_overlay.png"), overlay)
                record.update(status="ok" if valid else "invalid_geometry", width=width,
                              height=height, image=image.name, image_sha256=sha256(image),
                              polygons={key: value.tolist() for key, value in regions.items()},
                              geometry_source=str(source), geometry_sha256=sha256(source),
                              geometry_note="Cached landmarks from earlier continuous replay; frame association inferred from audit code.")
            records.append(record)
        cap.release()
        print(subject, flush=True)
    (OUT / "manifest.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
    print(f"Prepared {sum(r['status'] == 'ok' for r in records)}/{len(records)} frames in {OUT}")


if __name__ == "__main__":
    main()
