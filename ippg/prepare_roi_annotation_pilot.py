"""Extract unmarked pilot frames for independent ROI annotation; no signal processing."""
import hashlib
import json
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT.parent / 'dataset for ipph'
OUT = ROOT / 'outputs' / 'roi_abc_annotation_pilot'


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    records = []
    for number in (3, 5, 8, 9, 10, 11):
        name = f'subject{number}'
        matches = list(DATA.glob(f'{name}-*/{name}/vid.avi'))
        if len(matches) != 1:
            raise RuntimeError(f'{name}: expected one video, found {len(matches)}')
        video = matches[0]
        cap = cv2.VideoCapture(str(video))
        try:
            fps = cap.get(cv2.CAP_PROP_FPS)
            if not cap.isOpened() or not 0 < fps < float('inf'):
                raise RuntimeError(f'{name}: video or FPS invalid')
            index = round(2 * fps)
            cap.set(cv2.CAP_PROP_POS_FRAMES, index)
            ok, frame = cap.read()
            if not ok:
                raise RuntimeError(f'{name}: cannot read frame {index}')
            output = OUT / f'{name}_frame_{index}.png'
            if not cv2.imwrite(str(output), frame):
                raise RuntimeError(f'cannot write {output}')
            records.append(dict(subject=name, video=str(video), video_sha256=sha256(video),
                                fps=fps, frame_index=index, timestamp_seconds=index/fps,
                                width=frame.shape[1], height=frame.shape[0],
                                image=output.name, image_sha256=sha256(output)))
        finally:
            cap.release()
    (OUT / 'manifest.json').write_text(json.dumps(records, indent=2), encoding='utf-8')
    print(f'Prepared {len(records)} unmarked images in {OUT}')


if __name__ == '__main__':
    main()
