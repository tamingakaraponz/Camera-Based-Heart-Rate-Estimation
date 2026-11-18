"""Local, prediction-blind annotation for the prepared ROI review images."""
import argparse
import hashlib
import json
from pathlib import Path
import tkinter as tk
from tkinter import messagebox, ttk

import cv2
import numpy as np
from PIL import Image, ImageTk

ROOT = Path(__file__).resolve().parents[1]
REVIEW = ROOT / 'outputs' / 'roi_review_frames'
OUTPUT = ROOT / 'outputs' / 'roi_annotations'
COLORS = {0: (255, 210, 0), 1: (0, 220, 80), 2: (255, 50, 50), 3: (60, 130, 255)}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class AnnotationApp:
    def __init__(self, window, forehead_review=False):
        self.window = window
        self.records = json.loads((REVIEW / 'manifest.json').read_text(encoding='utf8'))
        self.records = [r for r in self.records if r['status'] == 'ok']
        self.output = OUTPUT
        self.forehead_review = forehead_review
        if forehead_review:
            self.output = ROOT / 'outputs' / 'forehead_human_review'
            self.records = [dict(r, polygons={'A': r['polygons']['A']}) for r in self.records
                            if r['subject'] in ('subject5', 'subject11')]
            self.records.sort(key=lambda r: (r['subject'] != 'subject11', -r['requested_seconds']))
        self.index = 0
        self.label = tk.IntVar(value=1)
        self.mode = tk.StringVar(value='brush')
        self.size = tk.IntVar(value=5)
        self.zoom = tk.IntVar(value=3)
        self.show_labels = tk.BooleanVar(value=True)
        self.annotator = tk.StringVar(value='')
        self.history = []
        self.vertices = []
        self.last = None
        window.title('ทำภาพเฉลยบริเวณใบหน้า')
        top = ttk.Frame(window, padding=8)
        top.pack(fill='x')
        ttk.Label(top, text='ผู้ทำเฉลย').pack(side='left')
        ttk.Entry(top, textvariable=self.annotator, width=14).pack(side='left', padx=5)
        self.choice = ttk.Combobox(top, state='readonly', width=22, values=[f"{r['subject']} | {r['requested_seconds']} วินาที" for r in self.records])
        self.choice.pack(side='left', padx=5)
        self.choice.bind('<<ComboboxSelected>>', self.select_image)
        for value, text in [(1, 'ผิว'), (2, 'ผม/ขน'), (3, 'สิ่งอื่น'), (0, 'ดูไม่ออก')]:
            ttk.Radiobutton(top, text=text, variable=self.label, value=value).pack(side='left')
        controls = ttk.Frame(window, padding=8)
        controls.pack(fill='x')
        for value, text in [('brush', 'พู่กัน'), ('polygon', 'รูปหลายเหลี่ยม')]:
            ttk.Radiobutton(controls, text=text, variable=self.mode, value=value, command=self.cancel_polygon).pack(side='left')
        ttk.Label(controls, text='ขนาดพู่กัน').pack(side='left', padx=(10, 2))
        ttk.Spinbox(controls, from_=1, to=100, width=4, textvariable=self.size).pack(side='left')
        ttk.Label(controls, text='ขยาย').pack(side='left', padx=(10, 2))
        ttk.Spinbox(controls, from_=1, to=8, width=4, textvariable=self.zoom, command=self.render).pack(side='left')
        ttk.Button(controls, text='ปิดรูปหลายเหลี่ยม', command=self.finish_polygon).pack(side='left', padx=4)
        ttk.Button(controls, text='ย้อนกลับ', command=self.undo).pack(side='left', padx=4)
        ttk.Checkbutton(controls, text='แสดงสีเฉลย', variable=self.show_labels, command=self.render).pack(side='left', padx=4)
        ttk.Button(controls, text='บันทึกร่าง', command=lambda: self.save(False)).pack(side='left', padx=4)
        ttk.Button(controls, text='บันทึกฉบับตรวจครบ', command=lambda: self.save(True)).pack(side='left', padx=4)
        self.status = ttk.Label(window, padding=8)
        self.status.pack(fill='x')
        frame = ttk.Frame(window)
        frame.pack(fill='both', expand=True)
        self.canvas = tk.Canvas(frame, background='#222222')
        hscroll = ttk.Scrollbar(frame, orient='horizontal', command=self.canvas.xview)
        vscroll = ttk.Scrollbar(frame, orient='vertical', command=self.canvas.yview)
        self.canvas.configure(xscrollcommand=hscroll.set, yscrollcommand=vscroll.set)
        self.canvas.grid(row=0, column=0, sticky='nsew')
        vscroll.grid(row=0, column=1, sticky='ns')
        hscroll.grid(row=1, column=0, sticky='ew')
        frame.rowconfigure(0, weight=1)
        frame.columnconfigure(0, weight=1)
        self.canvas.bind('<ButtonPress-1>', self.press)
        self.canvas.bind('<B1-Motion>', self.move)
        self.canvas.bind('<ButtonRelease-1>', self.release)
        self.choice.current(0)
        self.load()

    def select_image(self, event):
        if not messagebox.askyesno('เปลี่ยนภาพ', 'บันทึกร่างแล้วหรือยัง? กดใช่เพื่อเปลี่ยนภาพ'):
            self.choice.current(self.index)
            return
        self.index = self.choice.current()
        self.load()

    def load(self):
        self.record = self.records[self.index]
        path = REVIEW / self.record['image']
        if digest(path) != self.record['image_sha256']:
            raise ValueError('Image hash does not match manifest')
        self.image = cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)
        h, w = self.image.shape[:2]
        self.target = np.zeros((h, w), np.uint8)
        for polygon in self.record['polygons'].values():
            cv2.fillPoly(self.target, [np.array(polygon, dtype=np.int32)], 1)
        self.labels = np.full((h, w), 255, np.uint8)
        self.previous_meta = None
        self.vertices = []
        self.history = []
        self.output.mkdir(parents=True, exist_ok=True)
        self.base = self.output / Path(self.record['image']).stem
        label_path = self.base.with_suffix('.png')
        meta_path = self.base.with_suffix('.json')
        if label_path.exists() and meta_path.exists():
            meta = json.loads(meta_path.read_text(encoding='utf8'))
            saved = cv2.imread(str(label_path), cv2.IMREAD_UNCHANGED)
            if (meta['image_sha256'] != self.record['image_sha256'] or saved is None
                    or saved.shape != (h, w) or digest(label_path) != meta['labels_sha256']
                    or meta['polygons'] != self.record['polygons']):
                raise ValueError('Saved annotation does not match the source image and ROI')
            self.labels = saved
            self.previous_meta = meta
            self.annotator.set(meta['annotator'])
        self.origin = (0, 0)
        self.end = (w, h)
        if self.forehead_review:
            polygon = np.asarray(self.record['polygons']['A'])
            self.origin = tuple(np.maximum(polygon.min(axis=0) - 35, 0))
            self.end = tuple(np.minimum(polygon.max(axis=0) + 36, [w, h]))
        self.render()

    def render(self):
        view = self.image.copy()
        if self.show_labels.get():
            for label, color in COLORS.items():
                mask = self.labels == label
                view[mask] = (0.5 * view[mask] + 0.5 * np.array(color)).astype(np.uint8)
        for name, polygon in self.record['polygons'].items():
            poly = np.array(polygon, dtype=np.int32)
            cv2.polylines(view, [poly], True, (255, 255, 0), 1)
            cv2.putText(view, name, tuple(poly[0]), cv2.FONT_HERSHEY_SIMPLEX, .4, (255, 255, 0), 1)
        for point in self.vertices:
            cv2.circle(view, point, 2, (255, 255, 255), -1)
        scale = max(1, min(8, self.zoom.get()))
        view = view[self.origin[1]:self.end[1], self.origin[0]:self.end[0]]
        view = cv2.resize(view, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
        self.photo = ImageTk.PhotoImage(Image.fromarray(view))
        self.canvas.delete('all')
        self.canvas.create_image(0, 0, anchor='nw', image=self.photo)
        self.canvas.configure(scrollregion=(0, 0, view.shape[1], view.shape[0]))
        remaining = int(((self.target > 0) & (self.labels == 255)).sum())
        self.status.config(text=f'ยังไม่ได้ตรวจ {remaining} จุดในกรอบ | ผิว=เขียว ผม=แดง สิ่งอื่น=น้ำเงิน ดูไม่ออก=เหลือง | วาดรูปหลายเหลี่ยมแล้วกดปิดรูป')

    def position(self, event):
        scale = max(1, min(8, self.zoom.get()))
        x = self.origin[0] + int(self.canvas.canvasx(event.x) / scale)
        y = self.origin[1] + int(self.canvas.canvasy(event.y) / scale)
        return (max(0, min(self.labels.shape[1] - 1, x)), max(0, min(self.labels.shape[0] - 1, y)))

    def remember(self):
        self.history.append(self.labels.copy())
        if len(self.history) > 30:
            self.history.pop(0)

    def press(self, event):
        point = self.position(event)
        if self.mode.get() == 'polygon':
            self.vertices.append(point)
        else:
            self.remember()
            self.last = point
            cv2.circle(self.labels, point, max(0, self.size.get() // 2), self.label.get(), -1)
        self.render()

    def move(self, event):
        if self.mode.get() == 'brush' and self.last is not None:
            point = self.position(event)
            cv2.line(self.labels, self.last, point, self.label.get(), max(1, self.size.get()))
            self.last = point
            self.render()

    def release(self, event):
        self.last = None

    def finish_polygon(self):
        if len(self.vertices) < 3:
            messagebox.showinfo('รูปหลายเหลี่ยม', 'เลือกอย่างน้อย 3 จุด')
            return
        self.remember()
        cv2.fillPoly(self.labels, [np.array(self.vertices, dtype=np.int32)], self.label.get())
        self.vertices = []
        self.render()

    def cancel_polygon(self):
        self.vertices = []
        self.render()

    def undo(self):
        if self.vertices:
            self.vertices.pop()
        elif self.history:
            self.labels = self.history.pop()
        self.render()

    def save(self, complete):
        name = self.annotator.get().strip()
        if not name:
            messagebox.showerror('กรอกข้อมูล', 'กรุณาระบุชื่อหรือรหัสผู้ทำเฉลย')
            return
        remaining = int(((self.target > 0) & (self.labels == 255)).sum())
        if complete and remaining:
            messagebox.showerror('ยังตรวจไม่ครบ', f'ยังมี {remaining} จุดในกรอบที่ไม่ได้ตรวจ')
            return
        regions = ' '.join(self.record['polygons'])
        if complete and not messagebox.askyesno('ยืนยันการตรวจ', f'ตรวจทานทุกจุดในกรอบ {regions} แล้วใช่ไหม?'):
            return
        label_path = self.base.with_suffix('.png')
        if self.forehead_review:
            self.labels[self.target == 0] = 255
        if not cv2.imwrite(str(label_path), self.labels):
            raise OSError(f'Cannot write {label_path}')
        meta = {'image_sha256': self.record['image_sha256'], 'frame_index': self.record['frame_index'],
                'subject': self.record['subject'], 'polygons': self.record['polygons'],
                'annotator': name, 'review_status': 'complete' if complete else 'draft',
                'remaining': remaining, 'labels_sha256': digest(label_path),
                'previously_seen_predictions': (self.previous_meta or {}).get('previously_seen_predictions', 'not_recorded'),
                'initial_annotation': (self.previous_meta or {}).get('initial_annotation', (self.previous_meta or {}).get('annotator')),
                'annotation_version': 1}
        self.base.with_suffix('.json').write_text(json.dumps(meta, indent=2), encoding='utf8')
        self.previous_meta = meta
        messagebox.showinfo('บันทึกแล้ว', f'บันทึกที่ {label_path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--forehead-review', action='store_true')
    args = parser.parse_args()
    root = tk.Tk()
    root.geometry('1200x850')
    AnnotationApp(root, args.forehead_review)
    root.mainloop()
