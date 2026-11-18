"""Summarize ROI development results, preserving their draft-reference limits."""
import csv
import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/roi_review_development'
PEOPLE = ('subject3', 'subject5', 'subject8', 'subject9', 'subject10', 'subject11')


def read_csv(name):
    with (OUT / name).open(encoding='utf8', newline='') as stream:
        return list(csv.DictReader(stream))


def calculate(rows):
    valid = [float(row['retention']) for row in rows if row.get('retention') not in ('', None)]
    return {
        'images': len(rows), 'with_skin': len(valid),
        'mean_retention': statistics.mean(valid) if valid else None,
        'mean_unknown': statistics.mean(float(row['unknown_fraction']) for row in rows),
        'selected_known_nonskin': sum(int(row['fp']) for row in rows),
        'selected_unknown': sum(int(row['selected_unknown']) for row in rows),
    }


def percent(value):
    return 'N/A' if value is None else f'{100 * value:.1f}%'


def main():
    baseline = read_csv('draft_comparison.csv')
    chroma = read_csv('cheek_color_space_all.csv')
    methods = ('fixed', 'mediapipe', 'cheek_p90', 'color_only_p90')
    summary = []
    for person in PEOPLE:
        for method in methods:
            if method == 'color_only_p90':
                rows = [row for row in chroma if row['subject'] == person
                        and row['mode'] == 'color_only' and row['percentile'] == '90']
            else:
                rows = [row for row in baseline if row['subject'] == person
                        and row['region'] == 'A' and row['method'] == method]
            if len(rows) != 6 or any(row['status'] != 'ok' for row in rows):
                raise ValueError(f'Missing draft results: {person}, {method}')
            summary.append({'subject': person, 'method': method, **calculate(rows)})
    with (OUT / 'draft_subject_summary.csv').open('w', encoding='utf8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=summary[0].keys())
        writer.writeheader()
        writer.writerows(summary)

    text = [
        '# สรุปผลพัฒนาการเลือกผิวจาก 36 ภาพ',
        '',
        'ภาพจาก 6 คน คนละ 6 ช่วงเวลา เฉลยทุกภาพเป็นร่างที่ AI จัดทำหลังเคยเห็นผลบางส่วนและยังไม่ได้รับการตรวจทานโดยคน ผลใช้สำรวจแนวทาง',
        '',
        'ตารางแสดงค่าเฉลี่ยรายภาพของสัดส่วนหน้าผากที่วิธีเลือกได้จากส่วนที่ร่างระบุเป็นผิว เครื่องหมาย — หมายถึงร่างไม่มีจุดผิวให้คำนวณ',
        '',
        '| คน | สีเดิม | MediaPipe | สีแก้ม LAB P90 | สีแก้มเฉพาะสี P90 | เฉลี่ยส่วนที่ดูไม่ชัด |',
        '|---|---:|---:|---:|---:|---:|',
    ]
    for person in PEOPLE:
        person_rows = [row for row in summary if row['subject'] == person]
        values = [percent(row['mean_retention']) for row in person_rows]
        text.append(f"| {person} | {' | '.join(values)} | {percent(person_rows[0]['mean_unknown'])} |")
    text.extend([
        '',
        'Subject 11 ไม่มีผิวที่ระบุไว้ในร่างหน้าผากทั้งหกภาพ จึงคำนวณสัดส่วนผิวที่เลือกไม่ได้ ร่างระบุผมเพียงบางส่วนในแต่ละภาพ การวัดสิ่งที่เลือกปนยังครอบคลุมพื้นที่ที่มีผมน้อย',
        '',
        'ในหน้าผากของ Subject 5 และ 11 มีพื้นที่ดูไม่ชัดมาก จุดที่ดูไม่ชัดถูกตัดออกจากสูตรผิวและสิ่งปน แต่บันทึกจำนวนที่แต่ละวิธีเลือกไว้ใน draft_subject_summary.csv',
        '',
        'การระบุแก้มเป็นผิวทั้งหมดมาจากการดูภาพร่าง ไม่มีตัวอย่างสิ่งบังที่เฉลยไว้เพียงพอสำหรับสรุปความสามารถแยกผม ผล MediaPipe 100% ข้างต้นเกิดจากพื้นที่ที่ร่างระบุเป็นผิวและไม่ได้บอกความถูกต้องรวม',
        '',
        '## เวลาจากวิดีโอ',
        '',
        '| วิธี | ค่ากลางของเวลารายคน (ms/เฟรม) | P95 รายคนต่ำสุด–สูงสุด (ms) |',
        '|---|---:|---:|',
    ])
    timing_specs = [
        ('สีเดิม', ROOT / 'outputs/roi_method_timing/summary.json', 'fixed'),
        ('MediaPipe', ROOT / 'outputs/roi_method_timing/summary.json', 'mediapipe'),
        ('สีแก้ม LAB P90', ROOT / 'outputs/roi_method_timing/summary.json', 'cheek_p90'),
        ('สีแก้มเฉพาะสี P90', ROOT / 'outputs/roi_chroma_timing/summary.json', 'cheek_chroma_p90'),
    ]
    for label, path, method in timing_specs:
        timing = json.loads(path.read_text(encoding='utf8'))
        runs = [run for run in timing['runs'] if run['method'] == method]
        if len(runs) != 6:
            raise ValueError(f'Missing timing runs: {method}')
        med = statistics.median(run['total_roi_ms_median'] for run in runs)
        low = min(run['total_roi_ms_p95'] for run in runs)
        high = max(run['total_roi_ms_p95'] for run in runs)
        text.append(f'| {label} | {med:.1f} | {low:.1f}–{high:.1f} |')
    text.extend([
        '',
        'จับเวลาคนละรอบบนวิดีโอคนละ 10 วินาที หลัง 30 เฟรมเริ่มต้น รวมติดตามหน้า สร้างกรอบ และเลือกผิว ยังไม่รวมอ่านวิดีโอ แสดงผลหรือประมวลผลชีพจร สภาพเครื่องระหว่างรอบอาจต่างกัน วิธีสีเดิมจับเวลาแบบมีสถานะต่อเนื่อง ต่างจากการประเมินภาพนิ่ง',
        '',
        '## งานที่เหลือ',
        '',
        'ตรวจเฉลยทั้ง 36 ภาพ โดยเฉพาะเส้นผมและจุดดูไม่ชัด แล้วประเมินใหม่ในข้อมูลที่มีผมและสิ่งบังมากพอ ก่อนสรุปวิธีที่เลือกผิวถูกที่สุด ทดสอบคนใหม่แยกจากคนที่ใช้พัฒนาหลังล็อกค่าตั้ง',
    ])
    (OUT / 'ROI_36_DRAFT_SUMMARY.md').write_text('\n'.join(text) + '\n', encoding='utf8')
    print(f'Wrote {len(summary)} subject-method summaries and draft report.')


if __name__ == '__main__':
    main()
