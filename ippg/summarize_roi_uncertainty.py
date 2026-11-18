"""Summarize selected uncertain pixels in the unreviewed AI draft."""
import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/roi_review_development'
PEOPLE = ('subject3', 'subject5', 'subject8', 'subject9', 'subject10', 'subject11')


def load(name):
    with (OUT / name).open(encoding='utf8', newline='') as stream:
        return list(csv.DictReader(stream))


def main():
    baseline = load('draft_comparison.csv')
    chroma = load('cheek_color_space_all.csv')
    result = []
    for subject in PEOPLE:
        for method in ('fixed', 'mediapipe', 'cheek_p90', 'color_only_p90'):
            if method == 'color_only_p90':
                rows = [row for row in chroma if row['subject'] == subject and row['mode'] == 'color_only'
                        and row['percentile'] == '90']
            else:
                rows = [row for row in baseline if row['subject'] == subject and row['region'] == 'A'
                        and row['method'] == method]
            if len(rows) != 6 or any(row['status'] != 'ok' for row in rows):
                raise ValueError(f'Missing results: {subject} {method}')
            selected = sum(int(row['selected']) for row in rows)
            known_other = sum(int(row['fp']) for row in rows)
            uncertain = sum(int(row['selected_unknown']) for row in rows)
            known_skin = sum(int(row['tp']) for row in rows)
            if selected != known_other + uncertain + known_skin:
                raise ValueError(f'Selected counts do not add up: {subject} {method}')
            result.append({
                'subject': subject, 'method': method, 'images': len(rows),
                'selected_pixels': selected, 'draft_skin': known_skin,
                'draft_non_skin': known_other, 'selected_unknown': uncertain,
                'unknown_selected_fraction': uncertain / selected if selected else None,
                'possible_min_nonskin_fraction': known_other / selected if selected else None,
                'possible_max_nonskin_fraction': (known_other + uncertain) / selected if selected else None,
                'disclaimer': 'Conditional on unreviewed AI draft labels; pixels are not independent',
            })
    path = OUT / 'draft_uncertainty_summary.csv'
    with path.open('w', encoding='utf8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=result[0].keys())
        writer.writeheader()
        writer.writerows(result)
    for row in result:
        if row['subject'] in ('subject5', 'subject11'):
            print(row)


if __name__ == '__main__':
    main()
