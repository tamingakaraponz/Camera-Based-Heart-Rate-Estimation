"""Quantify how unknown draft pixels affect the possible contamination range."""
import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/roi_review_development'


def main():
    with (OUT / 'draft_comparison.csv').open(encoding='utf8', newline='') as stream:
        common = list(csv.DictReader(stream))
    with (OUT / 'cheek_color_space_all.csv').open(encoding='utf8', newline='') as stream:
        chroma = list(csv.DictReader(stream))
    records = []
    for person in ('subject5', 'subject11'):
        for name in ('fixed', 'mediapipe', 'cheek_p90', 'color_only'):
            if name == 'color_only':
                matches = [r for r in chroma if r['subject'] == person and r['time_s'] == '2'
                           and r['mode'] == 'color_only' and r['percentile'] == '90']
            else:
                matches = [r for r in common if r['subject'] == person and r['time_s'] == '2'
                           and r['region'] == 'A' and r['method'] == name]
            if len(matches) != 1 or matches[0]['status'] != 'ok':
                raise ValueError(f'Missing valid draft: {person} {name}')
            item = matches[0]
            skin, other = int(item['tp']), int(item['fp'])
            unknown, selected = int(item['selected_unknown']), int(item['selected'])
            if selected != skin + other + unknown:
                raise ValueError('Selected counts do not add up')
            records.append({'subject': person, 'method': name, 'selected': selected,
                            'draft_skin': skin, 'draft_non_skin': other, 'unknown': unknown,
                            'min_nonskin_fraction': other / selected if selected else None,
                            'max_nonskin_fraction': (other + unknown) / selected if selected else None})
    with (OUT / 'hair_uncertainty.csv').open('w', encoding='utf8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=records[0].keys())
        writer.writeheader()
        writer.writerows(records)
    for record in records:
        print(record)


if __name__ == '__main__':
    main()
