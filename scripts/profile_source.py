"""Full source profile without using source prevalence to choose the model cohort."""
from pathlib import Path
import csv
import json
from collections import Counter

ROOT = Path(__file__).resolve().parents[1]


def main():
    patients, events = set(), set()
    labels, drugs, cultures = Counter(), Counter(), Counter()
    time_min, time_max, rows = None, None, 0
    source = ROOT / 'data/raw/microbiology_cultures_cohort.csv'
    with source.open(newline='', encoding='utf-8') as stream:
        for record in csv.DictReader(stream):
            rows += 1
            patients.add(record['anon_id'])
            events.add((record['anon_id'], record['order_proc_id_coded']))
            labels[record['susceptibility']] += 1
            drugs[record['antibiotic']] += 1
            cultures[record['culture_description']] += 1
            t = record['order_time_jittered_utc']
            if t and t.lower() != 'null':
                time_min = min(time_min, t) if time_min else t
                time_max = max(time_max, t) if time_max else t
    result = {'source_file': source.name, 'rows': rows, 'patients': len(patients), 'culture_events': len(events),
              'shifted_time_min': time_min, 'shifted_time_max': time_max, 'label_counts': dict(labels),
              'antibiotic_counts': dict(drugs), 'culture_type_counts': dict(cultures),
              'note': 'Source profile only. Includes negative cultures and drugs outside the analysis panel.'}
    out = ROOT / 'results'
    out.mkdir(exist_ok=True)
    (out / 'full_source_profile.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if not k.endswith('counts')}, indent=2))


if __name__ == '__main__':
    main()
