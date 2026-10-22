import json
import argparse

SEPARATOR = "=" * 100


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--labels_file', required=True)
    parser.add_argument('--records_json', required=True)
    parser.add_argument('--records_txt', required=True)
    args = parser.parse_args()

    with open(args.labels_file) as f:
        labels = json.load(f)
    with open(args.records_json) as f:
        records = json.load(f)

    record_ids = [r['idx'] for r in records]
    missing = set(record_ids) - set(labels)
    extra = set(labels) - set(record_ids)
    assert not missing and not extra, f"label/record mismatch: missing={missing} extra={extra}"

    # manual_label is the last key, so these land right after it; existing manual labels are untouched
    for r in records:
        r['ai_label'] = labels[r['idx']]['ai_label']
        r['ai_label_reason'] = labels[r['idx']]['ai_label_reason']

    with open(args.records_json, 'w') as f:
        json.dump(records, f, indent=2)

    with open(args.records_txt) as f:
        lines = f.read().split('\n')

    out = []
    inserted = 0
    i = 0
    while i < len(lines):
        line = lines[i]
        out.append(line)
        # block header is: separator / "idx: ..." / "manual_label: ..."
        if line == SEPARATOR and i + 2 < len(lines) and lines[i + 1].startswith('idx: ') and lines[i + 2].startswith('manual_label:'):
            idx = lines[i + 1].split()[1]
            out.append(lines[i + 1])
            out.append(lines[i + 2])
            i += 3
            if i < len(lines) and lines[i].startswith('ai_label:'):
                i += 1  # replace a previously inserted ai_label line instead of duplicating it
            lab = labels[idx]
            out.append(f"ai_label: {lab['ai_label']}    (reason: {lab['ai_label_reason']})")
            inserted += 1
            continue
        i += 1

    assert inserted == len(records), f"inserted {inserted} ai_label lines but there are {len(records)} records"

    with open(args.records_txt, 'w') as f:
        f.write('\n'.join(out))

    print(f"Applied ai_label to {len(records)} records in json and {inserted} blocks in txt")


if __name__ == "__main__":
    main()
