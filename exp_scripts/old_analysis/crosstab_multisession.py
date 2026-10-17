import json
import argparse


def get_top_k_docs(doc_scores, k):
    sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
    return [doc for doc, _ in sorted_docs[:k]]


def load_jsonl_labels(path):
    labels = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            labels[entry['idx']] = entry['autoeval_label']['label']
    return labels


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--retrieval_result_file', required=True)
    parser.add_argument('--data_file', required=True, help='other_s.json (has idx/gt_docs)')
    parser.add_argument('--reference_file', required=True, help='other_s_original.json (has question_id/question_type)')
    parser.add_argument('--top5_eval_results', required=True)
    parser.add_argument('--oracle_eval_results', required=True)
    parser.add_argument('--question_type', default='multi-session')
    parser.add_argument('--k', type=int, default=5)
    args = parser.parse_args()

    with open(args.retrieval_result_file) as f:
        scores = json.load(f)
    with open(args.data_file) as f:
        samples = json.load(f)
    with open(args.reference_file) as f:
        references = json.load(f)

    qid2qtype = {entry['question_id']: entry['question_type'] for entry in references}
    top5_labels = load_jsonl_labels(args.top5_eval_results)
    oracle_labels = load_jsonl_labels(args.oracle_eval_results)

    buckets = {
        'incomplete_evidence__fixed_by_oracle': [],       # recall_all=0, top5 wrong, oracle correct
        'incomplete_evidence__still_wrong': [],           # recall_all=0, top5 wrong, oracle also wrong
        'incomplete_evidence__top5_lucky_stays_correct': [],   # recall_all=0, top5 correct, oracle also correct
        'incomplete_evidence__top5_lucky_oracle_regressed': [], # recall_all=0, top5 correct, oracle wrong
        'complete_evidence__consistent_correct': [],      # recall_all=1, top5 correct, oracle correct
        'complete_evidence__oracle_regressed': [],        # recall_all=1, top5 correct, oracle wrong
        'complete_evidence__top5_wrong_oracle_fixed': [], # recall_all=1, top5 wrong, oracle correct
        'complete_evidence__both_wrong': [],              # recall_all=1, top5 wrong, oracle wrong
    }

    n_total = 0
    for sample in samples:
        qid = str(sample.get('idx'))
        if qid2qtype.get(qid) != args.question_type:
            continue
        if qid not in top5_labels or qid not in oracle_labels:
            continue
        n_total += 1

        gt_docs = [str(ch) for ch in sample.get('gt_docs', [])]
        doc_scores = scores[qid]
        top_k_docs = get_top_k_docs(doc_scores, args.k)
        recall_all = all(ch in top_k_docs for ch in gt_docs)

        top5_correct = top5_labels[qid]
        oracle_correct = oracle_labels[qid]

        if not recall_all:
            if not top5_correct and oracle_correct:
                buckets['incomplete_evidence__fixed_by_oracle'].append(qid)
            elif not top5_correct and not oracle_correct:
                buckets['incomplete_evidence__still_wrong'].append(qid)
            elif top5_correct and oracle_correct:
                buckets['incomplete_evidence__top5_lucky_stays_correct'].append(qid)
            else:
                buckets['incomplete_evidence__top5_lucky_oracle_regressed'].append(qid)
        else:
            if top5_correct and oracle_correct:
                buckets['complete_evidence__consistent_correct'].append(qid)
            elif top5_correct and not oracle_correct:
                buckets['complete_evidence__oracle_regressed'].append(qid)
            elif not top5_correct and oracle_correct:
                buckets['complete_evidence__top5_wrong_oracle_fixed'].append(qid)
            else:
                buckets['complete_evidence__both_wrong'].append(qid)

    print(f"question_type={args.question_type}, k={args.k}, matched samples={n_total}")
    print()
    n_incomplete = sum(len(buckets[b]) for b in buckets if b.startswith('incomplete_evidence'))
    n_complete = sum(len(buckets[b]) for b in buckets if b.startswith('complete_evidence'))
    print(f"Incomplete evidence (Recall-all@{args.k}=0): {n_incomplete} samples")
    for b in ['incomplete_evidence__fixed_by_oracle', 'incomplete_evidence__still_wrong', 'incomplete_evidence__top5_lucky_stays_correct', 'incomplete_evidence__top5_lucky_oracle_regressed']:
        print(f"  {b}: {len(buckets[b])}")
    print()
    print(f"Complete evidence (Recall-all@{args.k}=1): {n_complete} samples")
    for b in ['complete_evidence__consistent_correct', 'complete_evidence__oracle_regressed', 'complete_evidence__top5_wrong_oracle_fixed', 'complete_evidence__both_wrong']:
        print(f"  {b}: {len(buckets[b])}")

    print()
    total_top5_correct = sum(1 for b in buckets for qid in buckets[b] if (top5_labels[qid]))
    total_oracle_correct = sum(1 for b in buckets for qid in buckets[b] if (oracle_labels[qid]))
    print(f"Sanity check -> top5 accuracy: {total_top5_correct}/{n_total} = {total_top5_correct/n_total:.4f}")
    print(f"Sanity check -> oracle accuracy: {total_oracle_correct}/{n_total} = {total_oracle_correct/n_total:.4f}")
    net_gain = len(buckets['incomplete_evidence__fixed_by_oracle']) + len(buckets['complete_evidence__top5_wrong_oracle_fixed']) \
        - len(buckets['complete_evidence__oracle_regressed']) - len(buckets['incomplete_evidence__top5_lucky_oracle_regressed'])
    print(f"Net samples flipped wrong->correct by oracle: {net_gain} "
          f"(of which {len(buckets['incomplete_evidence__fixed_by_oracle'])} were specifically missing-evidence cases)")

    print()
    print("=== idx lists per bucket (for manual inspection) ===")
    for b, qids in buckets.items():
        print(f"{b} ({len(qids)}): {qids}")


if __name__ == "__main__":
    main()
