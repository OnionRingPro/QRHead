import argparse
import json
import os
from typing import Dict, List, Any

from eval_retrieval import compute_recall, get_top_k_docs


def per_query_recall(samples: List[Dict[str, Any]], scores: Dict[str, Dict[str, float]], k: int) -> Dict[str, float]:
    """qid -> fraction of gt docs in the top-k (same definition as `Recall` in compute_recall)."""
    out = {}
    for sample in samples:
        qid = str(sample['idx'])
        gt_docs = [str(d) for d in sample.get('gt_docs', [])]
        top_k_docs = get_top_k_docs(scores[qid], k)
        out[qid] = len([d for d in top_k_docs if d in gt_docs]) / len(gt_docs) if gt_docs else 0
    return out


def main():

    parser = argparse.ArgumentParser(description='Compare retrieval result files on the queries that all of them cover')
    parser.add_argument('--data_file', required=True, help='Path to the ground truth data JSON file')
    parser.add_argument('--runs', nargs='+', required=True, help='name=path pairs; the first run is the baseline')
    parser.add_argument('--k', type=int, default=5)

    args = parser.parse_args()

    runs = {}
    for item in args.runs:
        name, path = item.split('=', 1)
        with open(path, 'r') as f:
            runs[name] = (path, json.load(f))
    with open(args.data_file, 'r') as f:
        samples = json.load(f)

    common = set.intersection(*(set(scores) for _, scores in runs.values()))
    samples = [s for s in samples if str(s['idx']) in common]
    print(f"Comparing {len(runs)} runs on the {len(samples)} queries they all cover, k={args.k}")

    baseline = next(iter(runs))
    base_per_query = per_query_recall(samples, runs[baseline][1], args.k)

    print(f"{'run':<14}{'Recall':>8}{'R-any':>8}{'R-all':>8}{'win/tie/loss vs ' + baseline:>22}{'avg CoT tok':>13}")
    for name, (path, scores) in runs.items():
        recall, recall_any, recall_all = compute_recall(samples, scores, args.k)

        per_query = per_query_recall(samples, scores, args.k)
        wins = sum(per_query[q] > base_per_query[q] for q in per_query)
        losses = sum(per_query[q] < base_per_query[q] for q in per_query)
        ties = len(per_query) - wins - losses

        # CoT runs from run_retrieval.py store their CoTs next to the scores
        avg_cot = '-'
        cot_path = os.path.splitext(path)[0] + '_cot.json'
        if os.path.exists(cot_path):
            with open(cot_path, 'r') as f:
                cots = json.load(f)
            lengths = [len(cots[q]['ids']) for q in common if q in cots]
            if lengths:
                avg_cot = f"{sum(lengths) / len(lengths):.0f}"

        print(f"{name:<14}{recall:>8.4f}{recall_any:>8.4f}{recall_all:>8.4f}{f'{wins}/{ties}/{losses}':>22}{avg_cot:>13}")


if __name__ == "__main__":
    main()
