import json
import argparse
from collections import defaultdict
from typing import Dict, List, Set, Tuple, Any


def get_top_k_docs(doc_scores: Dict[str, float], k: int) -> List[str]:
    """Get the top-k docs based on scores."""
    sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
    return [doc for doc, _ in sorted_docs[:k]]

def compute_auroc(doc_scores: Dict[str, float], gt_docs: Set[str]) -> float:
    """Probability that a gold doc outscores a non-gold one (ties count half), from average ranks."""
    sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1])
    ranks = {}
    i = 0
    while i < len(sorted_docs):
        j = i
        while j < len(sorted_docs) and sorted_docs[j][1] == sorted_docs[i][1]:
            j += 1
        for doc, _ in sorted_docs[i:j]:
            ranks[doc] = (i + 1 + j) / 2 # tied docs share the mean of 1-based ranks i+1..j
        i = j

    n_pos, n_neg = len(gt_docs), len(doc_scores) - len(gt_docs)
    assert n_neg > 0, "AUROC is undefined when every doc is gold"
    return (sum(ranks[d] for d in gt_docs) - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)

def compute_recall(samples: List[Dict[str, Any]], scores: Dict[str, Dict[str, float]], auroc: bool) -> Tuple[float, float, float]:
    """Mean Recall@|gold|, its random baseline, and AUROC (0 if not computed) over samples."""
    recall = 0
    random = 0
    auc = 0

    for sample in samples:
        qid = str(sample['idx'])
        gt_docs = set(str(ch) for ch in sample['gt_docs'])
        doc_scores = scores[qid]

        # k = number of gold docs, so a perfect ranking scores 1 whatever the gold size
        top_k_docs = get_top_k_docs(doc_scores, len(gt_docs))
        recall += len([ch for ch in top_k_docs if ch in gt_docs]) / len(gt_docs)
        # a random ranking puts each doc in the top-k with probability k / #docs
        random += len(gt_docs) / len(doc_scores)
        if auroc:
            auc += compute_auroc(doc_scores, gt_docs)

    n = len(samples)
    return recall / n, random / n, auc / n

def main():

    parser = argparse.ArgumentParser(description='Compute Recall@|gold| (with its random baseline) for doc retrieval on oolong, broken down by filter_type')
    parser.add_argument('--retrieval_result_file', required=True, help='Path to the retrieval scores JSON file')
    parser.add_argument('--data_file', required=True, help='Path to the oolong data JSON file from prepare_oolong.py (has gt_docs and filter_type)')
    parser.add_argument('--auroc', action='store_true', help='Also report AUROC (random ranking = 0.5)')

    args = parser.parse_args()

    # Load data
    with open(args.retrieval_result_file, 'r') as f:
        scores = json.load(f)
    with open(args.data_file, 'r') as f:
        samples = json.load(f)

    print(f"Loaded {len(scores)} score entries and {len(samples)} ground truth samples")
    # whole-context questions (prepare_oolong.py --all_questions) have no gold lines to recall
    n_no_gold = sum(not sample['gt_docs'] for sample in samples)
    if n_no_gold:
        samples = [sample for sample in samples if sample['gt_docs']]
        print(f"Skipping {n_no_gold} questions without gold docs; scoring {len(samples)}")

    samples_by_type = defaultdict(list)
    for sample in samples:
        samples_by_type[sample['filter_type']].append(sample)

    print("Recall@|gold| (k = number of gold docs of each query)")
    for name, group in [("ALL", samples)] + sorted(samples_by_type.items()):
        recall, random, auc = compute_recall(group, scores, args.auroc)
        line = f"  [{name} ({len(group)})] Recall: {recall:.4f}  Random: {random:.4f}"
        if args.auroc:
            line += f"  AUROC: {auc:.4f}"
        print(line)

if __name__ == "__main__":
    main()
