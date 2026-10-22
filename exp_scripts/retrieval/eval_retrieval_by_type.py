import json
import argparse
from collections import defaultdict
from typing import Dict, List, Tuple, Any


def get_top_k_docs(doc_scores: Dict[str, float], k: int) -> List[str]:
    sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
    return [doc for doc, _ in sorted_docs[:k]]


def compute_recall(samples: List[Dict[str, Any]], scores: Dict[str, Dict[str, float]], k: int) -> Tuple[float, float, float]:
    """Same metric definitions as eval_retrieval.py: Recall (partial credit),
    Recall-any (>=1 gt doc in top-k), Recall-all (all gt docs in top-k)."""
    recall = 0
    recall_any_count = 0
    recall_all_count = 0
    valid_samples = 0

    for sample in samples:
        qid = str(sample.get('idx'))
        gt_docs = [str(ch) for ch in sample.get('gt_docs', [])]

        doc_scores = scores[qid]
        top_k_docs = get_top_k_docs(doc_scores, k)

        any_match = any(ch in top_k_docs for ch in gt_docs)
        all_match = all(ch in top_k_docs for ch in gt_docs)
        count_matches = len([ch for ch in top_k_docs if ch in gt_docs])

        recall += count_matches / len(gt_docs) if gt_docs else 0
        recall_any_count += int(any_match)
        recall_all_count += int(all_match)
        valid_samples += 1

    if valid_samples == 0:
        return 0.0, 0.0, 0.0
    return recall / valid_samples, recall_any_count / valid_samples, recall_all_count / valid_samples


def main():
    parser = argparse.ArgumentParser(description='Compute recall metrics for doc retrieval, broken down by question_type')
    parser.add_argument('--retrieval_result_file', required=True, help='Path to the retrieval scores JSON file (e.g. lme_retrieval.json)')
    parser.add_argument('--data_file', required=True, help='Path to the ground truth data JSON file used for retrieval (e.g. other_s.json), has idx/gt_docs')
    parser.add_argument('--reference_file', required=True, help='Path to the original reference file with question_type (e.g. other_s_original.json)')
    args = parser.parse_args()

    with open(args.retrieval_result_file, 'r') as f:
        scores = json.load(f)
    with open(args.data_file, 'r') as f:
        samples = json.load(f)
    with open(args.reference_file, 'r') as f:
        references = json.load(f)

    qid2qtype = {entry['question_id']: entry['question_type'] for entry in references}

    samples_by_type = defaultdict(list)
    missing_type = 0
    for sample in samples:
        qid = str(sample.get('idx'))
        qtype = qid2qtype.get(qid)
        if qtype is None:
            missing_type += 1
            continue
        samples_by_type[qtype].append(sample)

    print(f"Loaded {len(scores)} score entries, {len(samples)} retrieval samples, {len(references)} reference samples")
    if missing_type:
        print(f"WARNING: {missing_type} retrieval samples had no matching question_type in reference file")
    print(f"Question types found: {sorted(samples_by_type.keys())}")
    print()

    for k in [3, 5, 10]:
        print(f"==== k={k} ====")
        overall_recall, overall_any, overall_all = compute_recall(samples, scores, k)
        print(f"  [ALL {len(samples)}] Recall: {overall_recall:.4f}  Recall-all: {overall_all:.4f}  Recall-any: {overall_any:.4f}")
        for qtype in sorted(samples_by_type.keys()):
            type_samples = samples_by_type[qtype]
            recall, recall_any, recall_all = compute_recall(type_samples, scores, k)
            print(f"  [{qtype} ({len(type_samples)})] Recall: {recall:.4f}  Recall-all: {recall_all:.4f}  Recall-any: {recall_any:.4f}")
        print()


if __name__ == "__main__":
    main()
