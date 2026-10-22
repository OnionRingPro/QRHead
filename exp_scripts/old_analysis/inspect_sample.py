import json
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--idx', required=True, nargs='+', help='one or more sample idx to inspect')
    parser.add_argument('--retrieval_data_file', required=True, help='other_s.json (idx/gt_docs/paragraphs)')
    parser.add_argument('--reference_file', required=True, help='other_s_original.json (question_id/answer/question_type)')
    parser.add_argument('--retrieval_result_file', required=True, help='lme_retrieval.json (scores)')
    parser.add_argument('--generation_result_file', required=True, help='lme_generation.json (hypotheses, jsonl)')
    parser.add_argument('--eval_results_file', required=True, help='lme_generation.json.eval-results (labels, jsonl)')
    parser.add_argument('--snippet_len', type=int, default=200)
    args = parser.parse_args()

    with open(args.retrieval_data_file) as f:
        retrieval_samples = {str(s['idx']): s for s in json.load(f)}
    with open(args.reference_file) as f:
        references = {e['question_id']: e for e in json.load(f)}
    with open(args.retrieval_result_file) as f:
        scores = json.load(f)
    hypotheses = {}
    with open(args.generation_result_file) as f:
        for line in f:
            line = line.strip()
            if line:
                e = json.loads(line)
                hypotheses[e['idx']] = e['hypothesis']
    eval_labels = {}
    with open(args.eval_results_file) as f:
        for line in f:
            line = line.strip()
            if line:
                e = json.loads(line)
                eval_labels[e['idx']] = e['autoeval_label']['label']

    for target in args.idx:
        print("=" * 100)
        print(f"idx = {target}")

        ref = references.get(target)
        rsample = retrieval_samples.get(target)

        if ref is None or rsample is None:
            print("  NOT FOUND in reference/retrieval data file")
            continue

        print(f"question_type: {ref.get('question_type')}")
        print(f"question: {ref.get('question')}")
        print(f"answer (ground truth): {ref.get('answer')}")
        print()

        gt_docs = [str(d) for d in rsample.get('gt_docs', [])]
        print(f"gt_docs ({len(gt_docs)}): {gt_docs}")

        para_text = {str(p['idx']): p.get('paragraph_text', '') for p in rsample.get('paragraphs', [])}

        doc_scores = scores.get(target, {})
        ranked = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
        rank_of = {doc_id: i + 1 for i, (doc_id, _) in enumerate(ranked)}
        n_docs = len(ranked)

        print()
        print(f"--- rank of each gt_doc among {n_docs} candidates (top-5 cutoff shown) ---")
        for d in gt_docs:
            r = rank_of.get(d)
            sc = doc_scores.get(d)
            text = para_text.get(d, '<paragraph text not found>')
            snippet = text[:args.snippet_len].replace('\n', ' ')
            in_top5 = "IN TOP-5" if r is not None and r <= 5 else "missed"
            print(f"  [{in_top5}] rank={r}/{n_docs}  score={sc}")
            print(f"      text: {snippet}...")

        print()
        print("--- top-5 retrieved (what the reader actually saw) ---")
        for doc_id, sc in ranked[:5]:
            text = para_text.get(doc_id, '<paragraph text not found>')
            snippet = text[:args.snippet_len].replace('\n', ' ')
            tag = "GT" if doc_id in gt_docs else "  "
            print(f"  [{tag}] score={sc:.4f}  {doc_id}")
            print(f"      text: {snippet}...")

        print()
        print(f"model hypothesis (top-5 retrieval): {hypotheses.get(target, '<not found>')}")
        print(f"judge label (top-5 retrieval correct?): {eval_labels.get(target, '<not found>')}")
        print()


if __name__ == "__main__":
    main()
