import json
import argparse


def load_jsonl(path, key='idx'):
    out = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                e = json.loads(line)
                out[e[key]] = e
    return out


def session_id_of(doc_id):
    # doc_id format: "{question_id}__{session_id}__{turn_idx}"
    parts = doc_id.split('__')
    return parts[1]


def turn_idx_of(doc_id):
    parts = doc_id.split('__')
    return int(parts[2])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--retrieval_data_file', required=True, help='other_s.json')
    parser.add_argument('--reference_file', required=True, help='other_s_original.json')
    parser.add_argument('--retrieval_result_file', required=True, help='lme_retrieval.json')
    parser.add_argument('--top5_generation_file', required=True, help='lme_generation.json (jsonl)')
    parser.add_argument('--top5_eval_results', required=True, help='lme_generation.json.eval-results (jsonl)')
    parser.add_argument('--oracle_generation_file', required=True, help='lme_generation_oracle.json (jsonl)')
    parser.add_argument('--oracle_eval_results', required=True, help='lme_generation_oracle.json.eval-results (jsonl)')
    parser.add_argument('--question_type', default='multi-session')
    parser.add_argument('--k', type=int, default=5)
    parser.add_argument('--out_txt', required=True)
    parser.add_argument('--out_json', required=True)
    args = parser.parse_args()

    with open(args.retrieval_data_file) as f:
        retrieval_samples = {str(s['idx']): s for s in json.load(f)}
    with open(args.reference_file) as f:
        references = {e['question_id']: e for e in json.load(f)}
    with open(args.retrieval_result_file) as f:
        scores = json.load(f)
    top5_hyp = load_jsonl(args.top5_generation_file)
    top5_labels = load_jsonl(args.top5_eval_results)
    oracle_hyp = load_jsonl(args.oracle_generation_file)
    oracle_labels = load_jsonl(args.oracle_eval_results)

    records = []

    for idx, ref in references.items():
        if ref.get('question_type') != args.question_type:
            continue
        if idx not in top5_labels or idx not in oracle_labels:
            continue

        top5_correct = top5_labels[idx]['autoeval_label']['label']
        oracle_correct = oracle_labels[idx]['autoeval_label']['label']

        # target population: oracle succeeded but the real experiment failed
        if top5_correct or not oracle_correct:
            continue

        rsample = retrieval_samples.get(idx)
        if rsample is None:
            continue

        gt_docs = [str(d) for d in rsample.get('gt_docs', [])]
        para_text = {str(p['idx']): p.get('paragraph_text', '') for p in rsample.get('paragraphs', [])}

        doc_scores = scores.get(idx, {})
        ranked = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
        rank_of = {doc_id: i + 1 for i, (doc_id, sc) in enumerate(ranked)}

        recall_all = all(rank_of.get(d, 10**9) <= args.k for d in gt_docs)
        failure_mode = 'complete_evidence__top5_wrong_oracle_fixed' if recall_all else 'incomplete_evidence__fixed_by_oracle'

        # group all paragraphs of the same idx by session, for "locate original text" context
        sessions = {}
        for doc_id in para_text:
            sid = session_id_of(doc_id)
            sessions.setdefault(sid, []).append(doc_id)
        for sid in sessions:
            sessions[sid].sort(key=turn_idx_of)

        gt_doc_info = []
        for d in gt_docs:
            sid = session_id_of(d)
            gt_doc_info.append({
                'doc_id': d,
                'session_id': sid,
                'rank': rank_of.get(d),
                'score': doc_scores.get(d),
                'in_top_k': rank_of.get(d, 10**9) <= args.k,
                'text': para_text.get(d, '<NOT FOUND>'),
                'full_session_doc_ids_in_order': sessions.get(sid, []),
                'full_session_text_in_order': [para_text.get(x, '<NOT FOUND>') for x in sessions.get(sid, [])],
            })

        top_k_retrieved = []
        for doc_id, sc in ranked[:args.k]:
            top_k_retrieved.append({
                'doc_id': doc_id,
                'score': sc,
                'is_gt': doc_id in gt_docs,
                'text': para_text.get(doc_id, '<NOT FOUND>'),
            })

        records.append({
            'idx': idx,
            'question_type': ref.get('question_type'),
            'question': ref.get('question'),
            'answer': ref.get('answer'),
            'question_date': ref.get('question_date'),
            'failure_mode': failure_mode,
            'gt_docs': gt_doc_info,
            'top_k_retrieved': top_k_retrieved,
            'top5_hypothesis': top5_hyp.get(idx, {}).get('hypothesis'),
            'oracle_hypothesis': oracle_hyp.get(idx, {}).get('hypothesis'),
            'manual_label': None,
        })

    print(f"Exported {len(records)} failure cases (question_type={args.question_type}, k={args.k})")

    with open(args.out_json, 'w') as f:
        json.dump(records, f, indent=2)

    with open(args.out_txt, 'w') as f:
        for rec in records:
            f.write("=" * 100 + "\n")
            f.write(f"idx: {rec['idx']}   failure_mode: {rec['failure_mode']}\n")
            f.write(f"manual_label: ____________    (bridging / aggregation / other)\n\n")
            f.write(f"question: {rec['question']}\n")
            f.write(f"answer (ground truth): {rec['answer']}\n")
            f.write(f"question_date: {rec['question_date']}\n\n")

            f.write(f"--- gt_docs ({len(rec['gt_docs'])}) ---\n")
            for g in rec['gt_docs']:
                tag = "IN TOP-K" if g['in_top_k'] else "MISSED"
                f.write(f"  [{tag}] rank={g['rank']} score={g['score']}  doc_id={g['doc_id']}\n")
                f.write(f"  text: {g['text']}\n")
                f.write(f"  -- full session ({g['session_id']}), in order --\n")
                for sid_doc, sid_text in zip(g['full_session_doc_ids_in_order'], g['full_session_text_in_order']):
                    marker = " <== THIS GT ROUND" if sid_doc == g['doc_id'] else ""
                    f.write(f"     [{sid_doc}] {sid_text}{marker}\n")
                f.write("\n")

            f.write(f"--- top-{len(rec['top_k_retrieved'])} retrieved (what the reader actually saw) ---\n")
            for t in rec['top_k_retrieved']:
                tag = "GT" if t['is_gt'] else "  "
                f.write(f"  [{tag}] score={t['score']:.4f}  doc_id={t['doc_id']}\n")
                f.write(f"  text: {t['text']}\n")
            f.write("\n")

            f.write(f"top-5 retrieval hypothesis (WRONG): {rec['top5_hypothesis']}\n\n")
            f.write(f"oracle hypothesis (CORRECT): {rec['oracle_hypothesis']}\n\n")
            f.write("\n")

    print(f"Wrote {args.out_txt} and {args.out_json}")


if __name__ == "__main__":
    main()
