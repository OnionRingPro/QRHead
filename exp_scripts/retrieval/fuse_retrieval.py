import argparse
import json


def rrf(doc_scores, k=60):
    """Reciprocal rank fusion term for each doc: 1 / (k + rank)."""
    ranked = sorted(doc_scores, key=doc_scores.get, reverse=True)
    return {doc: 1 / (k + rank) for rank, doc in enumerate(ranked, 1)}


def main():

    parser = argparse.ArgumentParser(description='Fuse retrieval result files with reciprocal rank fusion, on the queries they all cover')
    parser.add_argument('--result_files', nargs='+', required=True)
    parser.add_argument('--output_file', required=True)

    args = parser.parse_args()

    runs = [json.load(open(path)) for path in args.result_files]
    fused = {}
    for qid in set.intersection(*(set(run) for run in runs)):
        per_run = [rrf(run[qid]) for run in runs]
        fused[qid] = {doc: sum(r[doc] for r in per_run) for doc in per_run[0]}

    with open(args.output_file, 'w') as f:
        json.dump(fused, f)


if __name__ == "__main__":
    main()
