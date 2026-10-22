import argparse
import json
import re

from rank_bm25 import BM25Okapi


def tokenize(text):
    """Lowercase and split on word characters."""
    return re.findall(r'\w+', text.lower())


def main():

    parser = argparse.ArgumentParser(description="BM25 retrieval over each query's own paragraphs, optionally expanding the query with a CoT.")
    parser.add_argument("--input_file", type=str, required=True, help="Path to the input JSON file (same format as run_retrieval.py).")
    parser.add_argument("--output_file", type=str, required=True, help="Path to the output JSON file.")
    parser.add_argument("--mode", type=str, choices=["q", "cot", "q+cot"], default="q", help="What to use as the query.")
    parser.add_argument("--cot_file", type=str, default=None, help="*_cot.json from a cot_qr_head run; needed for the cot and q+cot modes.")
    parser.add_argument("--q_repeat", type=int, default=5, help="Times the question is repeated in q+cot mode (as in Query2doc). Default is 5.")

    args = parser.parse_args()
    if args.mode != "q" and args.cot_file is None:
        parser.error("--cot_file is required for the cot and q+cot modes")

    with open(args.input_file, "r") as f:
        data_instances = json.load(f)
    cots = {}
    if args.cot_file is not None:
        with open(args.cot_file, "r") as f:
            cots = json.load(f)

    results = {} # qid -> {doc_id -> score}, same format as run_retrieval.py
    for data in data_instances:
        qid = str(data['idx'])
        question = data['question']
        docs = data['paragraphs'] # each doc: {'idx': doc_id, 'paragraph_text': str}
        cot = cots[qid]['text'] if args.mode != "q" else ""

        # one index per query over its own paragraphs, so IDF is computed within this haystack
        bm25 = BM25Okapi([tokenize(doc['paragraph_text']) for doc in docs])

        # repeating the question gives its terms more weight next to the longer CoT (get_scores sums over query tokens)
        query = {"q": question, "cot": cot, "q+cot": " ".join([question] * args.q_repeat + [cot])}[args.mode]

        scores = bm25.get_scores(tokenize(query)) # aligned with docs
        results[qid] = {doc['idx']: float(score) for doc, score in zip(docs, scores)}

    with open(args.output_file, "w") as f:
        json.dump(results, f)


if __name__ == "__main__":
    main()
