import argparse
import json
import os
from tqdm import tqdm
import time
import numpy as np
from qrretriever.attn_retriever import FullHeadRetriever, QRRetriever, CoTQRRetriever, QRCoTRetriever


def save_checkpoint(output_file, results):
    tmp_path = output_file + ".tmp"
    with open(tmp_path, "w") as f:
        json.dump(results, f, indent=4)
    os.replace(tmp_path, output_file)


def main():

    parser = argparse.ArgumentParser(description="Run FullHeadRetriever on input file and write to output file.")
    parser.add_argument("--input_file", type=str, required=True, help="Path to the input JSON file.")
    parser.add_argument("--output_file", type=str, required=True, help="Path to the output JSON file.")

    parser.add_argument("--data_type", type=str, choices=["beir", "lme", "clipper", "oolong"], required=True, help="Type of data. Only beir changes behaviour (subsampling, doc order).")
    parser.add_argument("--truncate_by_space", type=int, default=0, help="Truncate paragraphs by number of words. Default is 0 (no truncation).")

    parser.add_argument("--retriever_type", type=str, choices=["full_head", "qr_head", "cot_qr_head", "qrcot"], default="full_head", help="Type of retriever to use. Default is 'full_head'.")
    parser.add_argument("--max_instances", type=int, default=0, help="Only run the first N data instances (for smoke tests). Default is 0 (all).")
    parser.add_argument("--subsample", type=int, default=0, help="Randomly sample N data instances with a fixed seed (for quick comparisons). Default is 0 (all).")
    parser.add_argument("--cot_instruction", type=str, default=None, help="Instruction placed before the query for cot_qr_head. Defaults to CoTQRRetriever's default.")
    parser.add_argument("--cot_file", type=str, default=None, help="Reuse CoTs from a previous run's *_cot.json instead of generating them.")
    parser.add_argument("--max_new_tokens",type=int, default=None, help="Max CoT length for cot_qr_head; 0 scores with the question only (no-CoT control). Defaults to CoTQRRetriever's default.")
    parser.add_argument("--generation_file", type=str, default=None, help="For qrcot (required): JSONL of generations over the full context, one line per question with 'idx', 'formatted_input' and 'output'.")

    parser.add_argument("--config_or_config_path", type=str, default=None, help="Path to the configuration file or a configuration string. If not provided, defaults will be used.")
    parser.add_argument("--model_name_or_path", type=str, default=None, help="Path to the model directory or model name.")

    args = parser.parse_args()
    if (args.retriever_type == "qrcot") != (args.generation_file is not None):
        parser.error("--generation_file is required for qrcot and used by nothing else")

    if args.retriever_type == "full_head":
        retriever = FullHeadRetriever(
            config_or_config_path=args.config_or_config_path,
            model_name_or_path=args.model_name_or_path,
        )
    elif args.retriever_type == "qr_head":
        retriever = QRRetriever(
            config_or_config_path=args.config_or_config_path,
            model_name_or_path=args.model_name_or_path,
        )
    elif args.retriever_type == "cot_qr_head":
        cot_kwargs = {k: v for k, v in [("cot_instruction", args.cot_instruction), ("max_new_tokens", args.max_new_tokens)] if v is not None}
        retriever = CoTQRRetriever(
            config_or_config_path=args.config_or_config_path,
            model_name_or_path=args.model_name_or_path,
            **cot_kwargs,
        )
        print(f"CoT instruction: {retriever.cot_instruction}", flush=True)
    elif args.retriever_type == "qrcot":
        retriever = QRCoTRetriever(
            config_or_config_path=args.config_or_config_path,
            model_name_or_path=args.model_name_or_path,
        )

    reuse_cots = json.load(open(args.cot_file)) if args.cot_file else {}

    generations = {} # idx -> {"formatted_input": str, "output": str, ...}, only for qrcot
    if args.generation_file:
        with open(args.generation_file, "r") as f:
            for line in f:
                entry = json.loads(line)
                generations[str(entry['idx'])] = entry
        print(f"Loaded {len(generations)} generations from {args.generation_file}", flush=True)

    # read input file
    print(f"Reading input file: {args.input_file}", flush=True)
    with open(args.input_file, "r") as f:
        data_instances = json.load(f)

    #####################################################################
    # Subsample to 512 data instances for BEIR
    #####################################################################
    if args.data_type == "beir":
        if len(data_instances) > 512:
            np.random.seed(42)
            indices = np.random.choice(len(data_instances), 512, replace=False)
            data_instances = [data_instances[i] for i in indices]

    # LME is sorted by question type, so use --subsample rather than --max_instances for representative subsets
    if args.subsample > 0 and len(data_instances) > args.subsample:
        np.random.seed(42)
        indices = np.random.choice(len(data_instances), args.subsample, replace=False)
        data_instances = [data_instances[i] for i in indices]

    if args.max_instances > 0:
        data_instances = data_instances[:args.max_instances]

    results = {}
    cots = {} # idx -> {"text": str, "ids": List[int]}, only for cot_qr_head
    cot_file = os.path.splitext(args.output_file)[0] + "_cot.json"
    latency_list = []

    # Resume from a previous (possibly OOM-killed) run, if a partial output exists.
    if os.path.exists(args.output_file):
        with open(args.output_file, "r") as f:
            results = json.load(f)
        print(f"Resuming from checkpoint: {len(results)} items already done.", flush=True)
    if args.retriever_type == "cot_qr_head" and os.path.exists(cot_file):
        with open(cot_file, "r") as f:
            cots = json.load(f)

    for i, data in enumerate(tqdm(data_instances)):

        if str(data['idx']) in results:
            continue

        query = data["question"]
        docs = data["paragraphs"]
        
        for p in docs:

            paragraph_text = p['paragraph_text'].strip()

            if args.truncate_by_space > 0:
                # Truncate each paragraph by space.
                if len(paragraph_text.split(' ')) > args.truncate_by_space:
                    print('number of words being truncated: ', len(paragraph_text.split(' ')) - args.truncate_by_space, flush=True)

                p['paragraph_text'] = ' '.join(paragraph_text.split(' ')[:args.truncate_by_space])

            else:
                p['paragraph_text'] = paragraph_text

        if args.data_type == "beir":
            # Reverse the order of docs for BEIR, to match the document order used in ICR baseline.
            docs = docs[::-1]

        start_time = time.time()
        if args.retriever_type == "cot_qr_head":
            retrieval_scores, cot_ids = retriever.score_docs(query, docs, cot_ids=reuse_cots[str(data['idx'])]['ids'] if reuse_cots else None, return_cot=True) # doc_id -> score
            cots[data['idx']] = {"text": retriever.tokenizer.decode(cot_ids), "ids": cot_ids}
            print(f"Query {i} CoT ({len(cot_ids)} tokens): {cots[data['idx']]['text']}", flush=True)
        elif args.retriever_type == "qrcot":
            generation = generations[str(data['idx'])]
            retrieval_scores = retriever.score_docs(generation['formatted_input'], generation['output'], docs) # doc_id -> score
        else:
            retrieval_scores = retriever.score_docs(query, docs) # doc_id -> score
        end_time = time.time()
        print(f"Query {i} processed in {(end_time - start_time) * 1000} ms", flush=True)
        latency_list.append((end_time - start_time) * 1000)  # Convert to milliseconds

        # save the CoT first: results is what resuming keys on
        if cots:
            save_checkpoint(cot_file, cots)
        results[data['idx']] = retrieval_scores
        save_checkpoint(args.output_file, results)

    print(f"Average Latency: {np.mean(latency_list)} ms", flush=True)


if __name__ == "__main__":
    main()