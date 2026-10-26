"""
Answer oolong-synth questions with vLLM over the full context or over k of its lines; adapted from the QR-CoT repo's
exp_scripts_qrcot/run_generation_oolong.py. Runs in the vLLM venv (slurm/vllm_env.sh), not in .venv.

    --method full       every line. Its outputs are what QRCoTRetriever scores (run_retrieval.py --retriever_type qrcot)
    --method random     k random lines per question (random.Random(seed) restarted for every k, as in the QR-CoT repo)
    --method retrieval  the k top-scored lines of --retrieval_file (a run_retrieval.py retrieval.json)
The kept lines stay in context order, and the line count, stated twice in both the prefix and the suffix, becomes k
(other numbers there, like yahoo's "one of 10 categories", stay; prepare_oolong.py checks the count appears exactly
so). The prompt is prefix + "\n\n" + lines + "\n" + suffix + "\n" + question, the QR-CoT repo's layout; with every
line kept it is context_window_text + "\n" + question.

Writes one JSONL per k into --output_dir (full.jsonl, random_<k>.jsonl, retrieval_<k>.jsonl), a line per question:
idx, method, k, num_lines, formatted_input (the chat-template-applied prompt: formatted_input + output is exactly
what the model saw), output, answer (the text after </think>; "" if the thinking never finished), finish_reason,
num_prompt_tokens, num_output_tokens. Questions already in an output file are skipped, so a timed-out job can be
resubmitted into the same directory.
"""
import argparse
import json
import os
import random
import re
import time
from collections import Counter
from typing import Dict, List, Optional, Set

import torch
from vllm import LLM, SamplingParams


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', type=str, required=True, help='prepare_oolong.py output (needs prefix/suffix, i.e. made on or after 2026-10-25).')
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--method', type=str, required=True, choices=['full', 'random', 'retrieval'])
    parser.add_argument('--topk', type=int, nargs='+', default=None, help='Numbers of lines to keep, for random and retrieval.')
    parser.add_argument('--retrieval_file', type=str, default=None, help='For retrieval: idx -> {line idx: score}, from run_retrieval.py.')
    parser.add_argument('--model', type=str, default='Qwen/Qwen3-4B-Thinking-2507', help='HF id (read from $HF_HOME) or path.')
    parser.add_argument('--max_model_len', type=int, default=262144)
    parser.add_argument('--max_tokens', type=int, default=16384, help='Generation budget per question (QR-CoT repo: 16384 for oolong).')
    parser.add_argument('--tensor_parallel_size', type=int, default=None, help='Default: every visible GPU.')
    parser.add_argument('--max_instances', type=int, default=0, help='Only the first N questions (pilots). Default is 0 (all).')
    parser.add_argument('--save_every', type=int, default=32, help='Questions per generate call; each call is appended to the output before the next.')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    if (args.method == 'full') != (args.topk is None):
        parser.error('--topk is required for random and retrieval, and not used by full')
    if (args.method == 'retrieval') != (args.retrieval_file is not None):
        parser.error('--retrieval_file is required for retrieval, and not used by the other methods')
    return args


def build_prompt(sample: Dict, keep: Optional[Set[str]]) -> str:
    """oolong's prompt over the lines in keep (all of them if None), in context order."""
    paragraphs = sample['paragraphs'] if keep is None else [p for p in sample['paragraphs'] if p['idx'] in keep]
    line_count = re.compile(rf"(?<!\d){len(sample['paragraphs'])}(?!\d)") # the count as a whole number only
    prefix = line_count.sub(str(len(paragraphs)), sample['prefix'])
    suffix = line_count.sub(str(len(paragraphs)), sample['suffix'])
    docs_text = "\n".join(p['paragraph_text'] for p in paragraphs)
    return f"{prefix}\n\n{docs_text}\n{suffix}" + "\n" + sample['question']


def select_lines(data: List[Dict], method: str, k: Optional[int], doc_scores: Dict, seed: int) -> List[Optional[Set[str]]]:
    """The lines to keep for every question, in data order (None = all)."""
    if method == 'full':
        return [None] * len(data)
    if method == 'random':
        rng = random.Random(seed) # restarted per k, and drawn for every question in order, so resuming draws the same lines
        return [set(rng.sample([p['idx'] for p in sample['paragraphs']], min(k, len(sample['paragraphs'])))) for sample in data]
    keep = []
    for sample in data:
        scores = doc_scores[sample['idx']]
        keep.append(set(sorted(scores, key=scores.get, reverse=True)[:k]))
    return keep


def load_done(path: str) -> Set[str]:
    """idx of the questions already written; drops a line cut off by a killed job."""
    if not os.path.exists(path):
        return set()
    entries = []
    with open(path) as f:
        for line in f:
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                print(f"Dropping a broken line in {path}", flush=True)
    with open(path, 'w') as f:
        for entry in entries:
            f.write(json.dumps(entry) + '\n')
    return {entry['idx'] for entry in entries}


def split_answer(output: str, formatted_input: str) -> str:
    """The reply after the reasoning; "" if the reasoning never finished (the generation hit max_tokens)."""
    if "</think>" in output:
        return output.split("</think>", 1)[1].strip()
    # the Thinking template opens the reply with <think> itself, so the output only has the closing tag
    if "<think>" in output or formatted_input.rstrip().endswith("<think>"):
        return ""
    return output


def main():
    args = parse_args()
    print(f"args: {vars(args)}", flush=True)

    with open(args.input_file) as f:
        data = json.load(f)
    if args.max_instances > 0:
        data = data[:args.max_instances]
    assert 'prefix' in data[0], f"{args.input_file} has no prefix/suffix; rerun prepare_oolong.py"

    doc_scores = {}
    if args.retrieval_file:
        with open(args.retrieval_file) as f:
            doc_scores = json.load(f)
        missing = [sample['idx'] for sample in data if sample['idx'] not in doc_scores]
        assert not missing, f"{len(missing)} questions have no scores in {args.retrieval_file}, e.g. {missing[:3]}"

    runs = [(None, os.path.join(args.output_dir, 'full.jsonl'))] if args.method == 'full' else \
           [(k, os.path.join(args.output_dir, f'{args.method}_{k}.jsonl')) for k in args.topk]
    todo = []
    for k, path in runs:
        done = load_done(path)
        print(f"{os.path.basename(path)}: {len(done)}/{len(data)} questions already done", flush=True)
        if len(done) < len(data):
            todo.append((k, path, done))
    if not todo:
        return

    llm = LLM(
        model=args.model,
        max_model_len=args.max_model_len,
        tensor_parallel_size=args.tensor_parallel_size or torch.cuda.device_count(),
        seed=args.seed,
    )
    tokenizer = llm.get_tokenizer()
    # Qwen3 Thinking's recommended sampling (as in the QR-CoT repo); the per-request seed makes each question's
    # sample independent of how requests are batched
    sampling_params = SamplingParams(
        temperature=0.6,
        top_p=0.95,
        top_k=20,
        min_p=0.0,
        max_tokens=args.max_tokens,
        seed=args.seed,
        stop_token_ids=[tokenizer.eos_token_id],
    )

    for k, path, done in todo:
        keep = select_lines(data, args.method, k, doc_scores, args.seed)
        pending = [(sample, kept) for sample, kept in zip(data, keep) if sample['idx'] not in done]
        finish_reasons = Counter()
        for c0 in range(0, len(pending), args.save_every):
            chunk = pending[c0:c0 + args.save_every]
            formatted_inputs = [
                tokenizer.apply_chat_template([{"role": "user", "content": build_prompt(sample, kept)}], tokenize=False, add_generation_prompt=True)
                for sample, kept in chunk
            ]
            start = time.time()
            outputs = llm.generate(formatted_inputs, sampling_params)
            with open(path, 'a') as f:
                for (sample, kept), formatted_input, out in zip(chunk, formatted_inputs, outputs):
                    completion = out.outputs[0]
                    finish_reasons[completion.finish_reason] += 1
                    f.write(json.dumps({
                        "idx": sample['idx'],
                        "method": args.method,
                        "k": k,
                        "num_lines": len(sample['paragraphs']) if kept is None else len(kept),
                        "formatted_input": formatted_input,
                        "output": completion.text,
                        "answer": split_answer(completion.text, formatted_input),
                        "finish_reason": completion.finish_reason,
                        "num_prompt_tokens": len(out.prompt_token_ids),
                        "num_output_tokens": len(completion.token_ids),
                    }) + '\n')
            print(f"{os.path.basename(path)}: {len(done) + c0 + len(chunk)}/{len(data)} done, last {len(chunk)} in {time.time() - start:.0f}s", flush=True)
        print(f"{os.path.basename(path)} finish reasons this run: {dict(finish_reasons)}", flush=True)


if __name__ == '__main__':
    main()
