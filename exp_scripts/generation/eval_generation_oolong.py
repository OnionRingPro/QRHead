"""
Score oolong-synth answers from run_generation_oolong.py with the oolong-synth rules, as the QR-CoT repo does
(eval_datasets/oolong.py): exact match on the parsed answer, 0.75^|gold - answer| for numbers, equality for dates.
Runs in the vLLM venv (slurm/vllm_env.sh), which has python-dateutil; .venv does not.

    python exp_scripts/generation/eval_generation_oolong.py --data_file <prepare_oolong.py json> --runs <run dir or .jsonl> ...

A run dir stands for every full.jsonl / random_<k>.jsonl / retrieval_<k>.jsonl in it. Prints one row per generation
file, scores x100 averaged over the questions it has:
    score     the QR-CoT repo's: parse the text after the last </think>, or the whole output if the thinking never
              finished (so a truncated reasoning can still earn partial credit for a number in it)
    strict    the same, but 0 for an output whose thinking never finished
    counting / user / timeline   score by task_group
    subset / whole               score on the questions with / without a line filter (input_subset True / False)
    trunc     % of outputs that hit max_tokens;  low_conf  % whose answer was parsed with low confidence
and writes each file's per-question scores next to it as scores_<name>.jsonl.
"""
import argparse
import ast
import json
import os
import re
from datetime import datetime
from typing import Dict, List, Tuple

import dateutil.parser

GENERATION_FILE_RE = re.compile(r"^(full|(random|retrieval)_(\d+))\.jsonl$")


# _synth_attempt_answer_parse and the scoring in eval_one are the QR-CoT repo's eval_datasets/oolong.py, unchanged
def _synth_attempt_answer_parse(answer: str):
    parse_confidence = "low"
    if ":" not in answer:
        if len(answer) < 20:
            return answer, parse_confidence
        else:
            return answer.split()[-1], parse_confidence
    candidate_answer = answer.split(":")[-1].strip()
    candidate_answer = candidate_answer.replace("*", "")
    candidate_answer = candidate_answer.replace("[", "")
    candidate_answer = candidate_answer.replace("]", "")
    parse_confidence = "med"
    if any(kw in answer for kw in ("User:", "Answer:", "Date:", "Label")):
        parse_confidence = "high"
    if len(candidate_answer) < 20:
        parse_confidence = "vhigh"
    elif "more common" in candidate_answer:
        candidate_answer = "more common"
    elif "less common" in candidate_answer:
        candidate_answer = "less common"
    elif "same frequency" in candidate_answer:
        candidate_answer = "same frequency"
    return candidate_answer, parse_confidence


def eval_one(response: str, datapoint: Dict) -> Tuple[float, str, str]:
    """Returns (score, parsed answer, parse confidence) for one raw model output."""
    if "</think>" in response:
        response = response.rsplit("</think>", 1)[1].strip()

    answer_str = datapoint["answer"]
    if "datetime" in answer_str:
        gold = datetime.strptime(answer_str, "[datetime.date(%Y, %m, %d)]")
    else:
        gold = ast.literal_eval(answer_str)[0]

    trimmed_output, parse_confidence = _synth_attempt_answer_parse(response)

    score = 0.0
    if str(trimmed_output) == str(gold):
        score = 1.0
    elif str(trimmed_output) in ("more common", "less common", "same frequency"):
        if str(trimmed_output) in str(gold):
            score = 1.0
    elif datapoint["answer_type"] == "ANSWER_TYPE.NUMERIC":
        try:
            score = 0.75 ** abs(int(gold) - int(trimmed_output))
        except Exception:
            parse_confidence = "low"
    elif datapoint["answer_type"] == "ANSWER_TYPE.DATE":
        try:
            score = float(dateutil.parser.parse(trimmed_output) == gold)
        except Exception:
            parse_confidence = "low"
    return score, str(trimmed_output), parse_confidence


def generation_files(run: str) -> List[str]:
    """run is a generation file, or a run dir whose generation files come full first, then by method and k."""
    if not os.path.isdir(run):
        return [run]
    matches = [GENERATION_FILE_RE.match(name) for name in os.listdir(run)]
    matches = sorted((m for m in matches if m), key=lambda m: (m.group(2) or '', int(m.group(3) or 0)))
    return [os.path.join(run, m.group(0)) for m in matches]


def score_file(path: str, data: Dict[str, Dict]) -> List[Dict]:
    rows = []
    with open(path) as f:
        for line in f:
            gen = json.loads(line)
            datapoint = data[gen['idx']]
            score, parsed, confidence = eval_one(gen['output'], datapoint)
            unfinished = "</think>" not in gen['output'] and gen['finish_reason'] == 'length'
            rows.append({
                "idx": gen['idx'],
                "score": score,
                "strict": 0.0 if unfinished else score,
                "parsed": parsed,
                "confidence": confidence,
                "gold": datapoint['answer'],
                "finish_reason": gen['finish_reason'],
                "num_output_tokens": gen['num_output_tokens'],
                "task_group": datapoint['task_group'],
                "task": datapoint['task'],
                "subset": datapoint['filter_type'] is not None,
            })
    out_path = os.path.join(os.path.dirname(path), 'scores_' + os.path.basename(path))
    with open(out_path, 'w') as f:
        for row in rows:
            f.write(json.dumps(row) + '\n')
    return rows


def mean(values: List[float]) -> str:
    return f"{100 * sum(values) / len(values):.1f}" if values else "-"


def main():
    parser = argparse.ArgumentParser(description="Score oolong-synth generations (oolong-synth rules, as in the QR-CoT repo).")
    parser.add_argument('--data_file', type=str, required=True, help='The prepare_oolong.py file the generations were made from (has answer, answer_type, task_group).')
    parser.add_argument('--runs', type=str, nargs='+', required=True, help='Generation .jsonl files and/or run dirs.')
    args = parser.parse_args()

    with open(args.data_file) as f:
        data = {sample['idx']: sample for sample in json.load(f)}

    columns = ["n", "score", "strict", "counting", "user", "timeline", "subset", "whole", "trunc", "low_conf"]
    print(f"{'file':<60}" + "".join(f"{c:>9}" for c in columns))
    for run in args.runs:
        for path in generation_files(run):
            rows = score_file(path, data)
            by = lambda key, value: [r['score'] for r in rows if r[key] == value]
            values = [
                str(len(rows)),
                mean([r['score'] for r in rows]),
                mean([r['strict'] for r in rows]),
                mean(by('task_group', 'counting')),
                mean(by('task_group', 'user')),
                mean(by('task_group', 'timeline')),
                mean(by('subset', True)),
                mean(by('subset', False)),
                mean([float(r['finish_reason'] == 'length') for r in rows]),
                mean([float(r['confidence'] == 'low') for r in rows]),
            ]
            name = os.path.join(os.path.basename(os.path.dirname(os.path.abspath(path))), os.path.basename(path))
            print(f"{name:<60}" + "".join(f"{v:>9}" for v in values))


if __name__ == '__main__':
    main()
