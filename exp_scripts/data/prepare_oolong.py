"""Convert oolong-synth into the QRHead retrieval format, one doc per context line.

Only questions with input_subset=True are kept: their filter (user IDs / month / date range)
selects a subset of the context lines, and that subset becomes gt_docs.

Context line formats:
    context_window_text:             Date: Dec 28, 2022 || User: 76063 || Instance: <text>
    context_window_text_with_labels: Date: Dec 28, 2022 || User: 76063 || Instance: <text> || Label: spam
"""
import argparse
import glob
import json
import os
import re
from datetime import datetime
from typing import Callable, Dict, List, Optional, Tuple

import pyarrow.parquet as pq

COLUMNS = ["id", "context_len", "dataset", "context_window_id", "context_window_text", "context_window_text_with_labels",
           "question", "task_group", "task", "answer", "answer_type", "input_subset"]
LINE_RE = re.compile(r"^Date: (?P<date>[A-Z][a-z]{2} \d{2}, \d{4}) \|\| User: (?P<user>\d+) \|\| Instance: ")
DATE_FMT = "%b %d, %Y"

# subset filters: the question's first sentence, "For the following question, only consider the subset of instances that ..."
USER_FILTER_RE = re.compile(r"associated with user IDs ([^.]+)\.")
MONTH_FILTER_RE = re.compile(r"occur in ([A-Z][a-z]+) of any year\.")
RANGE_FILTER_RE = re.compile(r"occur between ([A-Z][a-z]{2} \d{2}, \d{4}) and ([A-Z][a-z]{2} \d{2}, \d{4}), inclusive\.")

def split_context_lines(text: str) -> List[str]:
    """Keep only the data lines, dropping the instruction preamble and the 'Recall:' postamble."""
    return [l.rstrip() for l in text.split("\n") if LINE_RE.match(l)]


def parse_line(unlabeled: str, labeled: str) -> Dict:
    """Pair an unlabeled line with its labeled twin and pull out the metadata."""
    body, label = labeled.rsplit(" || Label: ", 1)
    assert body.rstrip() == unlabeled, f"line mismatch:\n{unlabeled}\n{labeled}"
    m = LINE_RE.match(unlabeled)
    return {
        "text": unlabeled, # what QRHead sees: no label
        "date": datetime.strptime(m.group("date"), DATE_FMT),
        "user": m.group("user"),
        "label": label.strip(),
    }


def parse_subset_filter(question: str) -> Optional[Tuple[str, Callable[[Dict], bool]]]:
    """Map a subset question to (filter_type, predicate over a parsed line) that picks its gold lines.

    The question opens with one of three templates:
        "...instances that are associated with user IDs 36861."
        "...instances that occur in August of any year."
        "...instances that occur between Aug 15, 2022 and Mar 26, 2024, inclusive."
    filter_type is one of "user", "month", "date_range". Return None if no template matches.
    """
    m = USER_FILTER_RE.search(question)
    if m:
        users = set(re.findall(r"\d+", m.group(1))) # "user IDs" is plural, so allow a list
        return "user", lambda line: line["user"] in users

    m = MONTH_FILTER_RE.search(question)
    if m:
        month = datetime.strptime(m.group(1), "%B").month # "August" -> 8, any year
        return "month", lambda line: line["date"].month == month

    m = RANGE_FILTER_RE.search(question)
    if m:
        start, end = (datetime.strptime(d, DATE_FMT) for d in m.groups())
        return "date_range", lambda line: start <= line["date"] <= end

    return None


def convert_row(row: Dict, filter_type: str, keep: Callable[[Dict], bool]) -> Dict:
    unlabeled = split_context_lines(row["context_window_text"])
    labeled = split_context_lines(row["context_window_text_with_labels"])
    assert len(unlabeled) == len(labeled), f"row {row['id']}: {len(unlabeled)} vs {len(labeled)} lines"
    # every instance carries exactly one label, so this catches instances that span several lines
    assert row["context_window_text_with_labels"].count(" || Label: ") == len(labeled), f"row {row['id']}: multi-line instances"
    lines = [parse_line(u, l) for u, l in zip(unlabeled, labeled)]

    qid = str(row["id"])
    paragraphs = [{"idx": f"{qid}__{i}", "paragraph_text": line["text"]} for i, line in enumerate(lines)]
    gt_docs = [p["idx"] for p, line in zip(paragraphs, lines) if keep(line)]

    return {
        "idx": qid,
        "question": row["question"],
        "paragraphs": paragraphs,
        "gt_docs": gt_docs,
        "num_gold_docs": len(gt_docs),
        # oolong metadata, for per-type breakdowns and answer scoring
        "filter_type": filter_type,
        "task_group": row["task_group"],
        "task": row["task"],
        "answer": row["answer"],
        "answer_type": row["answer_type"],
        "dataset": row["dataset"],
        "context_len": row["context_len"],
        "context_window_id": row["context_window_id"],
    }


def main():

    parser = argparse.ArgumentParser(description="Convert oolong-synth parquet files into QRHead retrieval JSON.")
    parser.add_argument("--raw_dir", type=str, required=True, help="Directory with the downloaded oolong-synth data/*.parquet files.")
    parser.add_argument("--output_file", type=str, required=True, help="Path to the output JSON file.")
    parser.add_argument("--split", type=str, choices=["validation", "test"], default="validation")
    parser.add_argument("--max_context_len", type=int, default=131072, help="Drop context windows whose context_len bucket is above this. Measured on validation, the QRHead prompt is 0.55-0.92x context_len Llama-3.1 tokens (~121K for the 131072 bucket), so 131072 is the largest bucket that fits Llama-3.1's 128K window.")
    args = parser.parse_args()

    files = sorted(glob.glob(os.path.join(args.raw_dir, f"{args.split}-*.parquet")))
    print(f"Found {len(files)} parquet files for split {args.split}", flush=True)

    data_instances = []
    stats = {"rows": 0, "not_subset": 0, "too_long": 0, "unparsed": 0, "empty_gold": 0}
    for path in files:
        # stream in small batches: some shards are hundreds of MB of long contexts
        for batch in pq.ParquetFile(path).iter_batches(columns=COLUMNS, batch_size=32):
            for row in batch.to_pylist():
                stats["rows"] += 1
                if str(row["input_subset"]) != "True":
                    stats["not_subset"] += 1 # whole-context question, no gold subset
                    continue
                if row["context_len"] > args.max_context_len:
                    stats["too_long"] += 1
                    continue
                parsed = parse_subset_filter(row["question"])
                if parsed is None:
                    # a subset question always has a filter, so this is a template we don't know yet
                    stats["unparsed"] += 1
                    print(f"Unparsed filter in row {row['id']}: {row['question'][:200]}", flush=True)
                    continue
                instance = convert_row(row, *parsed)
                if instance["num_gold_docs"] == 0:
                    stats["empty_gold"] += 1
                    continue
                data_instances.append(instance)

    print(f"Stats: {stats}, kept {len(data_instances)}", flush=True)
    os.makedirs(os.path.dirname(os.path.abspath(args.output_file)), exist_ok=True)
    with open(args.output_file, "w") as f:
        json.dump(data_instances, f, indent=4)


if __name__ == "__main__":
    main()
