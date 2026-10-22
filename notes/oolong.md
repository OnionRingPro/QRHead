# OOLONG 实验笔记

## 数据

- oolong-synth validation，`exp_scripts/data/prepare_oolong.py` → `$SCRATCH/qrhead_data/data/oolong_data/synth_validation.json`
- 1,300 题 → 只留 `input_subset=True`（剩 417）→ 只留 `context_len ≤ 131072`（剩 250），来自 31 个上下文
- 每行一个 doc，gold = 满足过滤条件的行；user 126 / month 75 / date_range 49
- 指标：Recall@|gold|（k = 该题 gold 数），附随机基线和 AUROC（`exp_scripts/retrieval/eval_retrieval_oolong.py`）

## 结果：Llama-3.1-8B-Instruct + LME QR heads（2026-10-21）

Recall@|gold|，括号内为 AUROC。

| run | ALL | user | month | date_range |
|---|---|---|---|---|
| 随机 | 0.149 | 0.067 | 0.088 | 0.453 |
| QRHead (q) | 0.549 (.859) | 0.552 (.928) | 0.582 (.911) | 0.492 (.602) |
| CoT 控制组（`max_new_tokens 0`） | 0.451 (.830) | 0.445 (.908) | 0.448 (.875) | 0.471 (.561) |
| CoT+q | 0.472 (.849) | 0.393 (.910) | 0.559 (.887) | 0.540 (.633) |
| BM25 q | 0.195 (.665) | 0.112 (.772) | 0.149 (.551) | 0.477 (.565) |
| BM25 cot | 0.442 (.776) | 0.406 (.823) | 0.457 (.804) | 0.511 (.611) |
| BM25 q+cot | 0.354 (.764) | 0.317 (.841) | 0.322 (.738) | 0.500 (.609) |

文件：`output/oolong_cot/`（qr、control、cot 及其 `_cot.json`）、`output/oolong_bm25/`、汇总 `output/oolong_results/oolong-eval_1886613.out`。
`output/oolong_results/qrhead_qrhead_1885447.out` 是写错参数（`--max_instances 0`）的旧控制组日志，不是正确结果。

## 结论

1. 换成 CoT 的 prompt 本身就掉约 10 个点（QRHead q → 控制组：−9.8，置信区间 [−11.5, −8.1]），比 CoT 的作用还大，原因未查明。
2. CoT 的作用取决于题型（CoT+q − 控制组）：month +11、date_range +5 到 +9，都显著；user 在 CoT 被截断时 −11.5（显著），未截断时约 0。
3. 一半的 CoT 达到 512 token 上限。user 题掉分的原因是 CoT 抄错了行和用户 ID：CoT 引用的行里 gold 不到一半时 −15.7，一半以上时 +3.7。
4. month 的 CoT 引用准确率只有 0.31 也照样加分，猜测是 CoT 把 "August" 写成了数据行里的 "Aug"（未验证）。
5. 只有 date_range 上 CoT+q 超过了原版 QRHead（+3 到 +6），但它本身只比随机高约 9 个点。

截断分析：`output/oolong_results/cot_truncation_eval.txt`。

## 下一步

- 换 Qwen3-4B-Instruct-2507 / Thinking-2507，用导师给的 NQ heads（`src/qrretriever/configs/Qwen3-4B-*_qr_head_NQ.yaml`）：
  需要 transformers ≥ 4.51（现在是 4.48.3）、带 QK-norm 的 `custom_modeling_qwen3.py`、在 `MODEL_SPECS` 里加一项
- 有了 256K 窗口以后把 `MAX_CONTEXT_LEN` 放宽到 262144，先实测 prompt 长度
- 查清 prompt 改动为什么掉 10 个点
