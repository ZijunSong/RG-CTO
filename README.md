# RG-CTO: Risk-Gated Contrastive Trajectory Optimization

Self-contained implementation of **CTO** (Contrastive Trajectory Optimization), **RSE** baseline, and **RG-CTO** (confidence-gated negative control). Migrated from `Contrastive-Trajectory-Optimization` and `RSE`.

## Project Structure

```text
RG-CTO/
├── code/
│   ├── standard_sampling.py          # iter0 baseline sampling
│   ├── experience_distillation.py    # proposition / pitfall extraction
│   ├── experience_dedup.py         # experience bank dedup
│   ├── experience_guided_search.py   # RSE guided search
│   ├── cto_guided_search.py          # vanilla CTO contrastive decoding
│   ├── rg_cto_common.py              # RG-CTO gate (support / locality / conflict)
│   └── rg_cto_guided_search.py       # RG-CTO guided search
├── scripts/
│   ├── run_rse.sh                    # RSE pipeline
│   ├── run_cto.sh                    # CTO pipeline
│   └── run_rg_cto.sh                 # RG-CTO pipeline
├── data/                             # HMMT24/25, HLE-Math-text, BambooQA, CodeContests, TravelPlanner_Val60
├── eval/                             # pass@k evaluation
└── results/iter0/                  # copied best iter0 rollouts per setting
```

## Iteration Protocol

| Step | Iteration | Method step |
|------|-----------|-------------|
| step1 | iter0 | Standard sampling (32 rollouts) |
| step3 | iter1 | RSE / CTO / RG-CTO guided search |
| step5 | iter2 | guided search |
| step7 | iter3 | guided search |

## Installation

```bash
conda create -n rg-cto python=3.10 -y
conda activate rg-cto
pip install -r requirements.txt
```

## Usage

Set environment variables, then run one pipeline:

```bash
export MODEL_NAME=/path/to/Qwen3-4B-Thinking-2507
export QUESTION_FILE=data/HMMT_24.jsonl
export OUT_PREFIX=runs/HMMT24_Qwen3_4B_RG_CTO
export EMB_MODEL=/path/to/all-MiniLM-L6-v2

# RSE baseline
bash scripts/run_rse.sh 0 100

# Vanilla CTO
bash scripts/run_cto.sh 0 100

# RG-CTO (risk-gated)
bash scripts/run_rg_cto.sh 0 100
```

### RG-CTO hyperparameters

`run_rse.sh`, `run_cto.sh`, and `run_rg_cto.sh` use the same defaults as the local Qwen3 runs. Calling the Python entry points without those flags falls back to the same values. The Phi-4 and 30B launchers override length, GPU memory, and a few hyperparameters themselves.

| Flag | Default | Meaning |
|------|---------|---------|
| `--temperature` | 0.6 | Sampling temperature |
| `--n-completions` | 32 | Rollouts per question |
| `--max-tokens` | 38912 | Generation length; distillation uses 8192 |
| `--alpha` | 0.7 (RG-CTO), 0.55 (CTO) | Base suppression α₀ |
| `--plausibility-top-k` | 8 | CTO candidate budget |
| `--gate-delta` | 0.4 | Filter pitfalls with w(e) < δ |
| `--tau-match` | 0.8 | Support match threshold |
| `--n-experience-completions` | 48 (CTO / RG-CTO), 32 (RSE) | Experience records merged per question |
| `--threshold` | 0.85 | Experience dedup cosine threshold |
| `--min-pitfall-support` | 2 | Minimum rollout support u(e) |
| `--pilot-n` | 4 | Retained flag; ignored by the current gate |

## Reusing iter0 Results

Pre-copied iter0 rollouts live under `results/iter0/`. To bootstrap a new run without re-sampling:

```bash
cp -a results/iter0/HMMT24/Qwen3_4B_Thinking_2507/results "${OUT_PREFIX}_step1/results"
```

Summary metrics: `results/summaries/iter0_pass_at1.json`.

## TravelPlanner (sole-planning)

`data/TravelPlanner_Val60.jsonl` is a 60-query stratified slice of the official validation split (7 easy and medium queries per day length, 6 hard). The filename selects the `agent` task, so the existing RSE / CTO / RG-CTO scripts pick up the planner prompts. Each record keeps the reference tables, and pass@1 is the official final-pass rule: every commonsense constraint and every applicable hard constraint holds. The same script also reports delivery rate plus commonsense and hard micro rates.

```bash
python scripts/prepare_travelplanner_val60.py   # rebuild the jsonl from validation.csv
export QUESTION_FILE=data/TravelPlanner_Val60.jsonl
export OUT_PREFIX=runs/TravelPlanner_Val60_RG_CTO
bash scripts/run_rg_cto.sh 0 60
```

## Method

RG-CTO applies per-pitfall confidence gating before contrastive decoding:

\[
w(e) = \mathrm{clip}\bigl[u(e)\cdot l(e,q)\cdot(1-c(e)),\,0,\,1\bigr],\quad
\alpha_r = \alpha_0 \cdot g^{(r)},\quad
g^{(r)} = \frac{1}{|\mathcal{R}_-|}\sum_{e} w(e)
\]

See `RG-CTO Method Core.md` for the full paper outline.
