#!/usr/bin/env python3

import json
import argparse
import os
import re
import sys
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
from tqdm import tqdm
import logging
from collections import defaultdict

# evaluation / sal 位于仓库的 eval/ 下，从项目根执行 python code/... 时需加入路径
_eval_root = Path(__file__).resolve().parent.parent / "eval"
if str(_eval_root) not in sys.path:
    sys.path.insert(0, str(_eval_root))

from evaluation.grader import math_equal
from sal.utils.math import extract_answer

# 兼容 vLLM 0.8.x 与新版 transformers：新版 transformers 移除了 all_special_tokens_extended，
# 但 vLLM 的 get_cached_tokenizer 仍会访问该属性（如 Qwen2Tokenizer）。在导入 vLLM 前为基类补上该属性。
def _patch_transformers_for_vllm():
    import transformers.tokenization_utils_base as _tokenizer_base
    if not hasattr(_tokenizer_base.PreTrainedTokenizerBase, "all_special_tokens_extended"):
        @property
        def all_special_tokens_extended(self):
            return self.all_special_tokens
        _tokenizer_base.PreTrainedTokenizerBase.all_special_tokens_extended = all_special_tokens_extended


_patch_transformers_for_vllm()

try:
    from vllm import LLM, SamplingParams
except ImportError:
    LLM = None
    SamplingParams = None

try:
    from transformers import AutoTokenizer, AutoModelForCausalLM
except ImportError:
    AutoTokenizer = None
    AutoModelForCausalLM = None

logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s', 
    handlers=[logging.StreamHandler()]
)

logger = logging.getLogger(__name__)


# Distillation output is JSON (short). Do not reserve max_model_len//4 for generation — that
# shrinks prompt budget and forces truncation of long rollouts (bad for Phi-4–class 32k windows).
_DISTILL_JSON_MAX_GEN = 4096


def _distill_gen_and_prompt_cap(max_model_len: int, max_tokens_arg: int) -> tuple[int, int]:
    """Split max_model_len: small gen budget for JSON; remainder for full prompt (rollouts)."""
    gen_cap = min(max_tokens_arg, _DISTILL_JSON_MAX_GEN)
    gen_cap = max(1, gen_cap)
    max_prompt_len = max(1, max_model_len - gen_cap)
    return gen_cap, max_prompt_len


def _fit_prompt_tokens(
    tokenizer,
    full_prompt: str,
    max_prompt_len: int,
    qid_hint: str,
    policy: str,
) -> Optional[str]:
    """Return prompt if it fits; if not, truncate or skip per policy ('truncate' | 'skip')."""
    ids = tokenizer.encode(full_prompt)
    if len(ids) <= max_prompt_len:
        return full_prompt
    if policy == "skip":
        logger.warning(
            "Skipping distillation task: prompt %d tokens > budget %d (question_id=%s). "
            "Use --oversized-prompt-policy truncate if you prefer to truncate.",
            len(ids),
            max_prompt_len,
            qid_hint,
        )
        return None
    logger.warning(
        "Truncating distillation prompt from %d to %d tokens (question_id=%s)",
        len(ids),
        max_prompt_len,
        qid_hint,
    )
    ids = ids[:max_prompt_len]
    return tokenizer.decode(ids, skip_special_tokens=False)


EXPERIENCE_DISTILLATION_SYSTEM_PROMPT = """"You are a Strategic Reasoning Distiller. Your goal is to construct a "Experience Bank" that will serve as the foundation for the student's next problem-solving iteration by extracting two specific lists:
1.  **Verified Propositions:** Irrefutable truths and intermediate conclusions derived correctly.
2.  **Critical Pitfalls:** Logical fallacies, dangerous operations, and dead ends to avoid.
The student will explicitly reference this data: 
- Utilizing **Verified Propositions** as established anchors to accelerate valid reasoning
- Consulting **Critical Pitfalls** to proactively avoid repeating previously identified errors, logic gaps, or dead ends.

**Constraint: strict_neutrality**
You have **NO access** to the golden answer. You must **NOT** make any assumptions about whether the student's final conclusion is correct or incorrect. Treat the student's work as an unverified hypothesis; verify the validity of each step strictly based on logic and mathematical axioms alone.

## Task 1: verified_propositions (List[str])

**Goal:** Extract *only* mathematically sound, reusable facts (Truth Anchors).

**Strict Inclusion Rules (Filter Aggressively):**
1.  **Independent Verification:** You must be able to independently verify the statement is true based on standard mathematical axioms or strictly derived from the previous valid steps.
2.  **Explicit Conditions:** Every proposition MUST state its necessary conditions (e.g., "If $x \neq 0$, then...", "For $a > 0$, implies..."). Do not assume global constraints apply unless stated.
3.  **Atomicity:** Break complex thoughts into the smallest reusable units.
4.  **No "Lucky Guesses":** Do not include conclusions that are "likely true" or "verified by plugging in numbers" but lack logical derivation in the text.
5.  **Self-Contained:** The string must be understandable without reading the original student text. Replace pronouns like "it" or "the equation" with specific variables or expressions.

**Content to Extract:**
*   **Valid Intermediate Calculations:** Concrete results derived accurately from previous steps (e.g., "The derivative $f'(x)$ is calculated as $3x^2 - 4$", "The discriminant $\Delta$ equals $16$", "The roots of the auxiliary equation are $x=2, x=3$").
*   **Algebraic Equivalences:** Correctly simplified or rearranged forms of equations/expressions (e.g., "Equation (1) is equivalent to $y = 2x + 1$ under the given constraints").
*   **Logical Implications & Domain Constraints:** Deductions regarding variable ranges, inequalities, or existence conditions (e.g., "Since $x$ is a length, $x > 0$", "Therefore, $a$ must be an integer").
*   **Correct Application of Theorems/Identities:** Standard mathematical definitions or theorems used where all conditions are visibly met (e.g., "Applying Pythagorean theorem: $a^2 + b^2 = c^2$").

**Format:**
*   `"<Complete Statement with Conditions>. (Source: <Derivation/Method>)"`

## Task 2: critical_pitfalls (List[str])

**Goal:** Identify "Negative Constraints" that serve as warning signs for future explorations.

**Focus on identifying these specific categories:**
1.  **Dead Ends (Strategy Failures):** Approaches that are technically valid but lead to unmanageable complexity, circular reasoning, or an unsolvable state (e.g., expanding a high-power polynomial unnecessarily).
2.  **Fatal Logic Flaws (Actual Errors):** Fundamental errors that ruin the attempt, such as non-equivalent transformations, confusing sufficient/necessary conditions, or calculation mistakes.
3.  **Potential Risks (Unsafe Operations):** Correct-looking steps that lack necessary checks (e.g., dividing by a variable that could be zero, squaring equations without checking for extraneous roots).
4.  **Missing Proof Obligations:** Leaps in logic where a case was ignored or a theorem was applied without verifying its preconditions.

**Format:**
*   `"<Context/Step> -> <Type: Dead End / Fatal Flaw / Potential Risk> -> <Explanation: Trigger + Invalid Action + Consequence>"`

**Explanation Requirements (The "WHY"):**
*   **Trigger:** What specific expression or structure caused the issue?
*   **Invalid Action:** What did the student fail to check, or do incorrectly?
*   **Consequence:** What is the mathematical result? (e.g., "Loss of valid solution x=1," "Explosion of terms making solution impossible," "False conclusion derived").

**Example:**
*   `"Dividing both sides by (x-1) -> Potential Risk -> Trigger: (x-1) in denominator; Invalid Action: Failed to verify x!=1; Consequence: Zero division error and loss of potential solution."`

## Output Requirements

*   **Output ONLY a raw JSON object.**
*   No Markdown formatting (no ```json ... ```), no explanations, no chat.
*   Ensure all LaTeX backslashes are escaped properly for JSON (e.g., `\\frac`).

**JSON Structure:**

{
    "verified_propositions": [
        "<Complete Statement with Conditions>. (Source: <Derivation/Method>)",
        "..."
    ],
    "critical_pitfalls": [
        "<Context/Step> -> <Type: Dead End / Fatal Flaw / Potential Risk> -> <Explanation: Trigger + Invalid Action + Consequence>",
        "..."
    ]
}

## Input Data

**Question:**
{{question}}

**Student's Attempt:**
{{attempt}}
"""

EXPERIENCE_CF_SYSTEM_PROMPT = """"You are a Counterfactual Experience Distiller.

Your goal is to construct an Experience Bank using success-failure trajectory pairs and results-supervised counterfactual attribution.

Core idea (do NOT summarize the entire trajectory):
1) You are given a SUCCESS attempt and a FAILURE attempt for the same question.
2) Both attempts share a long prefix, but diverge at a specific decision region.
3) Use the provided shared prefix and divergence fragments to locate the divergence region.
4) Attribute key decisions near the divergence:
   - Extract verified_propositions ONLY from the SUCCESS side (what the success decision implies).
   - Extract critical_pitfalls ONLY from the FAILURE side (what the failure decision leads to).
5) Counterfactual supervision:
   Assume the FAILURE would replace its divergence decision with the SUCCESS divergence decision.
   Based on the given success/failure outcomes and the divergence fragments, infer which experiences are beneficial (positive evidence) or risky (negative evidence).

Strict Output Rules:
Return ONLY a raw JSON object with the following structure. No Markdown.
The JSON must contain exactly:
{
  "verified_propositions": [ ... ],
  "critical_pitfalls": [ ... ]
}

verified_propositions:
- Must be mathematically sound and stated with necessary conditions.
- Must be self-contained and reusable. Avoid lucky guesses.

critical_pitfalls:
- Must describe the unsafe/incorrect reasoning pattern near the divergence.
- Format suggestion:
  "<Context/Decision> -> <Type: Dead End / Fatal Flaw / Potential Risk> -> <Explanation: Trigger + Invalid Action + Consequence>"

## Input Data (use only these fragments; do NOT summarize the whole trajectory)
**Question:**
{{question}}

**Shared Prefix:**
{{shared_prefix}}

**Success Divergence Fragment:**
{{success_divergence_fragment}}

**Failure Divergence Fragment:**
{{failure_divergence_fragment}}
"""

EXPERIENCE_CF_MIN_EDIT_SYSTEM_PROMPT = """"You are a Minimal-Counterfactual Experience Distiller.

Your goal is to bind experiences to the **smallest sufficient causal region** at the first decision point where two attempts diverge, instead of summarizing long arbitrary fragments.

Setup:
1) You are given a SUCCESS attempt and a FAILURE attempt for the same question.
2) They share a longest common **word-level** prefix; immediately after that prefix, the two continuations differ — the provided **minimal heads** are exactly the first few words on each side right after that shared prefix (the shortest contrast window at the fork).
3) Treat this as a **nearest flip**: the meaningful edit that distinguishes the two trajectories is concentrated in this minimal region (not a whole-paragraph replacement).

Tasks:
- Extract **verified_propositions** ONLY from what the SUCCESS minimal head and its immediate continuation imply (sound, reusable, self-contained).
- Extract **critical_pitfalls** ONLY from what the FAILURE minimal head implies (the risky decision at the fork).

Do NOT restate entire solutions. Anchor every item to the minimal divergence.

Strict Output Rules:
Return ONLY a raw JSON object. No Markdown.
{
  "verified_propositions": [ ... ],
  "critical_pitfalls": [ ... ]
}

## Input Data
**Question:**
{{question}}

**Shared Prefix (suffix near fork):**
{{shared_prefix}}

**SUCCESS minimal head (first words after shared prefix):**
{{success_minimal_head}}

**FAILURE minimal head (first words after shared prefix):**
{{failure_minimal_head}}
"""

EXPERIENCE_PAIRWISE_MARGIN_SYSTEM_PROMPT = """"You are a Pairwise-Ranking Experience Distiller (Bradley–Terry / margin at the decision point).

You are given two continuations after the **same** shared prefix: one leads to a **correct** final answer (SUCCESS branch), the other to an **incorrect** answer (FAILURE branch).

Goal:
- Perform **pairwise comparison** at the fork: treat the SUCCESS continuation as preferred over the FAILURE continuation with an implicit **margin** (why the good branch should win at this decision).
- Extract **verified_propositions** only from the SUCCESS side (what makes the preferred continuation locally justified).
- Extract **critical_pitfalls** only from the FAILURE side (what makes the dispreferred continuation locally risky), framed as contrast to the success branch.

This is NOT independent per-trajectory labeling: every item should be justified by **relative** superiority of the success minimal continuation over the failure minimal continuation at the same prefix.

Strict Output Rules:
Return ONLY a raw JSON object. No Markdown.
{
  "verified_propositions": [ ... ],
  "critical_pitfalls": [ ... ]
}

## Input Data
**Question:**
{{question}}

**Shared Prefix (suffix near fork):**
{{shared_prefix}}

**SUCCESS continuation head (after shared prefix):**
{{success_minimal_head}}

**FAILURE continuation head (after shared prefix):**
{{failure_minimal_head}}
"""

def load_jsonl(file_path):
    data_points = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                try:
                    data_points.append(json.loads(line.strip()))
                except json.JSONDecodeError:
                    continue
    return data_points

def save_to_jsonl(file_path, data):
    with open(file_path, 'w', encoding='utf-8') as f:
        for item in data:
            json.dump(item, f, ensure_ascii=False)
            f.write('\n')

def format_prompt(question: str, attempt: str) -> str:
    return EXPERIENCE_DISTILLATION_SYSTEM_PROMPT.replace("{{question}}", question).replace("{{attempt}}", attempt)

def shared_prefix_word_len(a: List[str], b: List[str]) -> int:
    """Compute longest common prefix length over word tokens."""
    m = min(len(a), len(b))
    i = 0
    while i < m and a[i] == b[i]:
        i += 1
    return i

def format_cf_prompt(question: str, success_attempt: str, failure_attempt: str) -> str:
    success_words = success_attempt.split()
    failure_words = failure_attempt.split()

    prefix_len = shared_prefix_word_len(success_words, failure_words)

    # Divergence context windows (word-level for speed).
    before = 40
    after = 120
    s_start = max(0, prefix_len - before)
    f_start = max(0, prefix_len - before)
    s_end = min(len(success_words), prefix_len + after)
    f_end = min(len(failure_words), prefix_len + after)

    shared_prefix = " ".join(success_words[:prefix_len][-before:]) if prefix_len > 0 else ""
    success_div = " ".join(success_words[s_start:s_end])
    failure_div = " ".join(failure_words[f_start:f_end])

    return (
        EXPERIENCE_CF_SYSTEM_PROMPT
        .replace("{{question}}", question)
        .replace("{{shared_prefix}}", shared_prefix)
        .replace("{{success_divergence_fragment}}", success_div)
        .replace("{{failure_divergence_fragment}}", failure_div)
        .replace("{{success_attempt}}", success_attempt)
        .replace("{{failure_attempt}}", failure_attempt)
    )


def _minimal_heads_after_lcp(
    success_attempt: str,
    failure_attempt: str,
    min_head_words: int,
) -> Tuple[str, str, str]:
    """Longest common word-prefix, then first `min_head_words` words on each side (minimal fork contrast)."""
    success_words = success_attempt.split()
    failure_words = failure_attempt.split()
    prefix_len = shared_prefix_word_len(success_words, failure_words)
    before = 40
    shared_prefix = " ".join(success_words[:prefix_len][-before:]) if prefix_len > 0 else ""
    success_minimal_head = " ".join(success_words[prefix_len : prefix_len + min_head_words])
    failure_minimal_head = " ".join(failure_words[prefix_len : prefix_len + min_head_words])
    return shared_prefix, success_minimal_head, failure_minimal_head


def format_cf_min_edit_prompt(
    question: str,
    success_attempt: str,
    failure_attempt: str,
    min_head_words: int = 32,
) -> str:
    shared_prefix, success_minimal_head, failure_minimal_head = _minimal_heads_after_lcp(
        success_attempt, failure_attempt, min_head_words
    )
    return (
        EXPERIENCE_CF_MIN_EDIT_SYSTEM_PROMPT.replace("{{question}}", question)
        .replace("{{shared_prefix}}", shared_prefix)
        .replace("{{success_minimal_head}}", success_minimal_head)
        .replace("{{failure_minimal_head}}", failure_minimal_head)
    )


def format_pairwise_margin_prompt(
    question: str,
    success_attempt: str,
    failure_attempt: str,
    min_head_words: int = 32,
) -> str:
    shared_prefix, success_minimal_head, failure_minimal_head = _minimal_heads_after_lcp(
        success_attempt, failure_attempt, min_head_words
    )
    return (
        EXPERIENCE_PAIRWISE_MARGIN_SYSTEM_PROMPT.replace("{{question}}", question)
        .replace("{{shared_prefix}}", shared_prefix)
        .replace("{{success_minimal_head}}", success_minimal_head)
        .replace("{{failure_minimal_head}}", failure_minimal_head)
    )


def append_cfexp_pair_supplements(
    question: str,
    attempt_texts: List[Tuple[int, str]],
    succ_idxs: List[int],
    fail_idxs: List[int],
    original_idx: int,
    q_id: str,
    all_tasks: List[Dict[str, Any]],
    max_pairs: int = 4,
) -> None:
    """Append cf_exp-style full-trajectory pair tasks (failure-anchored) without removing per-rollout coverage."""
    if not succ_idxs or not fail_idxs:
        return
    idx_to_words = {c_idx: txt.split() for c_idx, txt in attempt_texts}
    selected_pairs: List[Tuple[int, int, int]] = []

    for fail_idx in fail_idxs[:32]:
        f_words = idx_to_words.get(fail_idx, [])
        if not f_words:
            continue
        best = None
        for succ_idx in succ_idxs[:32]:
            s_words = idx_to_words.get(succ_idx, [])
            if not s_words:
                continue
            prefix_len = shared_prefix_word_len(s_words, f_words)
            if best is None or prefix_len > best[2]:
                best = (fail_idx, succ_idx, prefix_len)
        if best is not None:
            selected_pairs.append(best)

    selected_pairs.sort(key=lambda x: x[2], reverse=True)
    selected_pairs = selected_pairs[:max_pairs]

    for fail_idx, succ_idx, _ in selected_pairs:
        succ_text = next(txt for idx, txt in attempt_texts if idx == succ_idx)
        fail_text = next(txt for idx, txt in attempt_texts if idx == fail_idx)
        prompt = format_cf_prompt(
            question=question,
            success_attempt=succ_text,
            failure_attempt=fail_text,
        )
        all_tasks.append({
            'original_idx': original_idx,
            'question_id': q_id,
            'rollout_idx': fail_idx,
            'prompt': prompt,
            'retries': 0
        })


def is_success_attempt(attempt_text: str, ground_truth: str) -> bool:
    try:
        pred = extract_answer(attempt_text, "math")
    except Exception:
        return False
    try:
        return bool(math_equal(str(pred), str(ground_truth), timeout=True))
    except Exception:
        return False


def _majority_representative_preds(preds: List[Any]) -> Optional[Any]:
    """Largest math_equal cluster among non-empty extracted answers (pseudo-oracle)."""
    valid: List[Any] = []
    for p in preds:
        if p is None:
            continue
        if str(p).strip() == "":
            continue
        valid.append(p)
    if not valid:
        return None
    clusters: List[List[Any]] = []
    for x in valid:
        placed = False
        for g in clusters:
            try:
                if math_equal(str(x), str(g[0]), timeout=True):
                    g.append(x)
                    placed = True
                    break
            except Exception:
                continue
        if not placed:
            clusters.append([x])
    best = max(clusters, key=len)
    return best[0]


def pseudo_success_fail_indices(attempt_texts: List[Tuple[int, str]]) -> Tuple[List[int], List[int]]:
    """Label rollouts by agreement with plurality of extracted answers (no golden reference)."""
    preds_by_idx: List[Tuple[int, Any]] = []
    for c_idx, text in attempt_texts:
        try:
            pred = extract_answer(text, "math")
        except Exception:
            pred = None
        preds_by_idx.append((c_idx, pred))
    preds_only = [p for _, p in preds_by_idx if p is not None and str(p).strip()]
    rep = _majority_representative_preds(preds_only)
    if rep is None:
        return [], []
    succ: List[int] = []
    fail: List[int] = []
    for c_idx, pred in preds_by_idx:
        if pred is None or str(pred).strip() == "":
            fail.append(c_idx)
            continue
        try:
            if math_equal(str(pred), str(rep), timeout=True):
                succ.append(c_idx)
            else:
                fail.append(c_idx)
        except Exception:
            fail.append(c_idx)
    return succ, fail

def extract_and_validate_json(text: str) -> Optional[str]:
    cleaned_text = text
    
    if "</think>" in text:
        cleaned_text = text.split("</think>")[-1].strip()
    
    json_match = re.search(r"```json\s*(.*?)\s*```", cleaned_text, re.DOTALL)
    if json_match:
        candidate = json_match.group(1).strip()
    else:
        start = cleaned_text.find('{')
        end = cleaned_text.rfind('}')
        if start != -1 and end != -1:
            candidate = cleaned_text[start:end+1]
        else:
            candidate = cleaned_text

    try:
        json.loads(candidate)
        return candidate
    except json.JSONDecodeError:
        return None

def main():
    parser = argparse.ArgumentParser(description='Offline Experience Distillation (vLLM or HF)')
    parser.add_argument('--model', type=str, required=True, help='Model path')
    parser.add_argument('--question-file', type=str, required=True, help='Original question file (jsonl)')
    parser.add_argument('--answer-dir', type=str, required=True, help='Directory containing answer rollouts')
    parser.add_argument('--output-dir', type=str, required=True, help='Output directory')
    
    parser.add_argument('--tensor-parallel-size', '-tp', type=int, default=8, help='Number of GPUs')
    parser.add_argument(
        '--max-model-len',
        type=int,
        default=100000,
        help='vLLM: max sequence length (KV cache); must be <= model config (e.g. 32768 for Phi-4-reasoning)',
    )
    parser.add_argument(
        '--gpu-memory-utilization',
        type=float,
        default=0.90,
        help='vLLM: fraction of GPU memory reserved for the engine (weights + KV). Raise if KV init fails at large max_model_len.',
    )
    parser.add_argument(
        '--max-num-seqs',
        type=int,
        default=None,
        help='vLLM v1: max concurrent sequences；默认 1024 时 top-k/p 预热易 OOM，可设 256（4B/共享 GPU 上常用）。',
    )
    parser.add_argument(
        '--disable-custom-all-reduce',
        action='store_true',
        help='vLLM: use NCCL for TP all-reduce (avoids custom all-reduce kernel failures on some multi-GPU setups).',
    )
    parser.add_argument('--batch-size', '-b', type=int, default=100, help='Batch size')
    
    parser.add_argument('--temperature', type=float, default=0.7)
    parser.add_argument('--top-p', type=float, default=0.95)
    parser.add_argument('--top-k', type=int, default=20)
    parser.add_argument('--max-tokens', type=int, default=1024)
    parser.add_argument('--n-samples', type=int, default=3, help='Number of samples per inference (for robustness)')
    parser.add_argument(
        '--experience_judge_mode',
        type=str,
        default='llm_judge',
        choices=[
            'llm_judge',
            'llm_judge_plus_cfexp',
            'pseudo_cf_exp',
            'cf_exp',
            'cf_min_edit',
            'pairwise_margin',
        ],
        help=(
            'llm_judge=per-rollout RSE-style; llm_judge_plus_cfexp=dense llm_judge + optional cf_exp pair supplements; '
            'pseudo_cf_exp=dense llm_judge + cf pair supplements labeled by self-consensus (no golden answer); '
            'cf_exp=shared-prefix divergence + counterfactual; '
            'cf_min_edit=minimal first-divergence heads (nearest flip); '
            'pairwise_margin=pairwise ranking at fork (Bradley–Terry / margin framing).'
        ),
    )
    parser.add_argument(
        '--cf-min-head-words',
        type=int,
        default=32,
        help='For cf_min_edit and pairwise_margin: word length of each branch after LCP (minimal contrast window).',
    )

    parser.add_argument('--start-idx', type=int, default=0)
    parser.add_argument('--end-idx', type=int, default=None)
    parser.add_argument(
        '--oversized-prompt-policy',
        type=str,
        default='truncate',
        choices=['truncate', 'skip'],
        help='If distillation prompt exceeds max_prompt_len: truncate (default) or skip the task (RSE-style no truncation).',
    )
    parser.add_argument('--answer-file-prefix', type=str, default="", help='Prefix for answer files')
    parser.add_argument('--backend', type=str, default='vllm', choices=['vllm', 'hf'], help='Distillation LLM backend')
    parser.add_argument('--device-map', type=str, default='auto', help='HF: device_map')
    parser.add_argument('--dtype', type=str, default='auto', choices=['auto', 'float16', 'bfloat16'], help='HF: dtype')

    args = parser.parse_args()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Loading question file: {args.question_file}")
    questions = load_jsonl(args.question_file)
    
    for i, q in enumerate(questions):
        if 'index' not in q: q['index'] = i
            
    end_idx = args.end_idx if args.end_idx is not None else len(questions)
    questions = questions[args.start_idx : end_idx]
    logger.info(f"Processing range: {args.start_idx} - {end_idx} (Total {len(questions)} items)")

    all_tasks = [] 
    
    logger.info("Scanning and preparing tasks...")
    
    for q_item in tqdm(questions):
        original_idx = q_item['index']
        q_id = q_item.get('question_id', q_item.get('id', ''))
        
        output_file = Path(args.output_dir) / f"{original_idx}.jsonl"
        if output_file.exists():
            continue
            
        answer_path = Path(args.answer_dir) / f"{args.answer_file_prefix}{original_idx}.json"
        if not answer_path.exists():
            continue
            
        try:
            with open(answer_path, 'r') as f:
                answer_data = json.load(f)
        except Exception:
            continue
            
        if answer_data.get('question_id') != q_id:
            continue
            
        completions = answer_data.get('completions', [])
        if not completions:
            continue

        if args.experience_judge_mode == 'llm_judge':
            for c_idx, completion in enumerate(completions):
                attempt_text = ""
                if completion.get("reasoning_content"):
                    attempt_text += completion.get("reasoning_content", "") + "\n\n"
                if completion.get("text"):
                    attempt_text += completion.get("text", "")

                if not attempt_text.strip():
                    continue

                prompt = format_prompt(q_item['question'], attempt_text)

                all_tasks.append({
                    'original_idx': original_idx,
                    'question_id': q_id,
                    'rollout_idx': c_idx,
                    'prompt': prompt,
                    'retries': 0
                })
        elif args.experience_judge_mode == 'llm_judge_plus_cfexp':
            # (1) Same dense coverage as llm_judge: one distillation task per non-empty rollout.
            attempt_texts: List[Tuple[int, str]] = []
            for c_idx, completion in enumerate(completions):
                attempt_text = ""
                if completion.get("reasoning_content"):
                    attempt_text += completion.get("reasoning_content", "") + "\n\n"
                if completion.get("text"):
                    attempt_text += completion.get("text", "")
                if not attempt_text.strip():
                    continue
                attempt_texts.append((c_idx, attempt_text))
                prompt = format_prompt(q_item['question'], attempt_text)
                all_tasks.append({
                    'original_idx': original_idx,
                    'question_id': q_id,
                    'rollout_idx': c_idx,
                    'prompt': prompt,
                    'retries': 0
                })
            # (2) Optional cf_exp supplements: up to 4 (fail, succ) pairs when labels exist.
            ground_truth = q_item.get('answer', '')
            succ_idxs: List[int] = []
            fail_idxs: List[int] = []
            for c_idx, attempt_text in attempt_texts:
                if ground_truth != "" and is_success_attempt(attempt_text, ground_truth):
                    succ_idxs.append(c_idx)
                else:
                    fail_idxs.append(c_idx)
            append_cfexp_pair_supplements(
                q_item['question'],
                attempt_texts,
                succ_idxs,
                fail_idxs,
                original_idx,
                q_id,
                all_tasks,
            )
        elif args.experience_judge_mode == 'pseudo_cf_exp':
            attempt_texts = []
            for c_idx, completion in enumerate(completions):
                attempt_text = ""
                if completion.get("reasoning_content"):
                    attempt_text += completion.get("reasoning_content", "") + "\n\n"
                if completion.get("text"):
                    attempt_text += completion.get("text", "")
                if not attempt_text.strip():
                    continue
                attempt_texts.append((c_idx, attempt_text))
                prompt = format_prompt(q_item['question'], attempt_text)
                all_tasks.append({
                    'original_idx': original_idx,
                    'question_id': q_id,
                    'rollout_idx': c_idx,
                    'prompt': prompt,
                    'retries': 0
                })
            succ_idxs, fail_idxs = pseudo_success_fail_indices(attempt_texts)
            append_cfexp_pair_supplements(
                q_item['question'],
                attempt_texts,
                succ_idxs,
                fail_idxs,
                original_idx,
                q_id,
                all_tasks,
            )
        else:
            # Contrastive modes: build success-failure trajectory pairs (cf_exp / cf_min_edit / pairwise_margin).
            ground_truth = q_item.get('answer', '')
            attempt_texts: List[Tuple[int, str]] = []
            succ_idxs: List[int] = []
            fail_idxs: List[int] = []

            for c_idx, completion in enumerate(completions):
                attempt_text = ""
                if completion.get("reasoning_content"):
                    attempt_text += completion.get("reasoning_content", "") + "\n\n"
                if completion.get("text"):
                    attempt_text += completion.get("text", "")
                if not attempt_text.strip():
                    continue

                attempt_texts.append((c_idx, attempt_text))

                if ground_truth != "" and is_success_attempt(attempt_text, ground_truth):
                    succ_idxs.append(c_idx)
                else:
                    fail_idxs.append(c_idx)

            if not succ_idxs or not fail_idxs:
                # Fallback: if we cannot form pairs, revert to original per-completion extraction.
                for c_idx, attempt_text in attempt_texts:
                    prompt = format_prompt(q_item['question'], attempt_text)
                    all_tasks.append({
                        'original_idx': original_idx,
                        'question_id': q_id,
                        'rollout_idx': c_idx,
                        'prompt': prompt,
                        'retries': 0
                    })
            else:
                # Pre-tokenize at word-level for fast shared-prefix computation.
                idx_to_words = {c_idx: txt.split() for c_idx, txt in attempt_texts}
                max_pairs = 4
                selected_pairs: List[Tuple[int, int, int]] = []  # (fail_idx, succ_idx, shared_prefix_len)

                # For each failure, pick the best success by longest shared prefix.
                for fail_idx in fail_idxs[:32]:
                    f_words = idx_to_words.get(fail_idx, [])
                    if not f_words:
                        continue
                    best = None
                    for succ_idx in succ_idxs[:32]:
                        s_words = idx_to_words.get(succ_idx, [])
                        if not s_words:
                            continue
                        prefix_len = shared_prefix_word_len(s_words, f_words)
                        if best is None or prefix_len > best[2]:
                            best = (fail_idx, succ_idx, prefix_len)
                    if best is not None:
                        selected_pairs.append(best)

                selected_pairs.sort(key=lambda x: x[2], reverse=True)
                selected_pairs = selected_pairs[:max_pairs]

                # Build one distillation prompt per selected pair.
                for fail_idx, succ_idx, _ in selected_pairs:
                    succ_text = next(txt for idx, txt in attempt_texts if idx == succ_idx)
                    fail_text = next(txt for idx, txt in attempt_texts if idx == fail_idx)

                    if args.experience_judge_mode == 'cf_exp':
                        prompt = format_cf_prompt(
                            question=q_item['question'],
                            success_attempt=succ_text,
                            failure_attempt=fail_text,
                        )
                    elif args.experience_judge_mode == 'cf_min_edit':
                        prompt = format_cf_min_edit_prompt(
                            question=q_item['question'],
                            success_attempt=succ_text,
                            failure_attempt=fail_text,
                            min_head_words=args.cf_min_head_words,
                        )
                    else:
                        prompt = format_pairwise_margin_prompt(
                            question=q_item['question'],
                            success_attempt=succ_text,
                            failure_attempt=fail_text,
                            min_head_words=args.cf_min_head_words,
                        )

                    all_tasks.append({
                        'original_idx': original_idx,
                        'question_id': q_id,
                        'rollout_idx': fail_idx,  # anchor on the failure branch
                        'prompt': prompt,
                        'retries': 0
                    })

    if not all_tasks:
        logger.info("No new tasks to process.")
        return

    logger.info(f"Prepared {len(all_tasks)} initial tasks")

    llm = None
    tokenizer = None
    sampling_params = None

    if args.backend == "vllm":
        if LLM is None or SamplingParams is None:
            raise ImportError("vLLM backend selected but vllm is not installed: pip install vllm")
        logger.info(f"Initializing vLLM model: {args.model}")
        _eager = os.environ.get("VLLM_ENFORCE_EAGER", "").lower() in ("1", "true", "yes")
        llm_kw = dict(
            model=args.model,
            tensor_parallel_size=args.tensor_parallel_size,
            trust_remote_code=True,
            gpu_memory_utilization=args.gpu_memory_utilization,
            max_model_len=args.max_model_len,
            enforce_eager=_eager,
        )
        if args.disable_custom_all_reduce:
            llm_kw["disable_custom_all_reduce"] = True
        if args.max_num_seqs is not None:
            llm_kw["max_num_seqs"] = args.max_num_seqs
        llm = LLM(**llm_kw)
        tokenizer = llm.get_tokenizer()
    else:
        if AutoTokenizer is None or AutoModelForCausalLM is None:
            raise ImportError("HF backend selected but transformers is not installed: pip install transformers")
        import torch
        dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16, "auto": "auto"}
        torch_dtype = dtype_map.get(args.dtype, "auto")
        logger.info(f"Loading HF model for distillation: {args.model} (device_map={args.device_map}, dtype={args.dtype})")
        tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
        llm = AutoModelForCausalLM.from_pretrained(
            args.model,
            trust_remote_code=True,
            device_map=args.device_map,
            torch_dtype=torch_dtype,
        )
        llm.eval()

    gen_cap, max_prompt_len = _distill_gen_and_prompt_cap(args.max_model_len, args.max_tokens)
    logger.info(
        "Distillation context: max_model_len=%d gen_cap=%d max_prompt_len=%d (max-tokens arg=%d)",
        args.max_model_len,
        gen_cap,
        max_prompt_len,
        args.max_tokens,
    )
    if args.backend == "vllm":
        sampling_params = SamplingParams(
            n=args.n_samples,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            max_tokens=gen_cap,
        )

    pending_tasks = all_tasks
    max_retries = 5
    
    final_results = defaultdict(list)
    
    current_round_tasks = pending_tasks
    
    for round_idx in range(max_retries + 1):
        if not current_round_tasks:
            break
        
        next_round_tasks = []
        
        num_batches = (len(current_round_tasks) + args.batch_size - 1) // args.batch_size

        logger.info(f"=== Round {round_idx} (Tasks: {len(current_round_tasks)}, Num_Batch: {num_batches}) ===")
        
        for i in range(num_batches):
            batch_tasks = current_round_tasks[i * args.batch_size : (i + 1) * args.batch_size]
            
            batch_pairs: List[Tuple[Dict[str, Any], str]] = []
            for t in batch_tasks:
                messages = [{"role": "user", "content": t['prompt']}]
                full_prompt = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                qid = str(t.get("question_id", ""))
                fitted = _fit_prompt_tokens(
                    tokenizer, full_prompt, max_prompt_len, qid, args.oversized_prompt_policy
                )
                if fitted is None:
                    continue
                batch_pairs.append((t, fitted))

            if not batch_pairs:
                continue

            if args.backend == "vllm":
                prompt_strings = [p for _, p in batch_pairs]
                outputs = llm.generate(prompt_strings, sampling_params, use_tqdm=True)

                for (task, _), output in zip(batch_pairs, outputs):

                    valid_content = None
                    valid_raw = None

                    for sample in output.outputs:
                        raw_text = sample.text
                        parsed_json = extract_and_validate_json(raw_text)
                        if parsed_json:
                            valid_content = parsed_json
                            valid_raw = raw_text
                            break

                    if valid_content:
                        result_entry = {
                            "question_id": task['question_id'],
                            "rollout_idx": task['rollout_idx'],
                            "experience_raw": valid_raw,
                            "experience_parsed": valid_content
                        }
                        final_results[task['original_idx']].append(result_entry)
                    else:
                        if task['retries'] < max_retries:
                            task['retries'] += 1
                            next_round_tasks.append(task)
                        else:
                            logger.warning(f"Task failed after {max_retries} retries: QID {task['question_id']} Rollout {task['rollout_idx']}")
            else:
                import torch
                for j, (task, full_prompt) in enumerate(batch_pairs):
                    valid_content = None
                    valid_raw = None

                    for _s in range(max(1, args.n_samples)):
                        inp = tokenizer(full_prompt, return_tensors="pt")
                        inp = {k: v.to(llm.device) for k, v in inp.items()}
                        with torch.no_grad():
                            out = llm.generate(
                                **inp,
                                max_new_tokens=gen_cap,
                                do_sample=args.temperature > 0,
                                temperature=args.temperature,
                                top_p=args.top_p,
                                top_k=args.top_k,
                                pad_token_id=tokenizer.eos_token_id,
                            )
                        gen_ids = out[0][inp["input_ids"].shape[1]:]
                        raw_text = tokenizer.decode(gen_ids, skip_special_tokens=True)
                        parsed_json = extract_and_validate_json(raw_text)
                        if parsed_json:
                            valid_content = parsed_json
                            valid_raw = raw_text
                            break

                    if valid_content:
                        result_entry = {
                            "question_id": task['question_id'],
                            "rollout_idx": task['rollout_idx'],
                            "experience_raw": valid_raw,
                            "experience_parsed": valid_content
                        }
                        final_results[task['original_idx']].append(result_entry)
                    else:
                        if task['retries'] < max_retries:
                            task['retries'] += 1
                            next_round_tasks.append(task)
                        else:
                            logger.warning(f"Task failed after {max_retries} retries: QID {task['question_id']} Rollout {task['rollout_idx']}")
        
        current_round_tasks = next_round_tasks

    logger.info("Saving results...")
    for q_idx, results in final_results.items():
        results.sort(key=lambda x: x['rollout_idx'])
        output_file = Path(args.output_dir) / f"{q_idx}.jsonl"
        save_to_jsonl(output_file, results)
            
    logger.info(f"All done! Results saved to: {args.output_dir}")

if __name__ == "__main__":
    main()
