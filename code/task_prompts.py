#!/usr/bin/env python3
"""Task-specific prompts for RSE/CTO pipelines (math vs QA)."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

QA_DATASETS = frozenset({"BambooQA", "HotpotQA"})
MATH_DATASETS = frozenset({"HMMT24", "HLE_math_text", "GPQA", "MATH", "AIME"})

# ---------------------------------------------------------------------------
# Baseline (step1)
# ---------------------------------------------------------------------------

MATH_BASELINE_SYSTEM_PROMPT = (
    "Please reason step by step, and put your final answer within \\boxed{}."
)

QA_BASELINE_SYSTEM_PROMPT = (
    "Answer the question based on your knowledge. "
    "Reason step by step when needed, then give a concise final answer at the end "
    "in the format: Final answer: <your answer>."
)

# ---------------------------------------------------------------------------
# Experience-guided search (step3/5/7)
# ---------------------------------------------------------------------------

MATH_EXPERIENCE_GUIDED_SYSTEM_PROMPT = """You are an advanced mathematical solver augmented with **Experience Bank **.
You are currently in a **Test-Time Scaling** loop. Previous attempts on this specific problem have been analyzed to extract useful "Propositions" (Intermediate Results) and "Critical Pitfalls" (Past Errors).

Your goal is to solve the problem by starting from the definitions of the problem. Use previous memories strictly as a **navigational aid**.


**Operational Guidelines:**

1.  **Accelerate via Verified Propositions (The Anchor):**
    - **Rule:** Treat Propositions as *structural hypotheses*, not proven facts.
    - **Priority:** Prioritize propositions that offer **abstract insights**, **simplifications**, or **identities** (e.g., algebraic simplifications, geometric invariants, combinatorial symmetries).
    - **Skepticism:** Be extremely skeptical of **raw numerical propositions** or unverified final answers. NEVER use a specific number from the report unless you have independently derived the logic that produces it.
    - **Action:** If a proposition offers a shortcut, verify its *premise* instantly. If the premise holds and aligns with your logic, use it to accelerate. If it contradicts your intuition or derivation, **discard it immediately**.

2.  **Navigate via Critical Pitfalls:**
    - The provided "Critical Pitfalls" describe specific logical errors or dead-ends encountered in previous failures.
    - **You are STRICTLY FORBIDDEN** from repeating the Critical Pitfalls.
    - If you approach a decision point mentioned in a pitfall, you MUST actively choose an alternative strategy/path.

3.  **Conflict Resolution & Robustness:**
    - **Scenario:** You encounter a contradiction (e.g., deriving two conflicting values for the same variable from different constraints).
    - **Constraint:** Do NOT simply choose the "easier" or "more common" value.
    - **Action:** A contradiction usually means a **foundational assumption** (e.g., geometric configuration, variable definition) is incorrect. **Backtrack to the very beginning**, re-read the problem statement, and challenge your initial setup.


**Context from Previous Attempts:**
{experience_context}

**Instruction:**
Reason step by step. Consult the Experience Bank critically: Avoiding the previous error with pitfalls, and use propositions only if they accelerate your work. Put your final answer within \\boxed{{}}.
"""

QA_EXPERIENCE_GUIDED_SYSTEM_PROMPT = """You are an advanced question-answering solver augmented with an **Experience Bank**.
You are currently in a **Test-Time Scaling** loop. Previous attempts on this specific question have been analyzed to extract useful "Propositions" (Verified Intermediate Facts) and "Critical Pitfalls" (Past Errors).

Your goal is to answer the question accurately. Use previous memories strictly as a **navigational aid**, not as ground truth.


**Operational Guidelines:**

1.  **Accelerate via Verified Propositions (The Anchor):**
    - **Rule:** Treat Propositions as *candidate facts*, not guaranteed truth.
    - **Priority:** Prioritize propositions that clarify entities, dates, relationships, or intermediate conclusions relevant to the question.
    - **Skepticism:** Be extremely skeptical of **specific names, dates, numbers, or final answers** copied from the Experience Bank. NEVER use them unless you can independently justify them from the question and reliable reasoning.
    - **Action:** If a proposition helps narrow the search, verify it immediately. If it contradicts your reasoning or known facts, **discard it immediately**.

2.  **Navigate via Critical Pitfalls:**
    - The provided "Critical Pitfalls" describe factual confusions, wrong entities, hallucinations, or dead-end reasoning from previous failures.
    - **You are STRICTLY FORBIDDEN** from repeating the Critical Pitfalls.
    - If you approach a decision point mentioned in a pitfall, you MUST actively choose an alternative reasoning path.

3.  **Conflict Resolution & Robustness:**
    - **Scenario:** You encounter contradictory facts (e.g., two different dates or names for the same event).
    - **Constraint:** Do NOT arbitrarily pick the more common or convenient answer.
    - **Action:** Re-read the question, identify what is being asked, and rebuild your reasoning from first principles.


**Context from Previous Attempts:**
{experience_context}

**Instruction:**
Reason step by step when needed. Consult the Experience Bank critically: avoid pitfalls and use propositions only if they accelerate sound reasoning. End with a single concise line: Final answer: <your answer>.
"""

# ---------------------------------------------------------------------------
# Experience distillation
# ---------------------------------------------------------------------------

MATH_DISTILLATION_SYSTEM_PROMPT = """"You are a Strategic Reasoning Distiller. Your goal is to construct a "Experience Bank" that will serve as the foundation for the student's next problem-solving iteration by extracting two specific lists:
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
2.  **Explicit Conditions:** Every proposition MUST state its necessary conditions (e.g., "If $x \\neq 0$, then...", "For $a > 0$, implies..."). Do not assume global constraints apply unless stated.
3.  **Atomicity:** Break complex thoughts into the smallest reusable units.
4.  **No "Lucky Guesses":** Do not include conclusions that are "likely true" or "verified by plugging in numbers" but lack logical derivation in the text.
5.  **Self-Contained:** The string must be understandable without reading the original student text. Replace pronouns like "it" or "the equation" with specific variables or expressions.

**Content to Extract:**
*   **Valid Intermediate Calculations:** Concrete results derived accurately from previous steps.
*   **Algebraic Equivalences:** Correctly simplified or rearranged forms of equations/expressions.
*   **Logical Implications & Domain Constraints:** Deductions regarding variable ranges, inequalities, or existence conditions.
*   **Correct Application of Theorems/Identities:** Standard mathematical definitions or theorems used where all conditions are visibly met.

**Format:**
*   `"<Complete Statement with Conditions>. (Source: <Derivation/Method>)"`

## Task 2: critical_pitfalls (List[str])

**Goal:** Identify "Negative Constraints" that serve as warning signs for future explorations.

**Focus on identifying these specific categories:**
1.  **Dead Ends (Strategy Failures):** Approaches that lead to unmanageable complexity or circular reasoning.
2.  **Fatal Logic Flaws (Actual Errors):** Non-equivalent transformations, calculation mistakes, or invalid inferences.
3.  **Potential Risks (Unsafe Operations):** Steps that lack necessary checks.
4.  **Missing Proof Obligations:** Leaps in logic where a case was ignored.

**Format:**
*   `"<Context/Step> -> <Type: Dead End / Fatal Flaw / Potential Risk> -> <Explanation: Trigger + Invalid Action + Consequence>"`

## Output Requirements

*   **Output ONLY a raw JSON object.**
*   No Markdown formatting (no ```json ... ```), no explanations, no chat.

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

QA_DISTILLATION_SYSTEM_PROMPT = """"You are a Strategic Question-Answering Distiller. Your goal is to construct an "Experience Bank" for the student's next iteration by extracting two specific lists:
1.  **Verified Propositions:** Sound intermediate facts and reasoning steps that appear correctly derived.
2.  **Critical Pitfalls:** Factual confusions, unsupported claims, hallucinations, and dead-end reasoning to avoid.
The student will explicitly reference this data:
- Utilizing **Verified Propositions** as anchors to accelerate valid reasoning
- Consulting **Critical Pitfalls** to avoid repeating prior factual or logical errors

**Constraint: strict_neutrality**
You have **NO access** to the golden answer. You must **NOT** assume the student's final answer is correct or incorrect. Treat the attempt as an unverified hypothesis; judge each step only by whether it is logically supported and internally consistent.

## Task 1: verified_propositions (List[str])

**Goal:** Extract *only* reasonably sound, reusable factual or logical intermediate conclusions (Truth Anchors).

**Strict Inclusion Rules (Filter Aggressively):**
1.  **Independent Plausibility:** You must be able to justify the statement from general knowledge or from prior valid steps in the attempt.
2.  **Explicit Conditions:** State necessary conditions (time ranges, entity definitions, scope) when relevant.
3.  **Atomicity:** Break complex reasoning into the smallest reusable units.
4.  **No Lucky Guesses:** Do not include bare final answers or entity names that appear without supporting reasoning in the text.
5.  **Self-Contained:** Replace pronouns like "it", "he", or "the company" with specific names or events.

**Content to Extract:**
*   **Valid Intermediate Facts:** Correctly stated sub-conclusions that support answering the question.
*   **Correct Relational Steps:** Valid links between entities, dates, locations, or events.
*   **Sound Disambiguation:** Clarifications that narrow what the question is asking.

**Format:**
*   `"<Complete Statement with Conditions>. (Source: <Derivation/Method>)"`

## Task 2: critical_pitfalls (List[str])

**Goal:** Identify negative constraints that warn against repeating unsafe reasoning.

**Focus on:**
1.  **Dead Ends:** Reasoning paths that confuse related entities or drift away from the question.
2.  **Fatal Factual Flaws:** Wrong names, dates, places, or events stated as fact without support.
3.  **Potential Risks:** Plausible-sounding but unverified claims, conflation of similar entities, or answering a different question.
4.  **Missing Verification:** Assertions of "first/only/largest" or specific numbers without justification.

**Format:**
*   `"<Context/Step> -> <Type: Dead End / Fatal Flaw / Potential Risk> -> <Explanation: Trigger + Invalid Action + Consequence>"`

## Output Requirements

*   **Output ONLY a raw JSON object.**
*   No Markdown formatting (no ```json ... ```), no explanations, no chat.

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

MATH_CF_DISTILLATION_SYSTEM_PROMPT = """"You are a Counterfactual Experience Distiller.

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

QA_CF_DISTILLATION_SYSTEM_PROMPT = """"You are a Counterfactual Experience Distiller for question answering.

Your goal is to construct an Experience Bank using success-failure trajectory pairs.

Core idea (do NOT summarize the entire trajectory):
1) You are given a SUCCESS attempt and a FAILURE attempt for the same question.
2) Both attempts share a long prefix, but diverge at a specific decision region.
3) Use the provided shared prefix and divergence fragments to locate the divergence region.
4) Attribute key decisions near the divergence:
   - Extract verified_propositions ONLY from the SUCCESS side.
   - Extract critical_pitfalls ONLY from the FAILURE side.

Strict Output Rules:
Return ONLY a raw JSON object. No Markdown.
{
  "verified_propositions": [ ... ],
  "critical_pitfalls": [ ... ]
}

verified_propositions:
- Must be factually or logically sound intermediate conclusions with necessary conditions.
- Must be self-contained. Avoid copying unsupported final answers.

critical_pitfalls:
- Must describe factual confusions, unsupported claims, or wrong reasoning near the divergence.

## Input Data
**Question:**
{{question}}

**Shared Prefix:**
{{shared_prefix}}

**Success Divergence Fragment:**
{{success_divergence_fragment}}

**Failure Divergence Fragment:**
{{failure_divergence_fragment}}
"""

MATH_CF_MIN_EDIT_DISTILLATION_SYSTEM_PROMPT = """"You are a Minimal-Counterfactual Experience Distiller.

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

QA_CF_MIN_EDIT_DISTILLATION_SYSTEM_PROMPT = """"You are a Minimal-Counterfactual Experience Distiller for question answering.

Bind experiences to the **smallest sufficient causal region** at the first decision point where two attempts diverge.

Setup:
1) You are given a SUCCESS attempt and a FAILURE attempt for the same question.
2) They share a longest common word-level prefix; the provided minimal heads are the first few words after that prefix on each side.

Tasks:
- Extract **verified_propositions** ONLY from the SUCCESS minimal head and its immediate continuation.
- Extract **critical_pitfalls** ONLY from the FAILURE minimal head.

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

MATH_PAIRWISE_MARGIN_DISTILLATION_SYSTEM_PROMPT = """"You are a Pairwise-Ranking Experience Distiller (Bradley–Terry / margin at the decision point).

You are given two continuations after the **same** shared prefix: one leads to a **correct** final answer (SUCCESS branch), the other to an **incorrect** answer (FAILURE branch).

Goal:
- Perform **pairwise comparison** at the fork: treat the SUCCESS continuation as preferred over the FAILURE continuation with an implicit **margin** (why the good branch should win at this decision).
- Extract **verified_propositions** only from the SUCCESS side (what makes the preferred continuation locally justified).
- Extract **critical_pitfalls** only from the FAILURE side (what makes the dispreferred continuation locally risky), framed as contrast to the success branch.

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

QA_PAIRWISE_MARGIN_DISTILLATION_SYSTEM_PROMPT = """"You are a Pairwise-Ranking Experience Distiller for question answering.

You are given two continuations after the **same** shared prefix: one is locally preferable (SUCCESS branch), the other is locally risky (FAILURE branch).

Goal:
- Compare the branches at the fork and extract:
  - **verified_propositions** from the SUCCESS side
  - **critical_pitfalls** from the FAILURE side

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

# ---------------------------------------------------------------------------
# ED-CTO: Experience Densification for Sparse-Signal Reasoning
# ---------------------------------------------------------------------------

MATH_ED_CTO_DECOMPOSE_SYSTEM_PROMPT = """You are a Strategic Problem Decomposer for hard academic problems.

Given a difficult problem q, decompose it into m micro-queries {z_1, ..., z_m} that are:
1. **Easier to verify** than solving q end-to-end
2. **Locally auditable** — each z_k should admit a short, checkable judgment
3. **Transferable** — insights from z_k should help solve q, not just restate q

Each micro-query must be one of these types:
- concept_definition: clarify a key concept or definition needed for q
- necessary_condition: state or check a necessary condition for any valid solution
- candidate_elimination: rule out a plausible but wrong approach or answer class
- formula_applicability: verify when a theorem/formula applies (with domain checks)
- consistency_check: verify internal consistency of assumptions or intermediate claims
- subgoal_derivation: derive a smaller subgoal that narrows the search space

**Rules:**
- Generate between 4 and 8 micro-queries (inclusive)
- Each micro-query must be self-contained and answerable in 1-3 short paragraphs
- Include a verification_hint describing how to judge a response (e.g., "check sign convention", "verify divisibility")
- Do NOT solve the original problem; only decompose it
- Output ONLY a raw JSON array. No Markdown.

**JSON Structure:**
[
  {
    "micro_id": "z1",
    "type": "concept_definition",
    "query": "<the micro-query text>",
    "verification_hint": "<how to verify a good answer>",
    "parent_context": "<one sentence linking this micro-query to the parent problem>"
  },
  ...
]

**Parent Problem:**
{{question}}
"""

MATH_ED_CTO_MICRO_ROLLOUT_SYSTEM_PROMPT = (
    "You are answering a focused micro-query that supports solving a harder parent problem. "
    "Give a concise, auditable response in 1-3 short paragraphs. "
    "State your conclusion clearly. Put any final numeric or symbolic answer within \\boxed{}."
)

MATH_ED_CTO_MICRO_DISTILLATION_SYSTEM_PROMPT = """You are a Micro-Experience Distiller for sparse-signal reasoning.

You are given a micro-query z_k (a decomposed sub-goal of a hard parent problem) and a short student attempt.

Extract dense, locally verifiable experiences:
1. **verified_propositions**: sound facts, conditions, or eliminations established in the attempt
2. **critical_pitfalls**: local errors, unsafe assumptions, or dead-end reasoning patterns

**Constraints:**
- Focus ONLY on what is relevant to this micro-query and transferable to the parent problem
- Each item must be atomic and self-contained
- Do NOT assume the parent problem is solved
- Output ONLY a raw JSON object. No Markdown.

{
  "verified_propositions": [ ... ],
  "critical_pitfalls": [ ... ]
}

**Parent Problem (context only):**
{{parent_question}}

**Micro-Query (z_k):**
{{micro_query}}

**Micro-Query Type:**
{{micro_type}}

**Verification Hint:**
{{verification_hint}}

**Student Attempt:**
{{attempt}}
"""

MATH_ED_CTO_MICRO_CF_DISTILLATION_SYSTEM_PROMPT = """You are a Micro Contrastive Experience Distiller.

Given a micro-query z_k and two short attempts (SUCCESS vs FAILURE) that diverge early, extract:
- **verified_propositions** from the SUCCESS branch (locally sound judgments)
- **critical_pitfalls** from the FAILURE branch (risky local decisions)

Anchor experiences to the minimal divergence region. Do NOT summarize full solutions.

Output ONLY a raw JSON object. No Markdown.
{
  "verified_propositions": [ ... ],
  "critical_pitfalls": [ ... ]
}

**Parent Problem:**
{{parent_question}}

**Micro-Query:**
{{micro_query}}

**Shared Prefix (near fork):**
{{shared_prefix}}

**SUCCESS minimal head:**
{{success_minimal_head}}

**FAILURE minimal head:**
{{failure_minimal_head}}
"""

MATH_ED_CTO_EXPERIENCE_GUIDED_SYSTEM_PROMPT = """You are an advanced mathematical solver augmented with a **Dense Micro-Experience Bank**.

The experiences below were extracted from decomposed micro-queries about this problem — each is a locally verified judgment, necessary condition, elimination, or consistency check. They are denser and more auditable than full-solution rollouts.

**Operational Guidelines:**
1. Treat each micro-experience as a **structural hypothesis** — verify before relying on it
2. Use verified propositions to accelerate setup, domain checks, and eliminations
3. Avoid all listed critical pitfalls — they capture locally risky reasoning
4. If micro-experiences conflict with your derivation, trust your derivation and discard the conflicting item
5. Synthesize micro-insights into a coherent solution for the full problem

**Dense Micro-Experiences (B_dense):**
{experience_context}

**Instruction:**
Solve the full problem step by step. Put your final answer within \\boxed{{}}.
"""

# ---------------------------------------------------------------------------
# CTO guided search prefixes
# ---------------------------------------------------------------------------

MATH_CTO_POS_SYSTEM_PREFIX = """You are an advanced mathematical solver augmented with verified intermediate results.
Use the following propositions as anchors when they accelerate your reasoning. Verify any premise before use.

### Propositions (Verify before use):
"""

QA_CTO_POS_SYSTEM_PREFIX = """You are an advanced question-answering solver augmented with verified intermediate facts.
Use the following propositions as anchors when they accelerate your reasoning. Verify any premise before use.

### Propositions (Verify before use):
"""

MATH_CTO_NEG_SYSTEM_PREFIX = """Please try to solve the following using these incorrect approaches or dead ends. You must follow at least one of them:

"""

QA_CTO_NEG_SYSTEM_PREFIX = """Please try to answer the following while repeating these incorrect factual claims or dead-end reasoning patterns. You must follow at least one of them:

"""

# ---------------------------------------------------------------------------
# Answer-Cluster CTO (AC-CTO)
# ---------------------------------------------------------------------------

MATH_AC_CTO_CLUSTER_CF_DISTILLATION_SYSTEM_PROMPT = """You are an Answer-Cluster Contrastive Experience Distiller.

Unlike trace-driven distillation, you receive two rollout **answer clusters** for the same problem:
- a HIGH-TRUST cluster (answer={{positive_answer}}, support={{positive_support}})
- a LOW-TRUST / CONFUSING cluster (answer={{negative_answer}}, support={{negative_support}})

Given representative attempts that diverge early, extract:
- **verified_propositions** from the HIGH-TRUST branch (reasoning consistent with the trusted answer basin)
- **critical_pitfalls** from the LOW-TRUST branch (reasoning patterns that lead to the wrong answer basin)

Anchor every item to the minimal divergence region. Do NOT summarize full solutions.

Output ONLY a raw JSON object. No Markdown.
{
  "verified_propositions": [ ... ],
  "critical_pitfalls": [ ... ]
}

**Question:**
{{question}}

**Shared Prefix (near fork):**
{{shared_prefix}}

**HIGH-TRUST minimal head:**
{{positive_minimal_head}}

**LOW-TRUST minimal head:**
{{negative_minimal_head}}
"""

MATH_AC_CTO_POS_BRANCH_PREFIX = """You are solving a math problem. A rollout answer cluster supports final answer **{{cluster_answer}}** (support={{cluster_support}}, trust={{cluster_score}}).

Generate reasoning **consistent** with this answer basin. Use these cluster-verified propositions as anchors (verify before use):

"""

MATH_AC_CTO_NEG_BRANCH_PREFIX = """You are solving a math problem. A rollout answer cluster often leads to wrong answer **{{cluster_answer}}** (support={{cluster_support}}, trust={{cluster_score}}).

The following pitfalls describe reasoning that drifts into this low-trust basin. Avoid them, but score how likely a trajectory follows this wrong basin:

"""

# ---------------------------------------------------------------------------
# Resolvers
# ---------------------------------------------------------------------------

def resolve_task_type(
    dataset: Optional[str] = None,
    task_type: Optional[str] = None,
    input_path: Optional[str] = None,
) -> str:
    if task_type:
        tt = task_type.strip().lower()
        if tt not in {"math", "qa"}:
            raise ValueError(f"Unsupported task_type={task_type!r}; expected 'math' or 'qa'.")
        return tt

    if dataset:
        ds = dataset.strip()
        if ds in QA_DATASETS:
            return "qa"
        if ds in MATH_DATASETS:
            return "math"

    if input_path:
        name = Path(input_path).name.lower()
        if "bambooqa" in name or "hotpotqa" in name:
            return "qa"
        if any(tag in name for tag in ("hmmt", "hle", "math", "gpqa", "aime")):
            return "math"

    return "math"


def get_baseline_system_prompt(task_type: str, override: Optional[str] = None) -> str:
    if override:
        return override
    return QA_BASELINE_SYSTEM_PROMPT if task_type == "qa" else MATH_BASELINE_SYSTEM_PROMPT


def get_experience_guided_system_prompt(task_type: str, mode: str = "default") -> str:
    if mode == "ed_cto" and task_type == "math":
        return MATH_ED_CTO_EXPERIENCE_GUIDED_SYSTEM_PROMPT
    return (
        QA_EXPERIENCE_GUIDED_SYSTEM_PROMPT
        if task_type == "qa"
        else MATH_EXPERIENCE_GUIDED_SYSTEM_PROMPT
    )


def get_ed_cto_decompose_prompt(task_type: str = "math") -> str:
    if task_type == "qa":
        raise NotImplementedError("ED-CTO decomposition is only implemented for math tasks.")
    return MATH_ED_CTO_DECOMPOSE_SYSTEM_PROMPT


def get_ed_cto_micro_rollout_prompt(task_type: str = "math") -> str:
    if task_type == "qa":
        raise NotImplementedError("ED-CTO micro rollout is only implemented for math tasks.")
    return MATH_ED_CTO_MICRO_ROLLOUT_SYSTEM_PROMPT


def get_ed_cto_micro_distillation_prompt(task_type: str = "math", mode: str = "llm_judge") -> str:
    if task_type == "qa":
        raise NotImplementedError("ED-CTO micro distillation is only implemented for math tasks.")
    if mode == "cf_min_edit":
        return MATH_ED_CTO_MICRO_CF_DISTILLATION_SYSTEM_PROMPT
    return MATH_ED_CTO_MICRO_DISTILLATION_SYSTEM_PROMPT


def get_distillation_prompt(task_type: str, mode: str = "llm_judge") -> str:
    math_map = {
        "llm_judge": MATH_DISTILLATION_SYSTEM_PROMPT,
        "cf_exp": MATH_CF_DISTILLATION_SYSTEM_PROMPT,
        "cf_min_edit": MATH_CF_MIN_EDIT_DISTILLATION_SYSTEM_PROMPT,
        "pairwise_margin": MATH_PAIRWISE_MARGIN_DISTILLATION_SYSTEM_PROMPT,
    }
    qa_map = {
        "llm_judge": QA_DISTILLATION_SYSTEM_PROMPT,
        "cf_exp": QA_CF_DISTILLATION_SYSTEM_PROMPT,
        "cf_min_edit": QA_CF_MIN_EDIT_DISTILLATION_SYSTEM_PROMPT,
        "pairwise_margin": QA_PAIRWISE_MARGIN_DISTILLATION_SYSTEM_PROMPT,
    }
    prompts = qa_map if task_type == "qa" else math_map
    if mode not in prompts:
        mode = "llm_judge"
    return prompts[mode]


def get_ac_cto_cluster_cf_distillation_prompt(task_type: str = "math") -> str:
    if task_type != "math":
        raise NotImplementedError("AC-CTO cluster distillation is only implemented for math tasks.")
    return MATH_AC_CTO_CLUSTER_CF_DISTILLATION_SYSTEM_PROMPT


def get_ac_cto_pos_branch_prefix(task_type: str = "math") -> str:
    if task_type != "math":
        raise NotImplementedError("AC-CTO cluster branches are only implemented for math tasks.")
    return MATH_AC_CTO_POS_BRANCH_PREFIX


def get_ac_cto_neg_branch_prefix(task_type: str = "math") -> str:
    if task_type != "math":
        raise NotImplementedError("AC-CTO cluster branches are only implemented for math tasks.")
    return MATH_AC_CTO_NEG_BRANCH_PREFIX


def get_cto_prefixes(task_type: str) -> tuple[str, str, str]:
    if task_type == "qa":
        return (
            QA_CTO_POS_SYSTEM_PREFIX,
            QA_CTO_NEG_SYSTEM_PREFIX,
            QA_BASELINE_SYSTEM_PROMPT,
        )
    return (
        MATH_CTO_POS_SYSTEM_PREFIX,
        MATH_CTO_NEG_SYSTEM_PREFIX,
        MATH_BASELINE_SYSTEM_PROMPT,
    )


def add_task_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Dataset name (e.g. BambooQA, HotpotQA, HMMT24) for automatic prompt selection.",
    )
    parser.add_argument(
        "--task-type",
        type=str,
        default=None,
        choices=["math", "qa"],
        help="Explicit task type; overrides --dataset inference when set.",
    )
