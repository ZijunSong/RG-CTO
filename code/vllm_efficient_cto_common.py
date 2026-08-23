#!/usr/bin/env python3
"""Shared vLLM utilities for CTO-Rescore."""

from __future__ import annotations

import argparse
import logging
import math
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

import cto_guided_search as cto

logger = logging.getLogger(__name__)


def add_experience_args(parser: argparse.ArgumentParser) -> None:
    cto.add_task_args(parser)
    parser.add_argument("--n-experience-completions", type=int, default=5)
    parser.add_argument(
        "--experience-retrieval",
        type=str,
        default="first",
        choices=[
            "first",
            "embedding",
            "embedding_rerank",
            "cross_query_embedding",
            "cross_query_embedding_rerank",
        ],
    )
    parser.add_argument(
        "--retrieval-embedding-model",
        type=str,
        default="/data/ppnm/models/all-MiniLM-L6-v2",
    )
    parser.add_argument(
        "--retrieval-rerank-model",
        type=str,
        default=cto._DEFAULT_RETRIEVAL_RERANK_MODEL,
    )
    parser.add_argument("--retrieval-rerank-pool-mult", type=int, default=4)
    parser.add_argument("--max-aggregated-propositions", type=int, default=0)
    parser.add_argument("--max-aggregated-pitfalls", type=int, default=0)
    parser.add_argument(
        "--experience-aggregation",
        type=str,
        default="legacy_sorted",
        choices=["legacy_sorted", "flat_top_k", "semantic_merge", "recency_aware"],
    )
    parser.add_argument("--aggregation-semantic-threshold", type=float, default=0.82)
    parser.add_argument("--aggregation-recency-head-fraction", type=float, default=0.55)


def add_vllm_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument(
        "--disable-custom-all-reduce",
        action="store_true",
        help="vLLM: use NCCL for TP all-reduce.",
    )
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--max-model-len", type=int, default=100000)
    parser.add_argument("--max-num-seqs", type=int, default=None)
    parser.add_argument("--vllm-max-num-batched-tokens", type=int, default=None)
    parser.add_argument(
        "--vllm-score-batch-size",
        type=int,
        default=8,
        help="Batch size for teacher-forced prompt_logprobs scoring.",
    )
    parser.add_argument(
        "--vllm-max-score-prompt-tokens",
        type=int,
        default=65536,
        help="Cap prompt length during contrastive scoring (generation uses --max-model-len).",
    )


def init_vllm(args: argparse.Namespace) -> Tuple[Any, Any, Optional[int]]:
    try:
        from vllm import LLM
    except ImportError as exc:
        raise ImportError("vLLM is required: pip install vllm") from exc

    _eager = os.environ.get("VLLM_ENFORCE_EAGER", "").lower() in ("1", "true", "yes")
    llm_kw: Dict[str, Any] = dict(
        model=args.model,
        tensor_parallel_size=args.tensor_parallel_size,
        trust_remote_code=True,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        enforce_eager=_eager,
    )
    if args.max_num_seqs is not None:
        llm_kw["max_num_seqs"] = args.max_num_seqs
    if args.vllm_max_num_batched_tokens is not None:
        llm_kw["max_num_batched_tokens"] = args.vllm_max_num_batched_tokens
    if args.disable_custom_all_reduce:
        llm_kw["disable_custom_all_reduce"] = True

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    llm = LLM(**llm_kw)
    max_len = cto._get_vllm_max_model_len(llm) or args.max_model_len
    return llm, tokenizer, max_len


def build_pos_neg_prompts(
    tokenizer: Any,
    question_text: str,
    experience_data: Dict[str, Any],
) -> Tuple[str, str, List[str]]:
    props_str = cto.build_p_pos_content(experience_data)
    pitfalls_str = cto.build_p_neg_content(experience_data)
    pitfall_texts = [
        p for p in (experience_data.get("critical_pitfalls") or []) if isinstance(p, str)
    ]

    system_pos = (cto.P_POS_SYSTEM_PREFIX + props_str).strip() if props_str else ""
    system_neg = (
        (cto.P_NEG_SYSTEM_PREFIX + pitfalls_str).strip()
        if pitfalls_str
        else "Do not repeat the following errors:\n"
    )
    messages_pos = [
        {"role": "system", "content": system_pos or cto._CTO_FALLBACK_SYSTEM_PROMPT},
        {"role": "user", "content": question_text},
    ]
    messages_neg = [
        {"role": "system", "content": system_neg or "Please reason step by step."},
        {"role": "user", "content": question_text},
    ]
    prompt_pos = tokenizer.apply_chat_template(
        messages_pos, tokenize=False, add_generation_prompt=True
    )
    prompt_neg = tokenizer.apply_chat_template(
        messages_neg, tokenize=False, add_generation_prompt=True
    )
    return prompt_pos, prompt_neg, pitfall_texts


def _logprob_dict_values(logprob_dict: Any) -> List[float]:
    if not logprob_dict:
        return []
    if isinstance(logprob_dict, dict):
        out = []
        for v in logprob_dict.values():
            lp = getattr(v, "logprob", None)
            if lp is None:
                try:
                    lp = float(v)
                except Exception:
                    lp = None
            if lp is not None:
                out.append(float(lp))
        return out
    return []


def sum_output_logprobs(logprobs: Optional[Sequence[Any]]) -> float:
    """Sum observed-token logprobs from vLLM generation logprobs list."""
    if not logprobs:
        return 0.0
    total = 0.0
    for step in logprobs:
        if not step:
            continue
        if isinstance(step, dict):
            lp_obj = next(iter(step.values()), None)
            lp = getattr(lp_obj, "logprob", None)
            if lp is not None:
                total += float(lp)
        elif isinstance(step, list) and step:
            lp = getattr(step[0], "logprob", None)
            if lp is not None:
                total += float(lp)
    return float(total)


def _normalized_entropy_from_step(step: Any, top_k: int = 5) -> float:
    vals = _logprob_dict_values(step)
    if not vals:
        return 0.0
    vals = sorted(vals, reverse=True)[: max(1, top_k)]
    probs = np.exp(np.array(vals, dtype=np.float64))
    probs = probs / max(probs.sum(), 1e-12)
    ent = float(-(probs * np.log(np.clip(probs, 1e-12, 1.0))).sum())
    vocab = max(len(vals), 2)
    return ent / math.log(vocab)


def _max_pitfall_similarity(prefix_text: str, pitfall_texts: List[str], embed_model_path: str) -> float:
    if not pitfall_texts:
        return 0.0
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        return 0.0
    st = cto._EMB_ST.get(embed_model_path)
    if st is None:
        st = SentenceTransformer(embed_model_path)
        cto._EMB_ST[embed_model_path] = st
    q = prefix_text.strip()[:4000]
    if not q:
        return 0.0
    qv = st.encode([q], normalize_embeddings=True)
    pv = st.encode(pitfall_texts[:32], normalize_embeddings=True)
    sim = np.dot(pv, qv.T).flatten()
    return float(sim.max()) if sim.size else 0.0


def score_suffixes_batch(
    llm: Any,
    tokenizer: Any,
    prompt_prefix: str,
    suffixes: List[str],
    *,
    max_model_len: Optional[int],
    vllm_score_batch_size: int,
    vllm_max_score_prompt_tokens: int,
) -> List[float]:
    from vllm import SamplingParams

    if not suffixes:
        return []

    prefix_len = cto._token_count(tokenizer, prompt_prefix)
    vllm_cap = max_model_len or 131072
    score_prompt_budget = max(1, int(vllm_cap) - 1)
    if int(vllm_max_score_prompt_tokens) > 0:
        score_prompt_budget = min(score_prompt_budget, int(vllm_max_score_prompt_tokens))

    full_prompts: List[str] = []
    suffix_lens: List[int] = []
    for suffix in suffixes:
        fp, _pt, sl = cto._truncate_concat_for_vllm(
            tokenizer, prompt_prefix, suffix, score_prompt_budget
        )
        full_prompts.append(fp)
        suffix_lens.append(sl)

    score_params = SamplingParams(
        max_tokens=1,
        n=1,
        temperature=0.0,
        top_p=1.0,
        top_k=-1,
        prompt_logprobs=1,
    )
    scores: List[float] = []
    bs = max(1, int(vllm_score_batch_size))
    for start in range(0, len(full_prompts), bs):
        batch = full_prompts[start : start + bs]
        outs = llm.generate(batch, score_params)
        for bi, out in enumerate(outs):
            plp = getattr(out, "prompt_logprobs", None)
            scores.append(
                cto._sum_prompt_logprob_for_suffix(plp, suffix_lens[start + bi])
            )
    return scores


def load_experience_for_question(
    args: argparse.Namespace,
    experience_path: Any,
    orig_idx: int,
    question_text: str,
    cross_query_parsed: Any,
    cross_query_doc_emb: Any,
) -> Optional[Dict[str, Any]]:
    data = cto.load_and_aggregate_raw_experiences(
        experience_path,
        orig_idx,
        args.n_experience_completions,
        question_text=question_text,
        retrieval=args.experience_retrieval,
        embed_model_path=args.retrieval_embedding_model,
        rerank_model_name=args.retrieval_rerank_model,
        rerank_pool_mult=args.retrieval_rerank_pool_mult,
        cross_query_parsed=cross_query_parsed,
        cross_query_doc_emb=cross_query_doc_emb,
        max_aggregated_propositions=args.max_aggregated_propositions,
        max_aggregated_pitfalls=args.max_aggregated_pitfalls,
        experience_aggregation=args.experience_aggregation,
        aggregation_semantic_threshold=args.aggregation_semantic_threshold,
        aggregation_recency_head_fraction=args.aggregation_recency_head_fraction,
    )
    return data or None


def setup_task_prefixes(args: argparse.Namespace) -> str:
    task_type = cto.resolve_task_type(
        dataset=args.dataset,
        task_type=args.task_type,
        input_path=args.input,
    )
    cto.P_POS_SYSTEM_PREFIX, cto.P_NEG_SYSTEM_PREFIX, cto._CTO_FALLBACK_SYSTEM_PROMPT = (
        cto.get_cto_prefixes(task_type)
    )
    return task_type


def prepare_pending_questions(
    args: argparse.Namespace,
    questions: List[Dict[str, Any]],
    output_path: Any,
) -> List[Dict[str, Any]]:
    pending = []
    for i, item in enumerate(questions):
        orig_idx = args.start_idx + i
        item["index"] = orig_idx
        out_file = output_path / f"{orig_idx}.json"
        if args.force_rerun:
            pending.append(item)
            continue
        existing = cto.load_existing_output(out_file)
        if len(existing) >= args.n_completions:
            continue
        pending.append(item)
    return pending


def init_cross_query_pool(args: argparse.Namespace, experience_path: Any):
    if not args.experience_retrieval.startswith("cross_query"):
        return None, None
    idx2q = cto.load_index_to_question_map(args.input)
    gp = cto.build_global_experience_pool(experience_path, idx2q)
    parsed = cto.build_cross_query_parsed(gp)
    doc_emb = cto.precompute_doc_embeddings_for_parsed(
        parsed, args.retrieval_embedding_model
    )
    return parsed, doc_emb
