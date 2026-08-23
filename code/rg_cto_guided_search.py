#!/usr/bin/env python3
"""
Risk-Gated CTO (RG-CTO): confidence-gated contrastive decoding.

  s_i = l_pos,i - alpha_r * l_neg,i
  alpha_r = alpha_0 * g^(r)

where g^(r) aggregates per-pitfall weights w(e) = clip(u(e)*l(e,q)*(1-c(e)), 0, 1)
after filtering w(e) < delta.
"""

from __future__ import annotations

import argparse
import copy
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from tqdm import tqdm
from vllm import SamplingParams

_eval_root = Path(__file__).resolve().parent.parent / "eval"
if str(_eval_root) not in sys.path:
    sys.path.insert(0, str(_eval_root))

import cto_guided_search as cto
import cto_rescore_plus_guided_search as rescore_plus
import rg_cto_common as rgc
import vllm_efficient_cto_common as vec

logger = logging.getLogger(__name__)


def _load_raw_records(experience_dir: Path, original_idx: int) -> List[Dict[str, Any]]:
    experience_file = experience_dir / f"{original_idx}.jsonl"
    if not experience_file.exists():
        return []
    return cto.load_jsonl(str(experience_file))


def _parse_coverage_meta(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    for record in records:
        cd = cto._parse_record_content_dict(record)
        if not cd:
            continue
        meta = cd.get("coverage_meta")
        if isinstance(meta, dict):
            return meta
        if cd.get("no_reliable_negative_evidence"):
            return {"no_reliable_negative_evidence": True}
    return {}


def _load_experience_records(
    experience_dir: Path,
    original_idx: int,
    n_completions_to_use: int,
    question_text: str,
    args: argparse.Namespace,
    cross_query_parsed,
    cross_query_doc_emb,
) -> Tuple[Optional[Dict[str, Any]], List[Dict[str, Any]]]:
    records = _load_raw_records(experience_dir, original_idx)
    if not records:
        return None, []

    data = cto.load_and_aggregate_raw_experiences(
        experience_dir,
        original_idx,
        n_completions_to_use,
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
    if data is None:
        data = {"verified_propositions": [], "critical_pitfalls": []}
    cov = _parse_coverage_meta(records)
    if cov:
        data["coverage_meta"] = cov
    if cov.get("no_reliable_negative_evidence"):
        data["no_reliable_negative_evidence"] = True
    return data, records


def _run_pilot_phase(
    llm: Any,
    tokenizer: Any,
    *,
    prompt_pos: str,
    prompt_neg: str,
    pilot_n: int,
    max_pilot_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
    alpha_0: float,
    max_model_len: Optional[int],
    vllm_score_batch_size: int,
    vllm_max_score_prompt_tokens: int,
) -> Dict[str, Any]:
    n = max(2, int(pilot_n))
    gen_params = SamplingParams(
        n=n,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        max_tokens=max_pilot_tokens,
        logprobs=1,
    )
    gen_out = llm.generate([prompt_pos], gen_params)[0].outputs
    candidates = [o.text or "" for o in gen_out]
    pos_scores = [vec.sum_output_logprobs(getattr(o, "logprobs", None)) for o in gen_out]

    clusters = rescore_plus._cluster_by_answer(candidates)
    sorted_clusters = sorted(clusters, key=lambda c: (-c["size"], c["cluster_id"]))
    top_cluster = sorted_clusters[0] if sorted_clusters else {"indices": [], "size": 0}
    top_cluster_texts = [candidates[i] for i in top_cluster.get("indices", [])[:3]]
    cluster_majority = float(top_cluster.get("size", 0) / max(len(candidates), 1))

    all_neg: List[float] = []
    if candidates:
        all_neg = vec.score_suffixes_batch(
            llm,
            tokenizer,
            prompt_neg,
            candidates,
            max_model_len=max_model_len,
            vllm_score_batch_size=vllm_score_batch_size,
            vllm_max_score_prompt_tokens=vllm_max_score_prompt_tokens,
        )

    return {
        "pos_scores": pos_scores,
        "neg_scores": all_neg,
        "top_cluster_texts": top_cluster_texts,
        "cluster_majority": cluster_majority,
        "alpha_0": float(alpha_0),
        "pilot_candidates": len(candidates),
        "n_clusters": len(clusters),
    }


def _apply_gated_negatives(
    experience_data: Dict[str, Any],
    gate_factors: Dict[str, Any],
) -> Dict[str, Any]:
    gated = copy.deepcopy(experience_data)
    filtered = gate_factors.get("filtered_pitfalls") or []
    gated["critical_pitfalls"] = filtered
    gated["rg_cto_gate"] = float(gate_factors.get("gate", 0.0))
    return gated


def run_rg_cto_for_question(
    llm: Any,
    tokenizer: Any,
    question_text: str,
    experience_data: Dict[str, Any],
    experience_records: List[Dict[str, Any]],
    *,
    n_completions: int,
    candidate_k: int,
    max_new_tokens: int,
    alpha_0: float,
    alpha_floor: float,
    gate_delta: float,
    min_pitfall_support: int,
    temperature: float,
    top_p: float,
    top_k: int,
    embed_model_path: str,
    max_model_len: Optional[int],
    vllm_score_batch_size: int,
    vllm_max_score_prompt_tokens: int,
    pilot_meta: Optional[Dict[str, Any]] = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    gate_factors = rgc.compute_item_weights(
        experience_records,
        question_text,
        embed_model_path=embed_model_path,
        min_support=min_pitfall_support,
        delta=gate_delta,
        pilot_meta=pilot_meta,
    )

    coverage_meta = experience_data.get("coverage_meta") or {}
    no_reliable_neg = bool(
        experience_data.get("no_reliable_negative_evidence")
        or coverage_meta.get("no_reliable_negative_evidence")
    )

    gated_data = _apply_gated_negatives(experience_data, gate_factors)
    prompt_pos, prompt_neg, pitfall_texts = vec.build_pos_neg_prompts(
        tokenizer, question_text, gated_data
    )

    alpha_r = rgc.effective_alpha(alpha_0, gate_factors, alpha_floor=alpha_floor)
    if no_reliable_neg or not pitfall_texts:
        alpha_r = 0.0
        gate_factors = dict(gate_factors)
        gate_factors["gate"] = 0.0
        gate_factors["alpha_multiplier"] = 0.0

    cand_k = max(int(candidate_k), int(n_completions), 1)
    gen_params = SamplingParams(
        n=cand_k,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        max_tokens=max_new_tokens,
    )
    gen_out = llm.generate([prompt_pos], gen_params)[0].outputs
    candidates = [o.text or "" for o in gen_out]
    gen_tokens_total = int(sum(len(getattr(o, "token_ids", []) or []) for o in gen_out))

    prompt_pos_len = cto._token_count(tokenizer, prompt_pos)
    prompt_neg_len = cto._token_count(tokenizer, prompt_neg)
    vllm_cap = max_model_len if max_model_len is not None else cto._get_vllm_max_model_len(llm)
    if vllm_cap is None:
        vllm_cap = 131072
    score_prompt_budget = max(1, int(vllm_cap) - 1)
    if int(vllm_max_score_prompt_tokens) > 0:
        score_prompt_budget = min(score_prompt_budget, int(vllm_max_score_prompt_tokens))

    full_prompts_pos: List[str] = []
    full_prompts_neg: List[str] = []
    suffix_pos: List[int] = []
    suffix_neg: List[int] = []
    for c in candidates:
        fp, _ppt, sp = cto._truncate_concat_for_vllm(
            tokenizer, prompt_pos, c, score_prompt_budget
        )
        fn, _pnt, sn = cto._truncate_concat_for_vllm(
            tokenizer, prompt_neg, c, score_prompt_budget
        )
        full_prompts_pos.append(fp)
        full_prompts_neg.append(fn)
        suffix_pos.append(sp)
        suffix_neg.append(sn)

    score_params = SamplingParams(
        max_tokens=1,
        n=1,
        temperature=0.0,
        top_p=1.0,
        top_k=-1,
        prompt_logprobs=1,
    )

    scored: List[Tuple[float, int]] = []
    score_bs = max(1, int(vllm_score_batch_size))
    for start in range(0, len(candidates), score_bs):
        end = min(start + score_bs, len(candidates))
        pos_batch = full_prompts_pos[start:end]
        neg_batch = full_prompts_neg[start:end]
        outs_pos = llm.generate(pos_batch, score_params)
        outs_neg = llm.generate(neg_batch, score_params)
        for bi, i in enumerate(range(start, end)):
            plp_pos = getattr(outs_pos[bi], "prompt_logprobs", None)
            plp_neg = getattr(outs_neg[bi], "prompt_logprobs", None)
            logp_pos = cto._sum_prompt_logprob_for_suffix(plp_pos, suffix_pos[i])
            logp_neg = cto._sum_prompt_logprob_for_suffix(plp_neg, suffix_neg[i])
            scored.append((float(logp_pos - alpha_r * logp_neg), i))

    scored.sort(key=lambda x: x[0], reverse=True)
    picked = scored[: max(1, int(n_completions))]

    scoring_pos_tokens = sum(cto._token_count(tokenizer, fp) + 1 for fp in full_prompts_pos)
    scoring_neg_tokens = sum(cto._token_count(tokenizer, fn) + 1 for fn in full_prompts_neg)
    processed_tokens_total = prompt_pos_len + gen_tokens_total + scoring_pos_tokens + scoring_neg_tokens

    compute = {
        "backend": "vllm_rg_cto",
        "alpha_0": float(alpha_0),
        "alpha_r": float(alpha_r),
        "gate": float(gate_factors.get("gate", 0.0)),
        "gate_delta": float(gate_delta),
        "n_pitfalls_kept": int(gate_factors.get("n_pitfalls_kept", 0)),
        "n_pitfalls_total": int(gate_factors.get("n_pitfalls_total", 0)),
        "pilot_risk": float(gate_factors.get("pilot_risk", 0.0)),
        "candidate_k": int(len(candidates)),
        "prompt_pos_tokens": int(prompt_pos_len),
        "prompt_neg_tokens": int(prompt_neg_len),
        "candidate_generation_new_tokens": int(gen_tokens_total),
        "processed_tokens_total": int(processed_tokens_total),
        "processed_tokens_per_rollout_mean": int(
            processed_tokens_total / max(n_completions, 1)
        ),
        "generated_tokens_mean": float(gen_tokens_total / max(len(candidates), 1)),
        "pilot_meta": pilot_meta or {},
        "rg_cto_factors": gate_factors,
        "coverage_meta": coverage_meta,
        "no_reliable_negative_evidence": bool(no_reliable_neg),
        "mode": "positive_only" if alpha_r <= 0.0 else "rg_cto",
    }

    completions: List[Dict[str, Any]] = []
    for score, idx in picked:
        text_raw = candidates[idx]
        final_text, reasoning = cto.extract_thinking(text_raw)
        completions.append(
            {
                "text": final_text,
                "reasoning_content": reasoning,
                "tokens": cto._token_count(tokenizer, text_raw),
                "finish_reason": "rg_cto_rerank",
                "cto_score": score,
                "rg_cto_alpha_r": float(alpha_r),
            }
        )
    return completions, compute


def main() -> None:
    parser = argparse.ArgumentParser(
        description="RG-CTO: Risk-Gated Contrastive Trajectory Optimization (vLLM)"
    )
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--experience-dir", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--n-completions", type=int, default=1)
    parser.add_argument("--candidate-k", type=int, default=None)
    parser.add_argument("--alpha", type=float, default=0.7, help="alpha_0 base suppression")
    parser.add_argument(
        "--alpha-floor",
        type=float,
        default=0.02,
        help="Below this alpha_r, treat as positive-only",
    )
    parser.add_argument(
        "--gate-delta",
        type=float,
        default=0.15,
        help="Filter negative items with w(e) < delta",
    )
    parser.add_argument("--min-pitfall-support", type=int, default=2)
    parser.add_argument(
        "--pilot-n",
        type=int,
        default=4,
        help="Pilot rollouts for conflict-risk proxy",
    )
    parser.add_argument("--max-pilot-tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--start-idx", type=int, default=0)
    parser.add_argument("--end-idx", type=int, default=None)
    parser.add_argument("--force-rerun", action="store_true")
    vec.add_experience_args(parser)
    vec.add_vllm_args(parser)
    args = parser.parse_args()

    task_type = vec.setup_task_prefixes(args)
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)
    experience_path = Path(args.experience_dir)

    questions = cto.load_jsonl(args.input)
    end_idx = args.end_idx if args.end_idx is not None else len(questions)
    questions = questions[args.start_idx:end_idx]
    cand_k = args.candidate_k if args.candidate_k is not None else args.n_completions

    logger.info(
        "RG-CTO alpha_0=%.2f delta=%.2f pilot_n=%d min_support=%d | %s | %d questions",
        args.alpha,
        args.gate_delta,
        args.pilot_n,
        args.min_pitfall_support,
        task_type,
        len(questions),
    )

    cross_query_parsed, cross_query_doc_emb = vec.init_cross_query_pool(args, experience_path)
    pending = vec.prepare_pending_questions(args, questions, output_path)
    if not pending:
        logger.info("All questions already completed.")
        return

    llm, tokenizer, max_model_len = vec.init_vllm(args)

    for item in tqdm(pending, desc="RG-CTO", unit="q"):
        orig_idx = item["index"]
        question_text = item.get("question", "")
        experience_data, records = _load_experience_records(
            experience_path,
            orig_idx,
            args.n_experience_completions,
            question_text,
            args,
            cross_query_parsed,
            cross_query_doc_emb,
        )
        if not records:
            logger.warning("No experience file for question %s, skipping.", orig_idx)
            continue

        prompt_pos, prompt_neg, _ = vec.build_pos_neg_prompts(
            tokenizer, question_text, experience_data
        )
        pilot_meta = _run_pilot_phase(
            llm,
            tokenizer,
            prompt_pos=prompt_pos,
            prompt_neg=prompt_neg,
            pilot_n=args.pilot_n,
            max_pilot_tokens=args.max_pilot_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            alpha_0=args.alpha,
            max_model_len=max_model_len,
            vllm_score_batch_size=args.vllm_score_batch_size,
            vllm_max_score_prompt_tokens=args.vllm_max_score_prompt_tokens,
        )

        completions, compute = run_rg_cto_for_question(
            llm,
            tokenizer,
            question_text,
            experience_data,
            records,
            n_completions=args.n_completions,
            candidate_k=cand_k,
            max_new_tokens=args.max_tokens,
            alpha_0=args.alpha,
            alpha_floor=args.alpha_floor,
            gate_delta=args.gate_delta,
            min_pitfall_support=args.min_pitfall_support,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            embed_model_path=args.retrieval_embedding_model,
            max_model_len=max_model_len,
            vllm_score_batch_size=args.vllm_score_batch_size,
            vllm_max_score_prompt_tokens=args.vllm_max_score_prompt_tokens,
            pilot_meta=pilot_meta,
        )
        cto.save_result(
            output_path,
            orig_idx,
            item,
            completions,
            extra_fields={
                "rg_cto_compute": compute,
                "cto_compute": compute,
            },
        )

    logger.info("Done. Results in %s", args.output)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler()],
    )
    main()
