#!/usr/bin/env python3
"""
CTO-Rescore++: cluster-representative negative reranking (vLLM-only).

Pipeline per question:
  1. Positive-only generation of N trajectories
  2. Parse final answers and cluster trajectories by math_equal answer
  3. Pick one representative per cluster (shortest parseable / top_pos / centroid)
  4. Teacher-forced negative scoring on K cluster representatives only
  5. Propagate representative neg_score to all cluster members
  6. Rank by S(y) = logp_pos(y) - alpha*logp_neg(y_rep) + beta*log|C(a_y)| + gamma*Q_pos(y)
"""

from __future__ import annotations

import argparse
import logging
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from tqdm import tqdm
from vllm import SamplingParams

_eval_root = Path(__file__).resolve().parent.parent / "eval"
if str(_eval_root) not in sys.path:
    sys.path.insert(0, str(_eval_root))

from evaluation.grader import math_equal
from sal.utils.math import extract_answer

import cto_guided_search as cto
import vllm_efficient_cto_common as vec

logger = logging.getLogger(__name__)


def _parse_final_answer(text: str) -> Optional[Any]:
    try:
        pred = extract_answer(text, "math")
    except Exception:
        return None
    if pred is None or str(pred).strip() == "":
        return None
    return pred


def _answers_equivalent(a: Optional[Any], b: Optional[Any]) -> bool:
    if a is None and b is None:
        return True
    if a is None or b is None:
        return False
    try:
        return bool(math_equal(str(a), str(b), timeout=True))
    except Exception:
        return str(a).strip() == str(b).strip()


def _cluster_by_answer(
    candidates: List[str],
) -> List[Dict[str, Any]]:
    """Group trajectory indices by extracted final answer (math_equal)."""
    parsed: List[Optional[Any]] = [_parse_final_answer(t) for t in candidates]
    clusters: List[Dict[str, Any]] = []

    for i, ans in enumerate(parsed):
        placed = False
        for cluster in clusters:
            if _answers_equivalent(ans, cluster["answer"]):
                cluster["indices"].append(i)
                placed = True
                break
        if not placed:
            clusters.append({"answer": ans, "indices": [i]})

    for cid, cluster in enumerate(clusters):
        cluster["cluster_id"] = cid
        cluster["size"] = len(cluster["indices"])
    return clusters


def _q_pos_score(
    *,
    pos_score: float,
    token_count: int,
    parseable: bool,
    mode: str,
) -> float:
    tokens = max(1, int(token_count))
    if mode == "raw_pos":
        return float(pos_score)
    if mode == "parseable_bonus":
        bonus = 1.0 if parseable else 0.0
        return float(pos_score / tokens + bonus)
    # avg_logprob (default)
    return float(pos_score / tokens)


def _pick_cluster_representative(
    indices: List[int],
    candidates: List[str],
    pos_scores: List[float],
    *,
    rep_selection: str,
    embed_model_path: str,
) -> int:
    if len(indices) == 1:
        return indices[0]

    if rep_selection == "top_pos":
        return max(indices, key=lambda i: pos_scores[i])

    if rep_selection == "shortest_parseable":
        parseable = [i for i in indices if _parse_final_answer(candidates[i]) is not None]
        pool = parseable if parseable else indices
        return min(pool, key=lambda i: (len(candidates[i]), -pos_scores[i]))

    if rep_selection == "centroid":
        texts = [candidates[i].strip()[:4000] for i in indices]
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            return max(indices, key=lambda i: pos_scores[i])

        st = cto._EMB_ST.get(embed_model_path)
        if st is None:
            st = SentenceTransformer(embed_model_path)
            cto._EMB_ST[embed_model_path] = st
        emb = st.encode(texts, normalize_embeddings=True)
        centroid = emb.mean(axis=0)
        centroid = centroid / max(float(np.linalg.norm(centroid)), 1e-12)
        sims = np.dot(emb, centroid)
        best_local = int(np.argmax(sims))
        return indices[best_local]

    raise ValueError(f"Unknown rep_selection: {rep_selection}")


def _select_clusters_to_score(
    clusters: List[Dict[str, Any]],
    pos_scores: List[float],
    *,
    rescore_k: int,
    cluster_priority: str,
) -> List[Dict[str, Any]]:
    k = max(1, int(rescore_k))
    if len(clusters) <= k:
        return clusters

    if cluster_priority == "size":
        order = sorted(clusters, key=lambda c: (-c["size"], c["cluster_id"]))
    elif cluster_priority == "top_pos":
        order = sorted(
            clusters,
            key=lambda c: (
                -max(pos_scores[i] for i in c["indices"]),
                c["cluster_id"],
            ),
        )
    else:
        raise ValueError(f"Unknown cluster_priority: {cluster_priority}")

    selected = order[:k]
    selected.sort(key=lambda c: c["cluster_id"])
    return selected


def run_cto_rescore_plus_for_question(
    llm: Any,
    tokenizer: Any,
    question_text: str,
    experience_data: Dict[str, Any],
    *,
    n_completions: int,
    n_generate: int,
    rescore_k: int,
    rep_selection: str,
    cluster_priority: str,
    max_new_tokens: int,
    alpha: float,
    beta: float,
    gamma: float,
    q_pos_mode: str,
    temperature: float,
    top_p: float,
    top_k: int,
    embed_model_path: str,
    max_model_len: Optional[int],
    vllm_score_batch_size: int,
    vllm_max_score_prompt_tokens: int,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    prompt_pos, prompt_neg, _pitfall_texts = vec.build_pos_neg_prompts(
        tokenizer, question_text, experience_data
    )

    n_gen = max(int(n_generate), int(n_completions), 1)
    gen_params = SamplingParams(
        n=n_gen,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        max_tokens=max_new_tokens,
        logprobs=1,
    )
    gen_out = llm.generate([prompt_pos], gen_params)[0].outputs
    candidates = [o.text or "" for o in gen_out]
    pos_scores = [vec.sum_output_logprobs(getattr(o, "logprobs", None)) for o in gen_out]
    token_counts = [cto._token_count(tokenizer, t) for t in candidates]
    gen_tokens_total = int(sum(len(getattr(o, "token_ids", []) or []) for o in gen_out))

    clusters = _cluster_by_answer(candidates)
    scored_clusters = _select_clusters_to_score(
        clusters,
        pos_scores,
        rescore_k=rescore_k,
        cluster_priority=cluster_priority,
    )

    rep_by_cluster: Dict[int, int] = {}
    for cluster in scored_clusters:
        rep_idx = _pick_cluster_representative(
            cluster["indices"],
            candidates,
            pos_scores,
            rep_selection=rep_selection,
            embed_model_path=embed_model_path,
        )
        rep_by_cluster[cluster["cluster_id"]] = rep_idx

    rep_indices = [rep_by_cluster[c["cluster_id"]] for c in scored_clusters]
    rep_texts = [candidates[i] for i in rep_indices]
    neg_scores_rep: List[float] = []
    if rep_texts:
        neg_scores_rep = vec.score_suffixes_batch(
            llm,
            tokenizer,
            prompt_neg,
            rep_texts,
            max_model_len=max_model_len,
            vllm_score_batch_size=vllm_score_batch_size,
            vllm_max_score_prompt_tokens=vllm_max_score_prompt_tokens,
        )

    cluster_neg: Dict[int, float] = {}
    for cluster, neg in zip(scored_clusters, neg_scores_rep):
        cluster_neg[cluster["cluster_id"]] = float(neg)

    idx_to_cluster: Dict[int, Dict[str, Any]] = {}
    for cluster in clusters:
        for idx in cluster["indices"]:
            idx_to_cluster[idx] = cluster

    neg_scores_map: Dict[int, float] = {}
    for cluster in scored_clusters:
        rep_idx = rep_by_cluster[cluster["cluster_id"]]
        neg_val = cluster_neg[cluster["cluster_id"]]
        neg_scores_map[rep_idx] = neg_val
        for idx in cluster["indices"]:
            neg_scores_map[idx] = neg_val

    scored: List[Tuple[float, int]] = []
    for i, text in enumerate(candidates):
        cluster = idx_to_cluster[i]
        cluster_size = cluster["size"]
        parseable = _parse_final_answer(text) is not None
        neg = neg_scores_map.get(i, 0.0)
        q_pos = _q_pos_score(
            pos_score=pos_scores[i],
            token_count=token_counts[i],
            parseable=parseable,
            mode=q_pos_mode,
        )
        final = (
            float(pos_scores[i])
            - alpha * neg
            + beta * math.log(max(cluster_size, 1))
            + gamma * q_pos
        )
        scored.append((final, i))
    scored.sort(key=lambda x: x[0], reverse=True)
    picked = scored[: max(1, int(n_completions))]

    prompt_pos_len = cto._token_count(tokenizer, prompt_pos)
    prompt_neg_len = cto._token_count(tokenizer, prompt_neg)
    scoring_neg_tokens = 0
    for rep_idx in rep_indices:
        scoring_neg_tokens += cto._token_count(tokenizer, prompt_neg + candidates[rep_idx]) + 1

    processed_tokens_total = (
        prompt_pos_len + gen_tokens_total + scoring_neg_tokens
    )
    compute = {
        "backend": "vllm_cto_rescore_plus",
        "n_generate": int(n_gen),
        "n_clusters": int(len(clusters)),
        "rescore_k": int(len(scored_clusters)),
        "rep_selection": rep_selection,
        "cluster_priority": cluster_priority,
        "neg_branch_fraction": float(len(rep_indices) / max(n_gen, 1)),
        "cluster_sizes": [int(c["size"]) for c in clusters],
        "prompt_pos_tokens": int(prompt_pos_len),
        "prompt_neg_tokens": int(prompt_neg_len),
        "candidate_generation_new_tokens": int(gen_tokens_total),
        "processed_tokens_total": int(processed_tokens_total),
        "processed_tokens_per_rollout_mean": int(
            processed_tokens_total / max(n_completions, 1)
        ),
        "generated_tokens_mean": float(gen_tokens_total / max(n_gen, 1)),
        "alpha": float(alpha),
        "beta": float(beta),
        "gamma": float(gamma),
        "q_pos_mode": q_pos_mode,
    }

    completions: List[Dict[str, Any]] = []
    for score, idx in picked:
        text_raw = candidates[idx]
        final_text, reasoning = cto.extract_thinking(text_raw)
        cluster = idx_to_cluster[idx]
        rep_idx = rep_by_cluster.get(cluster["cluster_id"])
        completions.append(
            {
                "text": final_text,
                "reasoning_content": reasoning,
                "tokens": token_counts[idx],
                "finish_reason": "cluster_rescore_rerank",
                "cto_score": score,
                "pos_score": pos_scores[idx],
                "neg_score": neg_scores_map.get(idx, 0.0),
                "q_pos_score": _q_pos_score(
                    pos_score=pos_scores[idx],
                    token_count=token_counts[idx],
                    parseable=_parse_final_answer(text_raw) is not None,
                    mode=q_pos_mode,
                ),
                "cluster_id": cluster["cluster_id"],
                "cluster_size": cluster["size"],
                "parsed_answer": cluster["answer"],
                "is_cluster_representative": rep_idx == idx if rep_idx is not None else False,
                "rescored": idx in neg_scores_map,
            }
        )
    return completions, compute


def main() -> None:
    parser = argparse.ArgumentParser(
        description="CTO-Rescore++: cluster-representative negative reranking (vLLM)"
    )
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--experience-dir", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--n-completions", type=int, default=1)
    parser.add_argument("--n-generate", type=int, default=32)
    parser.add_argument(
        "--rescore-k",
        type=int,
        default=8,
        help="Max number of answer clusters to negative-score (typically 3-8).",
    )
    parser.add_argument(
        "--rep-selection",
        type=str,
        default="top_pos",
        choices=["shortest_parseable", "top_pos", "centroid"],
    )
    parser.add_argument(
        "--cluster-priority",
        type=str,
        default="size",
        choices=["size", "top_pos"],
        help="When clusters exceed --rescore-k, which clusters get negative scoring.",
    )
    parser.add_argument("--alpha", type=float, default=0.7)
    parser.add_argument("--beta", type=float, default=0.3)
    parser.add_argument("--gamma", type=float, default=0.1)
    parser.add_argument(
        "--q-pos-mode",
        type=str,
        default="avg_logprob",
        choices=["avg_logprob", "raw_pos", "parseable_bonus"],
    )
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
    logger.info(
        "CTO-Rescore++ N=%d K=%d rep=%s | %s | %d questions",
        args.n_generate,
        args.rescore_k,
        args.rep_selection,
        task_type,
        len(questions),
    )

    cross_query_parsed, cross_query_doc_emb = vec.init_cross_query_pool(args, experience_path)
    pending = vec.prepare_pending_questions(args, questions, output_path)
    if not pending:
        logger.info("All questions already completed.")
        return

    llm, tokenizer, max_model_len = vec.init_vllm(args)

    for item in tqdm(pending, desc="CTO-Rescore++", unit="q"):
        orig_idx = item["index"]
        question_text = item.get("question", "")
        experience_data = vec.load_experience_for_question(
            args,
            experience_path,
            orig_idx,
            question_text,
            cross_query_parsed,
            cross_query_doc_emb,
        )
        if not experience_data:
            logger.warning("No experience for question %s, skipping.", orig_idx)
            continue

        completions, compute = run_cto_rescore_plus_for_question(
            llm,
            tokenizer,
            question_text,
            experience_data,
            n_completions=args.n_completions,
            n_generate=args.n_generate,
            rescore_k=args.rescore_k,
            rep_selection=args.rep_selection,
            cluster_priority=args.cluster_priority,
            max_new_tokens=args.max_tokens,
            alpha=args.alpha,
            beta=args.beta,
            gamma=args.gamma,
            q_pos_mode=args.q_pos_mode,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            embed_model_path=args.retrieval_embedding_model,
            max_model_len=max_model_len,
            vllm_score_batch_size=args.vllm_score_batch_size,
            vllm_max_score_prompt_tokens=args.vllm_max_score_prompt_tokens,
        )
        cto.save_result(
            output_path,
            orig_idx,
            item,
            completions,
            extra_fields={
                "cto_rescore_plus_compute": compute,
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
