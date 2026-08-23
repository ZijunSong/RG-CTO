#!/usr/bin/env python3
"""
CTO (Contrastive Trajectory Optimization) Experience-Guided Search.

Uses dual-stream decoding: P_pos (question + verified propositions) and P_neg (question + induced pitfalls).
At each step: logits_CTO = logits_primary - alpha * logits_neg (masked to top-K of the primary logits),
unless an ablated branch-mode disables the penalty or merges bullets into one prompt-only policy.
Output format is identical to RSE (experience_guided_search.py) for compatibility with distillation and eval.
"""

import json
import argparse
import os
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
import logging
from tqdm import tqdm

import numpy as np
import torch

try:
    from transformers import AutoTokenizer
except ImportError:
    raise ImportError("CTO requires transformers (tokenizer only): pip install transformers")

try:
    from vllm import LLM, SamplingParams
except ImportError:
    LLM = None
    SamplingParams = None

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

# Local CrossEncoder snapshot (avoids Hugging Face Hub download each run; see download_models.py).
_DEFAULT_RETRIEVAL_RERANK_MODEL = "/data/ppnm/models/cross-encoder-ms-marco-MiniLM-L-6-v2"

# ---------------------------------------------------------------------------
# Prompt building (P_pos: propositions only; P_neg: inducing template + pitfalls)
# ---------------------------------------------------------------------------

P_POS_SYSTEM_PREFIX = """You are an advanced mathematical solver augmented with verified intermediate results.
Use the following propositions as anchors when they accelerate your reasoning. Verify any premise before use.

### Propositions (Verify before use):
"""

P_NEG_SYSTEM_PREFIX = """Please try to solve the following using these incorrect approaches or dead ends. You must follow at least one of them:

"""


P_BOTH_SYSTEM_FALLBACK = "Please reason step by step, and put your final answer within \\boxed{}."


def _build_prompt_both_system(props_str: str, pitfalls_str: str) -> str:
    """Positive + negative experience in one prompt; no logits-level CTO (plain sampling)."""
    blocks: List[str] = [
        (
            "You are an advanced mathematical solver augmented with distilled experience from "
            "prior attempts on related problems.\n\n"
            "### Verified propositions (verify before relying on them):\n"
        )
    ]
    if props_str.strip():
        blocks.append(props_str.strip())
    else:
        blocks.append("(none provided)\n")
    blocks.append(
        "\n### Critical pitfalls and dead ends (warning — do not repeat these mistakes):\n"
    )
    if pitfalls_str.strip():
        blocks.append(pitfalls_str.strip())
    else:
        blocks.append("(none provided)\n")
    blocks.append(
        "\nSolve the user problem carefully. Put your final answer within \\boxed{}."
    )
    return "".join(blocks)


def load_jsonl(file_path: str) -> List[Dict[str, Any]]:
    data = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                try:
                    data.append(json.loads(line.strip()))
                except json.JSONDecodeError:
                    continue
    return data


_EMB_ST: Dict[str, Any] = {}
_CE: Dict[str, Any] = {}


def _parse_record_content_dict(record: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    content = record.get("experience_parsed") or record.get("experience_raw")
    if isinstance(content, dict):
        return content
    if isinstance(content, str):
        try:
            clean_content = content.replace("```json", "").replace("```", "").strip()
            return json.loads(clean_content)
        except json.JSONDecodeError:
            return None
    return None


def _record_to_retrieval_doc(question_text: str, content_dict: Dict[str, Any]) -> str:
    """Concatenate question (for context) + propositions/pitfalls for embedding / reranking."""
    props = content_dict.get("verified_propositions", []) or []
    pitfalls = content_dict.get("critical_pitfalls", []) or []
    parts = [question_text.strip()[:2000], "PROPOSITIONS:"]
    parts.extend([str(p) for p in props[:24] if isinstance(p, str)])
    parts.append("PITFALLS:")
    parts.extend([str(p) for p in pitfalls[:24] if isinstance(p, str)])
    return "\n".join(parts)[:12000]


def _select_from_parsed_records(
    parsed: List[Tuple[Dict[str, Any], str]],
    query: str,
    n_take: int,
    retrieval: str,
    embed_model_path: str,
    rerank_model_name: Optional[str],
    rerank_pool_mult: int,
    doc_embeddings: Optional[np.ndarray] = None,
) -> List[Dict[str, Any]]:
    """Core bi-encoder (+ optional cross-encoder) ranking over (record, doc) pairs."""
    if not parsed:
        return []
    if len(parsed) <= n_take:
        return [p[0] for p in parsed]

    try:
        from sentence_transformers import SentenceTransformer
        from sentence_transformers import CrossEncoder
    except ImportError:
        logger.warning(
            "sentence-transformers not installed; fallback to first-n experience records. "
            "pip install sentence-transformers"
        )
        return [p[0] for p in parsed[:n_take]]

    q = query.strip()[:4000]
    records_only = [p[0] for p in parsed]
    docs = [p[1] for p in parsed]

    st = _EMB_ST.get(embed_model_path)
    if st is None:
        st = SentenceTransformer(embed_model_path)
        _EMB_ST[embed_model_path] = st
    qv = st.encode([q], normalize_embeddings=True)
    if doc_embeddings is not None:
        dv = doc_embeddings
        if dv.shape[0] != len(docs):
            logger.warning("doc_embeddings length mismatch; re-encoding docs")
            dv = st.encode(docs, normalize_embeddings=True)
    else:
        dv = st.encode(docs, normalize_embeddings=True)
    sim = np.dot(dv, qv.T).flatten()

    if retrieval in ("embedding", "cross_query_embedding"):
        order = np.argsort(-sim)[:n_take]
        return [records_only[int(i)] for i in order]

    if retrieval in ("embedding_rerank", "cross_query_embedding_rerank"):
        pool = min(len(docs), max(n_take * max(2, rerank_pool_mult), n_take))
        pre_idx = np.argsort(-sim)[:pool]

        rname = rerank_model_name or _DEFAULT_RETRIEVAL_RERANK_MODEL
        ce = _CE.get(rname)
        if ce is None:
            ce = CrossEncoder(rname)
            _CE[rname] = ce

        pairs = [[q, docs[int(i)]] for i in pre_idx]
        rr_scores = np.array(ce.predict(pairs))
        k = min(n_take, len(rr_scores))
        best_local = np.argsort(-rr_scores)[:k]
        chosen = [records_only[int(pre_idx[int(j)])] for j in best_local]
        return chosen

    return [records_only[i] for i in range(min(n_take, len(records_only)))]


def select_experience_records_for_question(
    raw_data: List[Dict[str, Any]],
    question_text: str,
    n_take: int,
    retrieval: str,
    embed_model_path: str,
    rerank_model_name: Optional[str],
    rerank_pool_mult: int,
) -> List[Dict[str, Any]]:
    """Order-preserving subset: take top-n by relevance to question (or first-n)."""
    if retrieval == "first" or not raw_data:
        return raw_data[:n_take]

    parsed: List[Tuple[Dict[str, Any], str]] = []
    for record in raw_data:
        cd = _parse_record_content_dict(record)
        if not cd:
            continue
        doc = _record_to_retrieval_doc(question_text, cd)
        parsed.append((record, doc))

    if not parsed:
        return raw_data[:n_take]

    inner_mode = "embedding_rerank" if retrieval == "embedding_rerank" else "embedding"

    return _select_from_parsed_records(
        parsed,
        question_text,
        n_take,
        inner_mode,
        embed_model_path,
        rerank_model_name,
        rerank_pool_mult,
        doc_embeddings=None,
    )


def load_index_to_question_map(question_jsonl: str) -> Dict[int, str]:
    """Map dataset index -> question text (for cross-query source labels)."""
    data = load_jsonl(question_jsonl)
    out: Dict[int, str] = {}
    for i, item in enumerate(data):
        idx = int(item.get("index", i))
        out[idx] = item.get("question", "") or ""
    return out


def build_global_experience_pool(
    experience_dir: Path,
    index_to_question: Dict[int, str],
) -> List[Tuple[Dict[str, Any], str, int]]:
    """
    Collect all distilled records from every *.jsonl under experience_dir.
    Each item is (record, source_question_text, source_problem_idx) where
    source_question_text is the HMMT/jsonl question that produced that experience.
    """
    pool: List[Tuple[Dict[str, Any], str, int]] = []
    if not experience_dir.is_dir():
        return pool
    for p in sorted(experience_dir.glob("*.jsonl")):
        try:
            idx = int(p.stem)
        except ValueError:
            continue
        src_q = index_to_question.get(idx, "")
        for record in load_jsonl(str(p)):
            if isinstance(record, dict):
                pool.append((record, src_q, idx))
    return pool


def build_cross_query_parsed(
    global_pool: List[Tuple[Dict[str, Any], str, int]],
) -> List[Tuple[Dict[str, Any], str, int]]:
    """
    Build (record, doc, source_problem_idx) using **source** question text in the doc.
    """
    parsed: List[Tuple[Dict[str, Any], str, int]] = []
    for record, src_q, src_idx in global_pool:
        cd = _parse_record_content_dict(record)
        if not cd:
            continue
        doc = _record_to_retrieval_doc(src_q, cd)
        parsed.append((record, doc, src_idx))
    return parsed


def _parsed_records_for_selection(
    parsed_with_idx: List[Tuple[Dict[str, Any], str, int]],
    *,
    exclude_problem_idx: Optional[int] = None,
) -> Tuple[List[Tuple[Dict[str, Any], str]], Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Strip source idx for _select_from_parsed_records; optionally drop current problem
    and align precomputed doc embeddings via a boolean keep-mask.
    """
    if exclude_problem_idx is None:
        pairs = [(r, d) for r, d, _ in parsed_with_idx]
        return pairs, None, None
    keep = np.array([src_idx != exclude_problem_idx for _, _, src_idx in parsed_with_idx])
    pairs = [(r, d) for (r, d, src_idx) in parsed_with_idx if src_idx != exclude_problem_idx]
    return pairs, keep, keep


def precompute_doc_embeddings_for_parsed(
    parsed: List[Tuple[Dict[str, Any], str, int]],
    embed_model_path: str,
) -> np.ndarray:
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        raise ImportError("sentence-transformers required for cross-query retrieval")
    st = _EMB_ST.get(embed_model_path)
    if st is None:
        st = SentenceTransformer(embed_model_path)
        _EMB_ST[embed_model_path] = st
    docs = [p[1] for p in parsed]
    return st.encode(docs, normalize_embeddings=True)


def load_and_aggregate_global_neg_experiences(
    experience_dir: Path,
    original_idx: int,
    n_completions_to_use: int,
    question_text: str = "",
    embed_model_path: str = "",
    rerank_model_name: Optional[str] = None,
    rerank_pool_mult: int = 4,
    *,
    cross_query_parsed: Optional[List[Tuple[Dict[str, Any], str, int]]] = None,
    cross_query_doc_emb: Optional[np.ndarray] = None,
    max_aggregated_propositions: int = 0,
    max_aggregated_pitfalls: int = 0,
    experience_aggregation: str = "legacy_sorted",
    aggregation_semantic_threshold: float = 0.82,
    aggregation_recency_head_fraction: float = 0.55,
) -> Optional[Dict[str, List[str]]]:
    """
    Global-Neg CTO experience merge:
    - Positive branch: B_pos_local from current problem's {idx}.jsonl (embedding_rerank).
    - Negative branch: B_neg_global from all other problems' pitfalls (cross-query rerank),
      excluding every record whose source_problem_idx == original_idx.
    """
    emb = embed_model_path or "/data/ppnm/models/all-MiniLM-L6-v2"

    experience_file = experience_dir / f"{original_idx}.jsonl"
    if not experience_file.exists():
        return None
    raw_data = load_jsonl(str(experience_file))
    if not raw_data:
        return None

    selected_pos = select_experience_records_for_question(
        raw_data,
        question_text,
        n_completions_to_use,
        "embedding_rerank",
        emb,
        rerank_model_name,
        rerank_pool_mult,
    )

    if not cross_query_parsed:
        logger.warning(
            "global_neg: empty cross-query pool; negative branch has no pitfalls for q=%s",
            original_idx,
        )
        selected_neg: List[Dict[str, Any]] = []
    else:
        neg_pairs, keep_mask, _ = _parsed_records_for_selection(
            cross_query_parsed,
            exclude_problem_idx=original_idx,
        )
        neg_doc_emb = None
        if cross_query_doc_emb is not None and keep_mask is not None:
            neg_doc_emb = cross_query_doc_emb[keep_mask]
        selected_neg = _select_from_parsed_records(
            neg_pairs,
            question_text,
            n_completions_to_use,
            "cross_query_embedding_rerank",
            emb,
            rerank_model_name,
            rerank_pool_mult,
            doc_embeddings=neg_doc_emb,
        )

    prop_sorted, _ = apply_experience_aggregation(
        selected_pos,
        max_aggregated_propositions,
        0,
        experience_aggregation,
        emb,
        aggregation_semantic_threshold,
        aggregation_recency_head_fraction,
    )
    _, pit_sorted = apply_experience_aggregation(
        selected_neg,
        0,
        max_aggregated_pitfalls,
        experience_aggregation,
        emb,
        aggregation_semantic_threshold,
        aggregation_recency_head_fraction,
    )
    return {
        "critical_pitfalls": pit_sorted,
        "verified_propositions": prop_sorted,
    }


def _ordered_unique_strings_from_records(
    records: List[Dict[str, Any]],
    field: str,
) -> List[str]:
    """First-seen order within retrieval-ordered records (flat_top-k baseline order)."""
    seen: set = set()
    out: List[str] = []
    for record in records:
        content_dict = _parse_record_content_dict(record)
        if not content_dict:
            continue
        for p in content_dict.get(field, []) or []:
            if isinstance(p, str) and p not in seen:
                seen.add(p)
                out.append(p)
    return out


def _maybe_truncate_strings(xs: List[str], max_k: int) -> List[str]:
    if max_k and max_k > 0:
        return xs[:max_k]
    return xs


def _semantic_merge_strings(
    texts: List[str],
    embed_model_path: str,
    threshold: float,
) -> List[str]:
    """
    Greedy clustering by cosine similarity on sentence embeddings; each cluster becomes one line
    (longest string kept as representative).
    """
    if len(texts) <= 1:
        return list(texts)
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        logger.warning("sentence-transformers missing; semantic_merge is a no-op")
        return list(texts)
    st = _EMB_ST.get(embed_model_path)
    if st is None:
        st = SentenceTransformer(embed_model_path)
        _EMB_ST[embed_model_path] = st
    emb = st.encode(texts, normalize_embeddings=True)
    n = len(texts)
    order = sorted(range(n), key=lambda i: len(texts[i]), reverse=True)
    used = [False] * n
    merged: List[str] = []
    for i in order:
        if used[i]:
            continue
        cluster = [i]
        used[i] = True
        for j in range(n):
            if used[j] or i == j:
                continue
            if float(np.dot(emb[i], emb[j])) >= threshold:
                used[j] = True
                cluster.append(j)
        rep = max([texts[k] for k in cluster], key=len)
        merged.append(rep)
    return merged


def _aggregate_recency_field(
    selected_data: List[Dict[str, Any]],
    field: str,
    max_k: int,
    head_slot_fraction: float,
) -> List[str]:
    """
    Split retrieval-ordered records at the midpoint; allocate ~head_slot_fraction of the max_k
    budget to unique strings from the head, remainder from the tail (within max_k total).
    """
    if not max_k or max_k <= 0:
        return _ordered_unique_strings_from_records(selected_data, field)
    n = len(selected_data)
    if n == 0:
        return []
    cut = max(1, n // 2)
    head_r = selected_data[:cut]
    tail_r = selected_data[cut:]
    kh = min(max_k, max(0, int(round(max_k * head_slot_fraction))))

    seen: set = set()
    out: List[str] = []

    def add_from(records: List[Dict[str, Any]], target_total: int) -> None:
        for record in records:
            if len(out) >= target_total:
                return
            cd = _parse_record_content_dict(record)
            if not cd:
                continue
            for p in cd.get(field, []) or []:
                if not isinstance(p, str) or p in seen:
                    continue
                seen.add(p)
                out.append(p)
                if len(out) >= target_total:
                    return

    add_from(head_r, kh)
    add_from(tail_r, max_k)
    return out[:max_k]


def apply_experience_aggregation(
    selected_data: List[Dict[str, Any]],
    max_prop: int,
    max_pit: int,
    mode: str,
    embed_model_path: str,
    semantic_threshold: float,
    recency_head_fraction: float,
) -> Tuple[List[str], List[str]]:
    if mode == "legacy_sorted":
        # Original behavior before aggregation-mode refactor: global set union, sorted(), truncate.
        aggregated_pitfalls: set = set()
        aggregated_propositions: set = set()
        for record in selected_data:
            content_dict = _parse_record_content_dict(record)
            if not content_dict:
                continue
            for p in content_dict.get("critical_pitfalls", []):
                if isinstance(p, str):
                    aggregated_pitfalls.add(p)
            for p in content_dict.get("verified_propositions", []):
                if isinstance(p, str):
                    aggregated_propositions.add(p)
        pit_sorted = sorted(list(aggregated_pitfalls))
        prop_sorted = sorted(list(aggregated_propositions))
        props = _maybe_truncate_strings(prop_sorted, max_prop)
        pits = _maybe_truncate_strings(pit_sorted, max_pit)
        return props, pits
    if mode == "flat_top_k":
        props = _ordered_unique_strings_from_records(selected_data, "verified_propositions")
        pits = _ordered_unique_strings_from_records(selected_data, "critical_pitfalls")
        props = _maybe_truncate_strings(props, max_prop)
        pits = _maybe_truncate_strings(pits, max_pit)
        return props, pits
    if mode == "semantic_merge":
        props = _ordered_unique_strings_from_records(selected_data, "verified_propositions")
        pits = _ordered_unique_strings_from_records(selected_data, "critical_pitfalls")
        props = _semantic_merge_strings(props, embed_model_path, semantic_threshold)
        pits = _semantic_merge_strings(pits, embed_model_path, semantic_threshold)
        props = _maybe_truncate_strings(props, max_prop)
        pits = _maybe_truncate_strings(pits, max_pit)
        return props, pits
    if mode == "recency_aware":
        props = _aggregate_recency_field(
            selected_data, "verified_propositions", max_prop, recency_head_fraction
        )
        pits = _aggregate_recency_field(
            selected_data, "critical_pitfalls", max_pit, recency_head_fraction
        )
        return props, pits
    raise ValueError(
        f"Unknown experience aggregation mode: {mode} "
        f"(expected legacy_sorted, flat_top_k, semantic_merge, or recency_aware)"
    )


def load_and_aggregate_raw_experiences(
    experience_dir: Path,
    original_idx: int,
    n_completions_to_use: int,
    question_text: str = "",
    retrieval: str = "first",
    embed_model_path: str = "",
    rerank_model_name: Optional[str] = None,
    rerank_pool_mult: int = 4,
    *,
    cross_query_parsed: Optional[List[Tuple[Dict[str, Any], str, int]]] = None,
    cross_query_doc_emb: Optional[np.ndarray] = None,
    max_aggregated_propositions: int = 0,
    max_aggregated_pitfalls: int = 0,
    experience_aggregation: str = "legacy_sorted",
    aggregation_semantic_threshold: float = 0.82,
    aggregation_recency_head_fraction: float = 0.55,
) -> Optional[Dict[str, List[str]]]:
    """Load dedup experience and return critical_pitfalls and verified_propositions."""
    emb = embed_model_path or "/data/ppnm/models/all-MiniLM-L6-v2"

    if retrieval.startswith("cross_query"):
        if not cross_query_parsed:
            return None
        inner = (
            "cross_query_embedding_rerank"
            if retrieval == "cross_query_embedding_rerank"
            else "cross_query_embedding"
        )
        cross_query_pairs = [(r, d) for r, d, _ in cross_query_parsed]
        selected_data = _select_from_parsed_records(
            cross_query_pairs,
            question_text,
            n_completions_to_use,
            inner,
            emb,
            rerank_model_name,
            rerank_pool_mult,
            doc_embeddings=cross_query_doc_emb,
        )
    else:
        experience_file = experience_dir / f"{original_idx}.jsonl"
        if not experience_file.exists():
            return None
        raw_data = load_jsonl(str(experience_file))
        if not raw_data:
            return None

        selected_data = select_experience_records_for_question(
            raw_data,
            question_text,
            n_completions_to_use,
            retrieval,
            emb,
            rerank_model_name,
            rerank_pool_mult,
        )

    prop_sorted, pit_sorted = apply_experience_aggregation(
        selected_data,
        max_aggregated_propositions,
        max_aggregated_pitfalls,
        experience_aggregation,
        emb,
        aggregation_semantic_threshold,
        aggregation_recency_head_fraction,
    )
    return {
        "critical_pitfalls": pit_sorted,
        "verified_propositions": prop_sorted,
    }


def build_p_pos_content(experience_data: Dict[str, Any]) -> str:
    """Content for positive stream: propositions only."""
    if not experience_data:
        return ""
    props = experience_data.get("verified_propositions", [])
    if not props:
        return ""
    return "\n".join([f"- {p}" for p in props])


def build_p_neg_content(experience_data: Dict[str, Any]) -> str:
    """Content for negative stream: inducing prompt + pitfalls list."""
    if not experience_data:
        return ""
    pitfalls = experience_data.get("critical_pitfalls", [])
    if not pitfalls:
        return ""
    return "\n".join([f"- {e}" for e in pitfalls])


def extract_thinking(text: str) -> Tuple[str, str]:
    end_tag = "</think>"
    if end_tag in text:
        parts = text.split(end_tag, 1)
        reasoning = parts[0].strip().replace("<think>", "").strip()
        final_text = parts[1].strip() if len(parts) > 1 else ""
        return final_text, reasoning
    return text.strip(), ""


def load_existing_output(output_file: Path) -> List[Dict[str, Any]]:
    if not output_file.exists():
        return []
    try:
        with open(output_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        return [
            c
            for c in data.get("completions", [])
            if c.get("text") or c.get("reasoning_content")
        ]
    except Exception:
        return []


def save_result(
    output_dir: Path,
    original_idx: int,
    question_data: Dict[str, Any],
    new_completions: List[Dict[str, Any]],
    *,
    cto_branch_mode: str = "full",
):
    output_file = output_dir / f"{original_idx}.json"
    existing = load_existing_output(output_file)
    all_completions = existing + new_completions
    result = {
        "index": original_idx,
        "question_id": question_data.get("question_id", f"q_{original_idx}"),
        "question": question_data.get("question", ""),
        "answer": question_data.get("answer", ""),
        "completions": all_completions,
        "n_completions": len(all_completions),
        "cto_branch_mode": cto_branch_mode,
    }
    for k, v in question_data.items():
        if k not in result:
            result[k] = v
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)


# ---------------------------------------------------------------------------
# CTO decoding: dual forward + logit penalty + plausibility + sample
# ---------------------------------------------------------------------------

def apply_plausibility_constraint(
    logits_pos: torch.Tensor,
    logits_cto: torch.Tensor,
    top_k: int,
) -> torch.Tensor:
    """Only keep CTO adjustment for tokens in top-K of logits_pos; else restore logits_pos."""
    # logits_pos, logits_cto: (vocab_size,) on device
    if top_k <= 0:
        return logits_cto
    _, top_indices = torch.topk(logits_pos, min(top_k, logits_pos.size(-1)), dim=-1)
    mask = torch.zeros_like(logits_pos, dtype=torch.bool, device=logits_pos.device)
    mask.scatter_(-1, top_indices, True)
    out = torch.where(mask, logits_cto, logits_pos)
    return out


def sample_from_logits(
    logits: torch.Tensor,
    temperature: float,
    top_p: float,
    top_k: int,
    pad_token_id: Optional[int] = None,
) -> int:
    """Single-token sampling with temperature, top_p, top_k. Returns token id (int)."""
    if temperature <= 0:
        return logits.argmax(dim=-1).item()
    logits = logits.float() / temperature
    # Top-k
    if top_k > 0:
        k = min(top_k, logits.size(-1))
        v, _ = torch.topk(logits, k, dim=-1)
        logits[logits < v[..., -1, None]] = float("-inf")
    # Top-p (nucleus)
    if top_p < 1.0 and top_p > 0:
        sorted_logits, _ = torch.sort(logits, descending=True)
        cumsum = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
        cutoff = (cumsum >= top_p).int().argmax(dim=-1).item()
        if cutoff < logits.size(-1) - 1:
            cutoff_val = sorted_logits[..., cutoff]
            logits[logits < cutoff_val] = float("-inf")
    probs = torch.softmax(logits, dim=-1)
    if pad_token_id is not None:
        probs[..., pad_token_id] = 0
        if probs.sum() <= 0:
            probs[..., pad_token_id] = 1.0
        probs = probs / probs.sum()
    token_id = torch.multinomial(probs, 1).item()
    return token_id


def autoregressive_decode_single(
    model: torch.nn.Module,
    tokenizer: Any,
    input_ids_prompt: torch.LongTensor,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
    eos_token_id: Optional[int],
    pad_token_id: Optional[int],
    device: torch.device,
) -> List[int]:
    """Standard single-stream decode (no CTO). Used for prompt-only pos+neg ablation."""
    model.eval()
    generated: List[int] = []
    current = input_ids_prompt.to(device).unsqueeze(0)
    past_key_values = None

    for _ in range(max_new_tokens):
        if past_key_values is None:
            with torch.no_grad():
                out = model(
                    input_ids=current,
                    use_cache=True,
                    return_dict=True,
                )
            logits = out.logits[:, -1, :].squeeze(0)
            past_key_values = out.past_key_values
        else:
            with torch.no_grad():
                out = model(
                    input_ids=current,
                    past_key_values=past_key_values,
                    use_cache=True,
                    return_dict=True,
                )
            logits = out.logits[:, -1, :].squeeze(0)
            past_key_values = out.past_key_values

        next_token = sample_from_logits(
            logits.unsqueeze(0),
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            pad_token_id=pad_token_id,
        )
        generated.append(next_token)

        if eos_token_id is not None and next_token == eos_token_id:
            break

        current = torch.tensor([[next_token]], dtype=current.dtype, device=device)

    return generated


def cto_decode_single(
    model: torch.nn.Module,
    tokenizer: Any,
    input_ids_primary: torch.LongTensor,
    input_ids_contrast: torch.LongTensor,
    max_new_tokens: int,
    alpha: float,
    plausibility_top_k: int,
    temperature: float,
    top_p: float,
    top_k: int,
    eos_token_id: Optional[int],
    pad_token_id: Optional[int],
    device: torch.device,
    *,
    use_contrast_stream: bool,
) -> List[int]:
    """
    Run CTO decoding for one sequence. Returns generated token ids (new tokens only).
    Primary stream carries the rollout being sampled; contrast stream supplies penalty logits.
    If use_contrast_stream is False (pos-only CTO), decoding follows primary logits only.
    """
    model.eval()
    generated: List[int] = []
    current_primary = input_ids_primary.to(device).unsqueeze(0)  # (1, L_pri)
    current_contrast = input_ids_contrast.to(device).unsqueeze(0)  # (1, L_neg)
    past_key_values_pri = None
    past_key_values_con = None

    for _ in range(max_new_tokens):
        # Primary forward
        if past_key_values_pri is None:
            with torch.no_grad():
                out_pri = model(
                    input_ids=current_primary,
                    use_cache=True,
                    return_dict=True,
                )
            logits_primary = out_pri.logits[:, -1, :].squeeze(0)  # (V,)
            past_key_values_pri = out_pri.past_key_values
        else:
            with torch.no_grad():
                out_pri = model(
                    input_ids=current_primary,
                    past_key_values=past_key_values_pri,
                    use_cache=True,
                    return_dict=True,
                )
            logits_primary = out_pri.logits[:, -1, :].squeeze(0)
            past_key_values_pri = out_pri.past_key_values

        if use_contrast_stream:
            if past_key_values_con is None:
                with torch.no_grad():
                    out_con = model(
                        input_ids=current_contrast,
                        use_cache=True,
                        return_dict=True,
                    )
                logits_contrast = out_con.logits[:, -1, :].squeeze(0)
                past_key_values_con = out_con.past_key_values
            else:
                with torch.no_grad():
                    out_con = model(
                        input_ids=current_contrast,
                        past_key_values=past_key_values_con,
                        use_cache=True,
                        return_dict=True,
                    )
                logits_contrast = out_con.logits[:, -1, :].squeeze(0)
                past_key_values_con = out_con.past_key_values

            logits_cto = logits_primary - alpha * logits_contrast
            logits_use = apply_plausibility_constraint(
                logits_primary, logits_cto, plausibility_top_k
            )
        else:
            logits_use = logits_primary

        next_token = sample_from_logits(
            logits_use.unsqueeze(0),
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            pad_token_id=pad_token_id,
        )
        generated.append(next_token)

        if eos_token_id is not None and next_token == eos_token_id:
            break

        next_t = torch.tensor([[next_token]], dtype=current_primary.dtype, device=device)
        current_primary = next_t
        if use_contrast_stream:
            current_contrast = next_t

    return generated


def run_cto_for_question(
    model: Any,
    tokenizer: Any,
    question_text: str,
    experience_data: Dict[str, Any],
    n_completions: int,
    max_new_tokens: int,
    alpha: float,
    plausibility_top_k: int,
    temperature: float,
    top_p: float,
    top_k: int,
    device: torch.device,
    branch_mode: str = "full",
) -> List[Dict[str, Any]]:
    """Build prompts from branch_mode and run CTO (or plain) decodes."""
    props_str = build_p_pos_content(experience_data)
    pitfalls_str = build_p_neg_content(experience_data)

    system_neg = (
        (P_NEG_SYSTEM_PREFIX + pitfalls_str).strip() if pitfalls_str else "Do not repeat the following errors:\n"
    )
    eos_id = getattr(tokenizer, "eos_token_id", None)
    pad_id = getattr(tokenizer, "pad_token_id", None) or eos_id

    if branch_mode == "prompt_both_no_contrast":
        system_text = _build_prompt_both_system(props_str, pitfalls_str)
        if not props_str.strip() and not pitfalls_str.strip():
            system_text = P_BOTH_SYSTEM_FALLBACK
        messages_flat = [
            {"role": "system", "content": system_text},
            {"role": "user", "content": question_text},
        ]
        input_ids_flat = tokenizer.apply_chat_template(
            messages_flat,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        ).squeeze(0)

        completions: List[Dict[str, Any]] = []
        for _ in range(n_completions):
            new_ids = autoregressive_decode_single(
                model=model,
                tokenizer=tokenizer,
                input_ids_prompt=input_ids_flat,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                eos_token_id=eos_id,
                pad_token_id=pad_id,
                device=device,
            )
            if not new_ids:
                text_raw = ""
            else:
                text_raw = tokenizer.decode(new_ids, skip_special_tokens=False)
            final_text, reasoning = extract_thinking(text_raw)
            completions.append(
                {
                    "text": final_text,
                    "reasoning_content": reasoning,
                    "tokens": len(new_ids),
                    "finish_reason": "stop" if (eos_id and new_ids and new_ids[-1] == eos_id) else "length",
                }
            )
        return completions

    # Dual-stream CTO variants
    system_pos = (P_POS_SYSTEM_PREFIX + props_str).strip() if props_str else ""

    messages_pos = [
        {
            "role": "system",
            "content": system_pos or "Please reason step by step, and put your final answer within \\boxed{}.",
        },
        {"role": "user", "content": question_text},
    ]
    messages_neg = [
        {"role": "system", "content": system_neg or "Please reason step by step."},
        {"role": "user", "content": question_text},
    ]
    messages_base = [
        {"role": "system", "content": P_BOTH_SYSTEM_FALLBACK},
        {"role": "user", "content": question_text},
    ]

    input_ids_pos = tokenizer.apply_chat_template(
        messages_pos,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    ).squeeze(0)
    input_ids_neg = tokenizer.apply_chat_template(
        messages_neg,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    ).squeeze(0)
    input_ids_base = tokenizer.apply_chat_template(
        messages_base,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    ).squeeze(0)

    use_contrast_stream = branch_mode != "pos_only"

    if branch_mode == "neg_only":
        input_ids_primary_eff = input_ids_base
        if not pitfalls_str.strip():
            logger.warning(
                "neg_only CTO requested but pitfalls are empty — decoding collapses to the baseline prompt (no penalty stream)."
            )
    elif branch_mode in ("full", "global_neg", "cfg_plain"):
        input_ids_primary_eff = input_ids_pos
    elif branch_mode == "pos_only":
        input_ids_primary_eff = input_ids_pos
    else:
        raise ValueError(f"Unknown CTO branch_mode for HF: {branch_mode}")

    if branch_mode == "cfg_plain":
        input_ids_contrast_eff = input_ids_base
        use_contrast_eff = True
    elif branch_mode == "neg_only":
        input_ids_contrast_eff = input_ids_neg
        use_contrast_eff = bool(pitfalls_str.strip())
    else:
        input_ids_contrast_eff = input_ids_neg
        use_contrast_eff = use_contrast_stream if branch_mode != "neg_only" else bool(pitfalls_str.strip())

    completions_hf: List[Dict[str, Any]] = []
    for _ in range(n_completions):
        new_ids = cto_decode_single(
            model=model,
            tokenizer=tokenizer,
            input_ids_primary=input_ids_primary_eff,
            input_ids_contrast=input_ids_contrast_eff,
            max_new_tokens=max_new_tokens,
            alpha=alpha,
            plausibility_top_k=plausibility_top_k,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            eos_token_id=eos_id,
            pad_token_id=pad_id,
            device=device,
            use_contrast_stream=use_contrast_eff,
        )
        if not new_ids:
            text_raw = ""
        else:
            text_raw = tokenizer.decode(new_ids, skip_special_tokens=False)
        final_text, reasoning = extract_thinking(text_raw)
        completions_hf.append(
            {
                "text": final_text,
                "reasoning_content": reasoning,
                "tokens": len(new_ids),
                "finish_reason": "stop" if (eos_id and new_ids and new_ids[-1] == eos_id) else "length",
            }
        )
    return completions_hf


def _token_count(tokenizer: Any, text: str) -> int:
    try:
        return len(tokenizer.encode(text, add_special_tokens=False))
    except Exception:
        return len(tokenizer(text, add_special_tokens=False).input_ids)


def _parse_pos_neg_weights(s: str) -> Tuple[int, int]:
    """'pos:neg' weight pair for splitting experience token budget, e.g. 1:1, 1:2, 2:1."""
    parts = s.strip().split(":", 1)
    if len(parts) != 2:
        raise ValueError("expected 'pos:neg' like 1:1 or 1:2")
    return int(parts[0].strip()), int(parts[1].strip())


def _prefix_strings_fitting_bullet_token_budget(
    tokenizer: Any,
    items: List[str],
    max_tokens: int,
) -> List[str]:
    """Keep a prefix of `items` such that - bullet formatting stays within max_tokens (tokenizer)."""
    if max_tokens <= 0:
        return []
    out: List[str] = []
    for p in items:
        if not isinstance(p, str):
            continue
        trial = out + [p]
        text = "\n".join(f"- {x}" for x in trial)
        if _token_count(tokenizer, text) <= max_tokens:
            out.append(p)
        else:
            break
    return out


def apply_experience_token_budget(
    experience_data: Dict[str, Any],
    tokenizer: Any,
    total_budget: int,
    pos_w: int,
    neg_w: int,
) -> Dict[str, Any]:
    """
    After aggregation: cap combined token count of experience bullet text only (lines '- ...' for
    propositions and pitfalls), then split the budget between P_pos and P_neg by pos_w:neg_w.
    total_budget=0 in CLI means this function is not called.
    """
    wsum = max(1, int(pos_w) + int(neg_w))
    tb = int(total_budget)
    t_pos = max(0, (tb * int(pos_w)) // wsum)
    t_neg = max(0, tb - t_pos)
    props = [p for p in (experience_data.get("verified_propositions") or []) if isinstance(p, str)]
    pits = [p for p in (experience_data.get("critical_pitfalls") or []) if isinstance(p, str)]
    return {
        "verified_propositions": _prefix_strings_fitting_bullet_token_budget(
            tokenizer, props, t_pos
        ),
        "critical_pitfalls": _prefix_strings_fitting_bullet_token_budget(
            tokenizer, pits, t_neg
        ),
    }


def _get_vllm_max_model_len(llm: Any) -> Optional[int]:
    """Effective max sequence length enforced by vLLM (may be below CLI --max-model-len)."""
    try:
        eng = getattr(llm, "llm_engine", None)
        if eng is None:
            return None
        cfg = getattr(eng, "model_config", None)
        if cfg is None:
            return None
        m = getattr(cfg, "max_model_len", None)
        return int(m) if m is not None else None
    except Exception:
        return None


def _truncate_concat_for_vllm(
    tokenizer: Any,
    prefix: str,
    suffix: str,
    max_tokens: int,
) -> Tuple[str, int, int]:
    """
    Ensure token count of (prefix + suffix_trunc) <= max_tokens by truncating suffix from the end.
    If prefix alone exceeds max_tokens, truncate prefix on the token axis (drops candidate).
    Returns (full_text, prefix_token_len, suffix_token_len) using the same counting as _token_count.
    """
    max_tokens = max(int(max_tokens), 1)
    try:
        ids_p = tokenizer.encode(prefix, add_special_tokens=False)
    except Exception:
        ids_p = tokenizer(prefix, add_special_tokens=False).input_ids

    if len(ids_p) > max_tokens:
        try:
            prefix_cut = tokenizer.decode(ids_p[:max_tokens], skip_special_tokens=False)
        except Exception:
            prefix_cut = prefix
        pt = _token_count(tokenizer, prefix_cut)
        ft = _token_count(tokenizer, prefix_cut)
        return prefix_cut, pt, ft - pt

    lo, hi = 0, len(suffix)
    best = ""
    while lo <= hi:
        mid = (lo + hi) // 2
        cand = suffix[:mid]
        full = prefix + cand
        n = _token_count(tokenizer, full)
        if n <= max_tokens:
            best = cand
            lo = mid + 1
        else:
            hi = mid - 1

    full_text = prefix + best
    pt = _token_count(tokenizer, prefix)
    ft = _token_count(tokenizer, full_text)
    return full_text, pt, ft - pt


def _sum_prompt_logprob_for_suffix(
    prompt_logprobs: Any,
    suffix_token_count: int,
) -> float:
    """
    vLLM returns prompt_logprobs as a list aligned to prompt tokens.
    Each element is usually a dict {token_id: Logprob} for the observed token.
    We sum the last `suffix_token_count` observed-token logprobs.
    """
    if not prompt_logprobs or suffix_token_count <= 0:
        return 0.0
    tail = prompt_logprobs[-suffix_token_count:]
    s = 0.0
    for x in tail:
        if not x:
            continue
        # x can be {token_id: Logprob} or list; handle dict first
        if isinstance(x, dict):
            lp_obj = next(iter(x.values()))
            lp = getattr(lp_obj, "logprob", None)
            if lp is None:
                try:
                    lp = float(lp_obj)
                except Exception:
                    lp = None
            if lp is not None:
                s += float(lp)
        elif isinstance(x, list) and len(x) > 0:
            lp_obj = x[0]
            lp = getattr(lp_obj, "logprob", None)
            if lp is None:
                try:
                    lp = float(lp_obj)
                except Exception:
                    lp = None
            if lp is not None:
                s += float(lp)
    return float(s)


def run_cto_for_question_vllm(
    llm: Any,
    tokenizer: Any,
    question_text: str,
    experience_data: Dict[str, Any],
    n_completions: int,
    max_new_tokens: int,
    alpha: float,
    plausibility_top_k: int,
    temperature: float,
    top_p: float,
    top_k: int,
    candidate_k: int,
    max_model_len: Optional[int] = None,
    vllm_score_batch_size: int = 1,
    branch_mode: str = "full",
) -> List[Dict[str, Any]]:
    """
    vLLM-backed CTO (sequence-level contrastive rerank), with branch ablations:
    - full: candidates from P_pos, score logp_pos - alpha * logp_neg
    - pos_only: same candidates, rank by logp_pos only (no neg penalty term)
    - neg_only: candidates from baseline prompt (no propositions), score logp_base - alpha * logp_neg
    - cfg_plain: candidates from P_pos, score logp_pos - alpha * logp_plain (neutral no-experience CFG control)
    - prompt_both_no_contrast: single merged prompt — plain sampling, no logits contrast
    """
    props_str = build_p_pos_content(experience_data)
    pitfalls_str = build_p_neg_content(experience_data)

    system_pos = (P_POS_SYSTEM_PREFIX + props_str).strip() if props_str else ""
    system_neg = (
        (P_NEG_SYSTEM_PREFIX + pitfalls_str).strip()
        if pitfalls_str
        else "Do not repeat the following errors:\n"
    )

    messages_pos = [
        {
            "role": "system",
            "content": system_pos or "Please reason step by step, and put your final answer within \\boxed{}.",
        },
        {"role": "user", "content": question_text},
    ]
    messages_neg = [
        {"role": "system", "content": system_neg or "Please reason step by step."},
        {"role": "user", "content": question_text},
    ]
    messages_base = [
        {"role": "system", "content": P_BOTH_SYSTEM_FALLBACK},
        {"role": "user", "content": question_text},
    ]

    prompt_pos = tokenizer.apply_chat_template(
        messages_pos,
        tokenize=False,
        add_generation_prompt=True,
    )
    prompt_neg = tokenizer.apply_chat_template(
        messages_neg,
        tokenize=False,
        add_generation_prompt=True,
    )
    prompt_base = tokenizer.apply_chat_template(
        messages_base,
        tokenize=False,
        add_generation_prompt=True,
    )

    if branch_mode == "prompt_both_no_contrast":
        sys_both = _build_prompt_both_system(props_str, pitfalls_str)
        if not props_str.strip() and not pitfalls_str.strip():
            sys_both = P_BOTH_SYSTEM_FALLBACK
        msgs_both = [
            {"role": "system", "content": sys_both},
            {"role": "user", "content": question_text},
        ]
        prompt_merged = tokenizer.apply_chat_template(
            msgs_both,
            tokenize=False,
            add_generation_prompt=True,
        )
        nc = max(int(n_completions), 1)
        gen_params = SamplingParams(
            n=nc,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            max_tokens=max_new_tokens,
        )
        gen_outputs = llm.generate([prompt_merged], gen_params)[0].outputs
        completions_merged: List[Dict[str, Any]] = []
        for o in gen_outputs[:nc]:
            text_raw = o.text or ""
            final_text, reasoning = extract_thinking(text_raw)
            completions_merged.append(
                {
                    "text": final_text,
                    "reasoning_content": reasoning,
                    "tokens": _token_count(tokenizer, text_raw),
                    "finish_reason": getattr(o, "finish_reason", None) or "stop",
                }
            )
        return completions_merged

    if branch_mode == "neg_only":
        prompt_gen = prompt_base
        if not pitfalls_str.strip():
            logger.warning(
                "neg_only vLLM CTO: empty pitfalls → generating from baseline and ranking by sequence logprob only."
            )
    else:
        prompt_gen = prompt_pos

    if branch_mode == "cfg_plain":
        prompt_contrast = prompt_base
        use_neg_scores = True
    else:
        prompt_contrast = prompt_neg
        use_neg_scores = branch_mode in ("full", "neg_only", "global_neg") and bool(pitfalls_str.strip())
    prompt_score_primary = prompt_base if branch_mode == "neg_only" else prompt_pos

    cand_k = max(int(candidate_k), int(n_completions), 1)
    gen_params = SamplingParams(
        n=cand_k,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        max_tokens=max_new_tokens,
    )
    gen_out = llm.generate([prompt_gen], gen_params)[0].outputs
    candidates = [o.text or "" for o in gen_out]

    prompt_primary_len = _token_count(tokenizer, prompt_score_primary)
    prompt_contrast_len = _token_count(tokenizer, prompt_contrast)

    vllm_cap = max_model_len if max_model_len is not None else _get_vllm_max_model_len(llm)
    if vllm_cap is None:
        try:
            vllm_cap = int(getattr(tokenizer, "model_max_length", 131072) or 131072)
        except Exception:
            vllm_cap = 131072

    score_prompt_budget = max(1, int(vllm_cap) - 1)

    full_prompts_primary: List[str] = []
    full_prompts_contrast: List[str] = []
    suffix_primary: List[int] = []
    suffix_contrast: List[int] = []
    for c in candidates:
        raw_sp = max(_token_count(tokenizer, prompt_score_primary + c) - prompt_primary_len, 0)
        raw_sn = max(_token_count(tokenizer, prompt_contrast + c) - prompt_contrast_len, 0)
        fp, _ppt, sp = _truncate_concat_for_vllm(
            tokenizer, prompt_score_primary, c, score_prompt_budget
        )
        fn, _pnt, sn = _truncate_concat_for_vllm(tokenizer, prompt_contrast, c, score_prompt_budget)
        full_prompts_primary.append(fp)
        full_prompts_contrast.append(fn)
        suffix_primary.append(sp)
        suffix_contrast.append(sn)
        if sp < raw_sp or sn < raw_sn:
            logger.warning(
                "Truncated CTO scoring prompt to fit vLLM (max_model_len=%d, score_prompt_budget=%d; pri_suffix=%d neg_suffix=%d).",
                vllm_cap,
                score_prompt_budget,
                sp,
                sn,
            )

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
        pri_batch = full_prompts_primary[start:end]

        outs_pri = llm.generate(pri_batch, score_params)
        outs_neg_gen: Optional[Any] = None
        if use_neg_scores:
            contrast_batch_s = full_prompts_contrast[start:end]
            outs_neg_gen = llm.generate(contrast_batch_s, score_params)

        def _scores_one_shard(
            shard_start: int,
            shard_end: int,
            shard_outs_pri: Any,
            shard_outs_neg: Optional[Any],
        ) -> None:
            for bi, i in enumerate(range(shard_start, shard_end)):
                out_pri_obj = shard_outs_pri[bi]
                plp_pri = getattr(out_pri_obj, "prompt_logprobs", None)
                logp_pri = _sum_prompt_logprob_for_suffix(plp_pri, suffix_primary[i])

                if not use_neg_scores or shard_outs_neg is None:
                    scored.append((float(logp_pri), i))
                    continue

                out_neg_obj = shard_outs_neg[bi]
                plp_neg = getattr(out_neg_obj, "prompt_logprobs", None)
                logp_neg = _sum_prompt_logprob_for_suffix(plp_neg, suffix_contrast[i])
                scored.append((float(logp_pri - alpha * logp_neg), i))

        if len(outs_pri) != end - start:
            logger.warning(
                "CTO vLLM pri batch mismatch (got %d vs %d); falling back to one-by-one for this chunk.",
                len(outs_pri),
                end - start,
            )
            for i in range(start, end):
                out_pri = llm.generate([full_prompts_primary[i]], score_params)[0]
                outs_neg_i = (
                    llm.generate([full_prompts_contrast[i]], score_params)[0]
                    if use_neg_scores
                    else None
                )
                lp_p = getattr(out_pri, "prompt_logprobs", None)
                logp_pri = _sum_prompt_logprob_for_suffix(lp_p, suffix_primary[i])
                if not use_neg_scores or outs_neg_i is None:
                    scored.append((float(logp_pri), i))
                else:
                    lp_n = getattr(outs_neg_i, "prompt_logprobs", None)
                    logp_neg = _sum_prompt_logprob_for_suffix(lp_n, suffix_contrast[i])
                    scored.append((float(logp_pri - alpha * logp_neg), i))
            continue

        if use_neg_scores:
            assert outs_neg_gen is not None
            if len(outs_neg_gen) != end - start:
                logger.warning(
                    "CTO vLLM neg batch mismatch (got %d vs %d); falling back per-candidate scoring.",
                    len(outs_neg_gen),
                    end - start,
                )
                for i in range(start, end):
                    out_pri = llm.generate([full_prompts_primary[i]], score_params)[0]
                    out_neg = llm.generate([full_prompts_contrast[i]], score_params)[0]
                    lp_p = getattr(out_pri, "prompt_logprobs", None)
                    lp_n = getattr(out_neg, "prompt_logprobs", None)
                    logp_pri = _sum_prompt_logprob_for_suffix(lp_p, suffix_primary[i])
                    logp_neg = _sum_prompt_logprob_for_suffix(lp_n, suffix_contrast[i])
                    scored.append((float(logp_pri - alpha * logp_neg), i))
            else:
                _scores_one_shard(start, end, outs_pri, outs_neg_gen)
        else:
            _scores_one_shard(start, end, outs_pri, None)

    scored.sort(key=lambda x: x[0], reverse=True)
    picked = scored[: int(n_completions)]

    completions: List[Dict[str, Any]] = []
    for score, idx in picked:
        text_raw = candidates[idx]
        final_text, reasoning = extract_thinking(text_raw)
        completions.append(
            {
                "text": final_text,
                "reasoning_content": reasoning,
                "tokens": _token_count(tokenizer, text_raw),
                "finish_reason": "score_rerank",
                "cto_score": score,
            }
        )
    return completions


def main():
    parser = argparse.ArgumentParser(
        description="CTO: Experience-Guided Search with Contrastive (Dual-Stream) Decoding"
    )
    parser.add_argument("--model", type=str, required=True, help="Path to HuggingFace model")
    parser.add_argument("--input", type=str, required=True, help="Input JSONL (questions)")
    parser.add_argument("--experience-dir", type=str, required=True, help="Dedup experience JSONL dir")
    parser.add_argument("--output", type=str, required=True, help="Output directory")
    parser.add_argument("--n-experience-completions", type=int, default=5)
    parser.add_argument("--n-completions", type=int, default=1)
    parser.add_argument("--alpha", "--contrastive-alpha", type=float, default=0.7, dest="alpha",
                        help="CTO penalty: logits_CTO = logits_pos - alpha * logits_neg")
    parser.add_argument(
        "--plausibility-top-k", type=int, default=20,
        help="Only apply CTO adjustment on top-K logits of the primary stream (HF path; vLLM uses this as default candidate-K budget).")
    parser.add_argument(
        "--cto-branch-mode",
        type=str,
        default="full",
        choices=["full", "pos_only", "neg_only", "global_neg", "cfg_plain", "prompt_both_no_contrast"],
        help=(
            "Branch ablation / prompt-only baseline: "
            "full=dual CTO (local pos + local neg); "
            "global_neg=local pos + cross-problem neg (excludes current problem_id); "
            "pos_only=P_pos generation without subtracting logits_neg "
            "(vLLM: rerank by log p under P_pos only); "
            "neg_only=no verified propositions in the primary prompt, contrast against P_neg; "
            "cfg_plain=P_pos generation with CFG-style neutral contrast (logp_pos - alpha * logp_plain); "
            "prompt_both_no_contrast=both bullets in one system prompt + plain decoding (no logit subtraction)."
        ),
    )
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--start-idx", type=int, default=0)
    parser.add_argument("--end-idx", type=int, default=None)
    parser.add_argument(
        "--force-rerun",
        action="store_true",
        help=(
            "Ignore existing per-index JSON outputs and re-decode all questions in the start..end range. "
            "Otherwise, questions with >= n_completions entries in the output file are skipped (resume)."
        ),
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="vllm",
        choices=["vllm", "hf"],
        help="Inference backend. vllm = fast sequence-level rerank CTO; hf = exact token-level dual-logit CTO.",
    )
    parser.add_argument("--tensor-parallel-size", type=int, default=1, help="vLLM tensor parallel size")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85, help="vLLM: fraction of GPU memory for KV cache")
    parser.add_argument("--max-model-len", type=int, default=100000, help="vLLM: cap model max sequence length to limit KV cache")
    parser.add_argument("--candidate-k", type=int, default=None, help="vLLM: number of P_pos candidates to generate before rerank (default: plausibility-top-k)")
    parser.add_argument(
        "--vllm-score-batch-size",
        type=int,
        default=1,
        help=(
            "vLLM CTO: how many candidates to score per generate() call on each stream (P_pos / P_neg). "
            "1 = legacy (slowest, lowest peak VRAM). 4–16 typically much faster on multi-GPU; OOM时调回 1 或 2。"
        ),
    )
    parser.add_argument("--device", type=str, default="cuda", help="HF backend device (e.g. cuda:0)")
    parser.add_argument("--device-map", type=str, default=None, help="HF backend device_map (e.g. auto)")
    parser.add_argument("--dtype", type=str, default="auto", choices=["auto", "float16", "bfloat16"], help="Model dtype")
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
        help=(
            "Which experience JSONL records to merge for this question: "
            "first=first n (legacy); embedding=cosine(query, record doc) with bi-encoder; "
            "embedding_rerank=bi-encoder pool then cross-encoder rerank; "
            "cross_query_*=pool ALL records from ALL *.jsonl in experience-dir, retrieve by similarity "
            "of current question to (source question + propositions/pitfalls) — no oracle, true TTS."
        ),
    )
    parser.add_argument(
        "--retrieval-embedding-model",
        type=str,
        default="/data/ppnm/models/all-MiniLM-L6-v2",
        help="sentence-transformers model path for embedding / first stage of rerank",
    )
    parser.add_argument(
        "--retrieval-rerank-model",
        type=str,
        default=_DEFAULT_RETRIEVAL_RERANK_MODEL,
        help="sentence-transformers CrossEncoder: local path or Hugging Face model id",
    )
    parser.add_argument(
        "--retrieval-rerank-pool-mult",
        type=int,
        default=4,
        help="For embedding_rerank: pool size = n_experience_completions * this (capped by #records)",
    )
    parser.add_argument(
        "--max-aggregated-propositions",
        type=int,
        default=0,
        help="After merging selected experience records, keep at most this many distinct propositions (0=unlimited).",
    )
    parser.add_argument(
        "--max-aggregated-pitfalls",
        type=int,
        default=0,
        help="After merging selected experience records, keep at most this many distinct pitfalls (0=unlimited).",
    )
    parser.add_argument(
        "--experience-aggregation",
        type=str,
        default="legacy_sorted",
        choices=["legacy_sorted", "flat_top_k", "semantic_merge", "recency_aware"],
        help=(
            "After retrieval: how to merge strings across selected records. "
            "legacy_sorted=original behavior: set union, lexicographic sort, truncate (default); "
            "flat_top_k=deduplicate keeping retrieval order then truncate; "
            "semantic_merge=embedding-similarity clusters then one line per cluster; "
            "recency_aware=split records at midpoint, allocate slots between head/tail."
        ),
    )
    parser.add_argument(
        "--aggregation-semantic-threshold",
        type=float,
        default=0.82,
        help="Cosine similarity threshold for semantic_merge (normalized embeddings).",
    )
    parser.add_argument(
        "--aggregation-recency-head-fraction",
        type=float,
        default=0.55,
        help="For recency_aware: target fraction of max_* bullets taken from retrieval-head records before tail.",
    )
    parser.add_argument(
        "--experience-token-budget",
        type=int,
        default=0,
        help=(
            "0=off. Otherwise max total tokenizer tokens for experience bullets only "
            "('- item' for propositions + pitfalls) split by --experience-pos-neg-weights. "
            "After aggregation, lists are truncated to fit (prefix of bullets per branch)."
        ),
    )
    parser.add_argument(
        "--experience-pos-neg-weights",
        type=str,
        default="1:1",
        help="Split experience-token-budget between pos/neg bullet blocks, e.g. 1:1, 1:2, 2:1 (pos:neg).",
    )
    args = parser.parse_args()
    if int(getattr(args, "experience_token_budget", 0) or 0) > 0:
        try:
            _pw, _nw = _parse_pos_neg_weights(args.experience_pos_neg_weights)
        except Exception as e:
            raise SystemExit(f"Invalid --experience-pos-neg-weights: {e}")
        logger.info(
            "Experience token budget: total=%d pos:neg weights=%d:%d (bullet text only, after aggregation)",
            args.experience_token_budget,
            _pw,
            _nw,
        )

    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)
    experience_path = Path(args.experience_dir)

    questions = load_jsonl(args.input)
    end_idx = args.end_idx if args.end_idx is not None else len(questions)
    questions = questions[args.start_idx:end_idx]
    logger.info(f"Processing {len(questions)} questions (idx {args.start_idx}..{end_idx}); cto_branch_mode={args.cto_branch_mode}")
    if args.experience_retrieval != "first":
        logger.info(
            "Experience retrieval: mode=%s embed_model=%s rerank_model=%s pool_mult=%d",
            args.experience_retrieval,
            args.retrieval_embedding_model,
            args.retrieval_rerank_model,
            args.retrieval_rerank_pool_mult,
        )
    if args.experience_aggregation != "legacy_sorted":
        logger.info(
            "Experience aggregation: mode=%s semantic_thr=%.3f recency_head_frac=%.3f",
            args.experience_aggregation,
            args.aggregation_semantic_threshold,
            args.aggregation_recency_head_fraction,
        )
    else:
        logger.info("Experience aggregation: mode=legacy_sorted (set union + lexicographic sort + truncate)")

    cross_query_parsed: Optional[List[Tuple[Dict[str, Any], str, int]]] = None
    cross_query_doc_emb: Optional[np.ndarray] = None
    if args.experience_retrieval.startswith("cross_query") or args.cto_branch_mode == "global_neg":
        idx2q = load_index_to_question_map(args.input)
        gp = build_global_experience_pool(experience_path, idx2q)
        cross_query_parsed = build_cross_query_parsed(gp)
        if not cross_query_parsed:
            logger.error("Cross-query experience pool is empty (no parseable records under %s).", experience_path)
            return
        logger.info(
            "Cross-query pool: %d records from all jsonl files; precomputing doc embeddings.",
            len(cross_query_parsed),
        )
        cross_query_doc_emb = precompute_doc_embeddings_for_parsed(
            cross_query_parsed,
            args.retrieval_embedding_model,
        )
        if args.cto_branch_mode == "global_neg":
            logger.info(
                "global_neg CTO: positive=local embedding_rerank; negative=cross-query pool excluding self problem_id."
            )

    pending = []
    for i, item in enumerate(questions):
        orig_idx = args.start_idx + i
        item["index"] = orig_idx
        out_file = output_path / f"{orig_idx}.json"
        if args.force_rerun:
            pending.append(item)
            continue
        existing = load_existing_output(out_file)
        if len(existing) >= args.n_completions:
            continue
        pending.append(item)

    if not pending:
        logger.info("All questions already completed.")
        return

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    if args.backend == "vllm":
        if LLM is None or SamplingParams is None:
            raise ImportError("vLLM backend selected but vllm is not installed: pip install vllm")
        logger.info(
            f"Initializing vLLM for CTO (tensor_parallel_size={args.tensor_parallel_size}). "
            "This uses sequence-level contrastive reranking (fast)."
        )
        llm = LLM(
            model=args.model,
            tensor_parallel_size=args.tensor_parallel_size,
            trust_remote_code=True,
            gpu_memory_utilization=args.gpu_memory_utilization,
            max_model_len=args.max_model_len,
        )
        vllm_max_model_len = _get_vllm_max_model_len(llm)
        if vllm_max_model_len is None:
            vllm_max_model_len = args.max_model_len
        model = llm
        device = torch.device("cuda")
    else:
        from transformers import AutoModelForCausalLM  # local import to keep vllm-only envs lighter

        dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16, "auto": "auto"}
        dtype = dtype_map.get(args.dtype, "auto")
        device_map = getattr(args, "device_map", None)
        load_kw = dict(
            torch_dtype=dtype if isinstance(dtype, str) else dtype,
            trust_remote_code=True,
        )
        if device_map:
            load_kw["device_map"] = device_map
        else:
            load_kw["device_map"] = args.device
        logger.info(f"Loading HF model: {args.model} (dtype={dtype}, device_map={load_kw['device_map']})")
        model = AutoModelForCausalLM.from_pretrained(args.model, **load_kw)
        device = next(model.parameters()).device

    iter_items = pending
    if args.backend == "hf":
        # hf backend is much slower: show progress bar + ETA.
        iter_items = tqdm(pending, total=len(pending), desc="CTO(hf) decoding", unit="q")

    for item in iter_items:
        orig_idx = item["index"]
        question_text = item.get("question", "")
        if args.cto_branch_mode == "global_neg":
            experience_data = load_and_aggregate_global_neg_experiences(
                experience_path,
                orig_idx,
                args.n_experience_completions,
                question_text=question_text,
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
        else:
            experience_data = load_and_aggregate_raw_experiences(
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
        if not experience_data:
            logger.warning(f"No experience for question {orig_idx}, skipping.")
            continue
        if int(getattr(args, "experience_token_budget", 0) or 0) > 0:
            pw, nw = _parse_pos_neg_weights(args.experience_pos_neg_weights)
            experience_data = apply_experience_token_budget(
                experience_data,
                tokenizer,
                int(args.experience_token_budget),
                pw,
                nw,
            )
        if args.backend == "vllm":
            cand_k = args.candidate_k if args.candidate_k is not None else args.plausibility_top_k
            completions = run_cto_for_question_vllm(
                llm=model,
                tokenizer=tokenizer,
                question_text=question_text,
                experience_data=experience_data,
                n_completions=args.n_completions,
                max_new_tokens=args.max_tokens,
                alpha=args.alpha,
                plausibility_top_k=args.plausibility_top_k,
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.top_k,
                candidate_k=cand_k,
                max_model_len=vllm_max_model_len,
                vllm_score_batch_size=args.vllm_score_batch_size,
                branch_mode=args.cto_branch_mode,
            )
        else:
            completions = run_cto_for_question(
                model=model,
                tokenizer=tokenizer,
                question_text=question_text,
                experience_data=experience_data,
                n_completions=args.n_completions,
                max_new_tokens=args.max_tokens,
                alpha=args.alpha,
                plausibility_top_k=args.plausibility_top_k,
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.top_k,
                device=device,
                branch_mode=args.cto_branch_mode,
            )
        save_result(output_path, orig_idx, item, completions, cto_branch_mode=args.cto_branch_mode)
        logger.info(f"Saved question {orig_idx}: {len(completions)} completions")

    logger.info(f"Done. Results in {args.output}")


if __name__ == "__main__":
    main()
