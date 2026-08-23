#!/usr/bin/env python3
"""Multi-stage experience dropping utilities (extraction / retrieval / render)."""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

import cto_guided_search as cto

_EMB_ST: Dict[str, Any] = {}

_RISK_KEYWORDS = (
    "invalid",
    "avoid",
    "wrong",
    "assume",
    "assuming",
    "do not",
    "never",
    "incorrect",
    "mistake",
    "error",
)


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def _shorten(text: str, max_words: int = 18) -> str:
    words = text.strip().split()
    if len(words) <= max_words:
        return text.strip()
    return " ".join(words[:max_words]) + "..."


def _get_embedder(embed_model_path: str):
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        return None
    st = _EMB_ST.get(embed_model_path)
    if st is None:
        st = SentenceTransformer(embed_model_path)
        _EMB_ST[embed_model_path] = st
    return st


def _encode(st: Any, texts: List[str]) -> np.ndarray:
    if not texts:
        return np.zeros((0, 1), dtype=np.float64)
    return st.encode(texts, normalize_embeddings=True)


def _cos(a: np.ndarray, b: np.ndarray) -> float:
    if a.size == 0 or b.size == 0:
        return 0.0
    return float(np.dot(a, b))


def extract_strings_from_records(
    records: List[Dict[str, Any]],
) -> Tuple[List[str], List[str]]:
    pits: List[str] = []
    props: List[str] = []
    for record in records:
        cd = cto._parse_record_content_dict(record)
        if not cd:
            continue
        for p in cd.get("critical_pitfalls", []) or []:
            if isinstance(p, str) and p.strip():
                pits.append(p.strip())
            elif isinstance(p, dict):
                if "error_location" in p and "reasoning_flaw" in p:
                    pits.append(f"{p['error_location']}: {p['reasoning_flaw']}")
                elif "content" in p:
                    pits.append(str(p["content"]))
        for p in cd.get("verified_propositions", []) or []:
            if isinstance(p, str) and p.strip():
                props.append(p.strip())
            elif isinstance(p, dict) and "content" in p:
                props.append(str(p["content"]))
    return pits, props


def tally_support(items: List[str]) -> Dict[str, Dict[str, Any]]:
    """Map normalized text -> {text, support_count}."""
    out: Dict[str, Dict[str, Any]] = {}
    for item in items:
        key = _normalize_text(item)
        if key not in out:
            out[key] = {"text": item.strip(), "support_count": 0}
        out[key]["support_count"] += 1
    return out


def score_pitfall_metrics(
    pitfall: str,
    *,
    support_count: int,
    propositions: List[str],
    embed_model_path: str,
) -> Dict[str, float]:
    st = _get_embedder(embed_model_path)
    conflict = 0.0
    if st and propositions:
        pv = _encode(st, propositions)
        pit_v = _encode(st, [pitfall])[0]
        sims = np.dot(pv, pit_v)
        conflict = float(sims.max()) if sims.size else 0.0

    false_suppression = 0.0
    if support_count <= 1:
        false_suppression += 0.45
    if len(pitfall.split()) > 40:
        false_suppression += 0.15
    if conflict > 0.72:
        false_suppression += 0.25

    return {
        "support_count": float(support_count),
        "conflict_score": float(conflict),
        "false_suppression_risk": float(min(false_suppression, 1.0)),
    }


def filter_pitfalls_extraction(
    pitfall_stats: Dict[str, Dict[str, Any]],
    *,
    min_support: int,
    max_conflict: float,
    max_false_suppression: float,
) -> Tuple[List[str], List[Dict[str, Any]]]:
    kept: List[str] = []
    dropped: List[Dict[str, Any]] = []
    for _key, entry in pitfall_stats.items():
        text = entry["text"]
        metrics = entry.get("metrics", {})
        support = int(metrics.get("support_count", entry.get("support_count", 0)))
        conflict = float(metrics.get("conflict_score", 0.0))
        fsr = float(metrics.get("false_suppression_risk", 0.0))
        reason = None
        if support < min_support:
            reason = "low_support"
        elif conflict > max_conflict:
            reason = "high_conflict"
        elif fsr > max_false_suppression:
            reason = "high_false_suppression"
        if reason:
            dropped.append({"text": text, "reason": reason, **metrics})
        else:
            kept.append(text)
    return kept, dropped


def filter_propositions_extraction(
    prop_stats: Dict[str, Dict[str, Any]],
    *,
    min_support: int,
) -> List[str]:
    kept: List[str] = []
    for _key, entry in prop_stats.items():
        support = int(entry.get("support_count", 0))
        if support >= min_support:
            kept.append(entry["text"])
    return kept


def apply_extraction_dropping(
    records: List[Dict[str, Any]],
    *,
    source_round: int,
    embed_model_path: str,
    min_pitfall_support: int = 2,
    min_prop_support: int = 1,
    max_conflict: float = 0.75,
    max_false_suppression: float = 0.60,
) -> Tuple[List[str], List[str], Dict[str, Any]]:
    raw_pits, raw_props = extract_strings_from_records(records)
    pit_tally = tally_support(raw_pits)
    prop_tally = tally_support(raw_props)

    for entry in pit_tally.values():
        entry["metrics"] = score_pitfall_metrics(
            entry["text"],
            support_count=int(entry["support_count"]),
            propositions=[v["text"] for v in prop_tally.values()],
            embed_model_path=embed_model_path,
        )
        entry["metrics"]["source_round"] = float(source_round)

    kept_pits, dropped_pits = filter_pitfalls_extraction(
        pit_tally,
        min_support=min_pitfall_support,
        max_conflict=max_conflict,
        max_false_suppression=max_false_suppression,
    )
    kept_props = filter_propositions_extraction(
        prop_tally, min_support=min_prop_support
    )

    meta = {
        "source_round": int(source_round),
        "raw_pitfall_count": len(raw_pits),
        "raw_prop_count": len(raw_props),
        "kept_pitfall_count": len(kept_pits),
        "kept_prop_count": len(kept_props),
        "dropped_pitfalls": dropped_pits,
    }
    return kept_props, kept_pits, meta


def score_pitfall_retrieval_risk(
    pitfall: str,
    question_text: str,
    propositions: List[str],
    embed_model_path: str,
) -> float:
    st = _get_embedder(embed_model_path)
    if st is None:
        base = 0.5
    else:
        qv = _encode(st, [question_text.strip()[:4000]])[0]
        pv = _encode(st, [pitfall])[0]
        base = _cos(pv, qv)
        if propositions:
            prop_v = _encode(st, propositions)
            conflict = float(np.dot(prop_v, pv).max())
            base = base - 0.35 * conflict

    lower = pitfall.lower()
    if any(k in lower for k in _RISK_KEYWORDS):
        base += 0.08
    return float(base)


def adaptive_top_k_select(
    propositions: List[str],
    pitfalls: List[str],
    question_text: str,
    *,
    k_pos_max: int,
    k_neg_max: int,
    risk_threshold: float,
    embed_model_path: str,
) -> Tuple[List[str], List[str], Dict[str, Any]]:
    k_pos = max(0, int(k_pos_max))
    k_neg_cap = max(0, int(k_neg_max))

    props = list(propositions)[:k_pos] if k_pos else []

    if not pitfalls or k_neg_cap == 0:
        return props, [], {
            "k_pos": len(props),
            "k_neg": 0,
            "high_risk_pitfall_count": 0,
        }

    scored = [
        (
            score_pitfall_retrieval_risk(
                p, question_text, props, embed_model_path
            ),
            p,
        )
        for p in pitfalls
    ]
    scored.sort(key=lambda x: x[0], reverse=True)
    high_risk = [p for s, p in scored if s >= risk_threshold]
    k_neg = min(k_neg_cap, max(1, len(high_risk))) if high_risk else min(k_neg_cap, len(scored))
    pits = [p for _, p in scored[:k_neg]]

    return props, pits, {
        "k_pos": len(props),
        "k_neg": len(pits),
        "high_risk_pitfall_count": len(high_risk),
        "k_neg_adaptive": k_neg,
    }


def compress_proposition(text: str) -> str:
    lower = text.lower()
    body = _shorten(text)
    if "invariant" in lower or "always holds" in lower:
        return f"[+] Use invariant: {body}"
    if any(k in lower for k in ("constraint", "must", "check", "verify")):
        return f"[+] Check constraint: {body}"
    if any(k in lower for k in ("identity", "formula", "equation")):
        return f"[+] Key equation: {body}"
    return f"[+] Key fact: {body}"


def compress_pitfall(text: str) -> str:
    lower = text.lower()
    body = _shorten(text)
    if "assume" in lower or "assuming" in lower:
        return f"[-] Avoid assuming: {body}"
    if any(k in lower for k in ("transform", "substitut", "convert")):
        return f"[-] Invalid transform: {body}"
    if any(k in lower for k in ("divide", "zero", "domain")):
        return f"[-] Domain error: {body}"
    return f"[-] Avoid error: {body}"


def build_compressed_pos_content(propositions: List[str]) -> str:
    if not propositions:
        return ""
    return "\n".join(compress_proposition(p) for p in propositions)


def build_compressed_neg_content(pitfalls: List[str]) -> str:
    if not pitfalls:
        return ""
    return "\n".join(compress_pitfall(p) for p in pitfalls)


def load_and_aggregate_dropped_experiences(
    experience_dir: Any,
    original_idx: int,
    n_completions_to_use: int,
    question_text: str = "",
    retrieval: str = "first",
    embed_model_path: str = "",
    rerank_model_name: Optional[str] = None,
    rerank_pool_mult: int = 4,
    *,
    cross_query_parsed: Optional[List[Tuple[Dict[str, Any], str]]] = None,
    cross_query_doc_emb: Optional[np.ndarray] = None,
    k_pos_max: int = 8,
    k_neg_max: int = 4,
    risk_threshold: float = 0.45,
    experience_aggregation: str = "flat_top_k",
    aggregation_semantic_threshold: float = 0.82,
    aggregation_recency_head_fraction: float = 0.55,
) -> Optional[Dict[str, Any]]:
    """Load experience, apply retrieval-time adaptive top-k, return lists for render."""
    emb = embed_model_path or "/data/ppnm/models/all-MiniLM-L6-v2"
    pool_n = max(int(n_completions_to_use), int(k_pos_max) * 4, 16)

    data = cto.load_and_aggregate_raw_experiences(
        experience_dir,
        original_idx,
        pool_n,
        question_text=question_text,
        retrieval=retrieval,
        embed_model_path=emb,
        rerank_model_name=rerank_model_name,
        rerank_pool_mult=rerank_pool_mult,
        cross_query_parsed=cross_query_parsed,
        cross_query_doc_emb=cross_query_doc_emb,
        max_aggregated_propositions=0,
        max_aggregated_pitfalls=0,
        experience_aggregation=experience_aggregation,
        aggregation_semantic_threshold=aggregation_semantic_threshold,
        aggregation_recency_head_fraction=aggregation_recency_head_fraction,
    )
    if not data:
        return None

    props, pits, sel_meta = adaptive_top_k_select(
        data.get("verified_propositions", []) or [],
        data.get("critical_pitfalls", []) or [],
        question_text,
        k_pos_max=k_pos_max,
        k_neg_max=k_neg_max,
        risk_threshold=risk_threshold,
        embed_model_path=emb,
    )
    return {
        "verified_propositions": props,
        "critical_pitfalls": pits,
        "retrieval_meta": sel_meta,
    }


def build_expdrop_prompt_strings(
    experience_data: Dict[str, Any],
) -> Tuple[str, str]:
    props_str = build_compressed_pos_content(
        experience_data.get("verified_propositions", []) or []
    )
    pits_str = build_compressed_neg_content(
        experience_data.get("critical_pitfalls", []) or []
    )
    return props_str, pits_str
