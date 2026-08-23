#!/usr/bin/env python3
"""Risk-Gated CTO (RG-CTO): confidence-gated negative control for contrastive decoding.

Implements CG-CTO from the method doc:
  w(e) = clip(u(e) * l(e, q) * (1 - c(e)), 0, 1)
  filter w(e) < delta;  g = mean(w);  alpha_r = alpha_0 * g
"""

from __future__ import annotations

import math
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

import experience_drop_common as edc

_EMB_ST: Dict[str, Any] = {}


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


def _snippet(text: str, max_chars: int = 1200) -> str:
    t = text.strip()
    if len(t) <= max_chars:
        return t
    head = t[: max_chars // 2]
    tail = t[-max_chars // 2 :]
    return head + "\n...\n" + tail


def extract_records_pitfalls(
    records: List[Dict[str, Any]],
) -> Tuple[List[str], Dict[str, Dict[str, Any]]]:
    pits, _props = edc.extract_strings_from_records(records)
    tally = edc.tally_support(pits)
    unique = [v["text"] for v in tally.values()]
    return unique, tally


def compute_support_score(support_count: int, *, min_support: int = 2) -> float:
    """u(e): multi-rollout support for pitfall e."""
    denom = max(int(min_support), 1)
    return float(min(1.0, support_count / denom))


def compute_locality_item(
    pitfall: str,
    question_text: str,
    *,
    embed_model_path: str,
) -> float:
    """l(e, q): semantic relevance of pitfall e to the current question."""
    st = _get_embedder(embed_model_path)
    if st is None:
        return 0.55
    qv = _encode(st, [question_text.strip()[:4000]])[0]
    pv = _encode(st, [pitfall])[0]
    sim = float(np.dot(pv, qv))
    return float(max(0.0, min(1.0, (sim - 0.15) / 0.70)))


def compute_conflict_risk_item(
    pitfall: str,
    propositions: Sequence[str],
    top_cluster_texts: Sequence[str],
    *,
    embed_model_path: str,
) -> float:
    """c(e): conflict between pitfall e and positive evidence / majority cluster."""
    st = _get_embedder(embed_model_path)
    if st is None:
        return 0.35

    pv = _encode(st, [pitfall])[0]
    cluster_conflict = 0.0
    cluster_snips = [_snippet(t) for t in top_cluster_texts if t.strip()]
    if cluster_snips:
        cv = _encode(st, cluster_snips)
        cluster_conflict = float(np.dot(pv, cv.T).max()) if cv.size else 0.0

    prop_conflict = 0.0
    if propositions:
        prop_v = _encode(st, list(propositions))
        prop_conflict = float(np.dot(pv, prop_v.T).max()) if prop_v.size else 0.0

    risk = 0.55 * cluster_conflict + 0.45 * prop_conflict
    return float(max(0.0, min(1.0, risk)))


def compute_pilot_decoding_risk(
    *,
    pos_scores: Sequence[float],
    neg_scores: Sequence[float],
    cluster_majority: float,
    alpha_0: float,
) -> Tuple[float, Dict[str, Any]]:
    """Round-level pilot proxy: does contrastive reranking harm majority consistency?"""
    n = len(pos_scores)
    if n == 0:
        return 0.0, {"pilot_n": 0}

    order_pos = sorted(range(n), key=lambda i: pos_scores[i], reverse=True)
    order_cto = sorted(
        range(n),
        key=lambda i: float(pos_scores[i] - alpha_0 * neg_scores[i]),
        reverse=True,
    )
    top_pos = order_pos[0]
    top_cto = order_cto[0]

    consistency_drop = 0.0
    if top_pos != top_cto:
        consistency_drop = max(0.15, cluster_majority)

    neg_strength = 0.0
    if neg_scores:
        pos_best_neg = float(neg_scores[top_pos])
        neg_strength = float(max(0.0, min(1.0, pos_best_neg / 40.0)))

    risk = max(consistency_drop * 0.75, neg_strength * 0.55)
    meta = {
        "pilot_n": int(n),
        "cluster_majority": float(cluster_majority),
        "top_pos_idx": int(top_pos),
        "top_cto_idx": int(top_cto),
        "consistency_drop": float(consistency_drop),
        "neg_strength": float(neg_strength),
    }
    return float(max(0.0, min(1.0, risk))), meta


def compute_item_weights(
    records: List[Dict[str, Any]],
    question_text: str,
    *,
    embed_model_path: str,
    min_support: int = 2,
    delta: float = 0.15,
    pilot_meta: Optional[Dict[str, Any]] = None,
    pilot_risk_weight: float = 0.35,
) -> Dict[str, Any]:
    """
    Per-negative-item weights and round-level gate g^(r).

    Returns dict with gate, alpha_multiplier, filtered pitfalls, and diagnostics.
    """
    pitfalls, tally = extract_records_pitfalls(records)
    props, _ = edc.extract_strings_from_records(records)
    prop_unique = list(dict.fromkeys(props))

    top_cluster_texts: List[str] = []
    pilot_risk = 0.0
    pilot_detail: Dict[str, Any] = {}
    if pilot_meta:
        top_cluster_texts = list(pilot_meta.get("top_cluster_texts") or [])
        pilot_risk, pilot_detail = compute_pilot_decoding_risk(
            pos_scores=pilot_meta.get("pos_scores") or [],
            neg_scores=pilot_meta.get("neg_scores") or [],
            cluster_majority=float(pilot_meta.get("cluster_majority", 0.0)),
            alpha_0=float(pilot_meta.get("alpha_0", 0.7)),
        )

    item_details: List[Dict[str, Any]] = []
    weights: List[float] = []
    for entry in tally.values():
        text = entry.get("text", "")
        support = int(entry.get("support_count", 0))
        u_e = compute_support_score(support, min_support=min_support)
        l_e = compute_locality_item(text, question_text, embed_model_path=embed_model_path)
        c_e = compute_conflict_risk_item(
            text,
            prop_unique,
            top_cluster_texts,
            embed_model_path=embed_model_path,
        )
        if pilot_risk > 0.0:
            c_e = float(max(0.0, min(1.0, c_e + pilot_risk_weight * pilot_risk)))
        w_e = float(max(0.0, min(1.0, u_e * l_e * (1.0 - c_e))))
        kept = w_e >= float(delta)
        item_details.append(
            {
                "text": text,
                "support_count": support,
                "u": float(u_e),
                "l": float(l_e),
                "c": float(c_e),
                "w": float(w_e),
                "kept": bool(kept),
            }
        )
        if kept:
            weights.append(w_e)

    gate = float(sum(weights) / len(weights)) if weights else 0.0
    filtered_pitfalls = [d["text"] for d in item_details if d["kept"]]

    return {
        "gate": gate,
        "alpha_multiplier": gate,
        "delta": float(delta),
        "n_pitfalls_total": int(len(pitfalls)),
        "n_pitfalls_kept": int(len(filtered_pitfalls)),
        "filtered_pitfalls": filtered_pitfalls,
        "item_details": item_details,
        "pilot_risk": float(pilot_risk),
        "pilot_detail": pilot_detail,
        "n_records": int(len(records)),
    }


def effective_alpha(
    alpha_0: float,
    gate_factors: Dict[str, Any],
    *,
    alpha_floor: float = 0.02,
) -> float:
    """alpha_r = alpha_0 * g^(r); zero when below floor."""
    alpha_r = float(alpha_0 * gate_factors.get("gate", 0.0))
    if alpha_r < alpha_floor:
        return 0.0
    return alpha_r


def effective_alpha_summary(
    alpha_values: Sequence[float],
    *,
    alpha_floor: float = 0.02,
) -> Dict[str, float]:
    if not alpha_values:
        return {"alpha_mean": 0.0, "alpha_min": 0.0, "alpha_max": 0.0, "positive_only": 1.0}
    arr = np.array(alpha_values, dtype=np.float64)
    positive_only = float(np.mean(arr < alpha_floor))
    return {
        "alpha_mean": float(arr.mean()),
        "alpha_min": float(arr.min()),
        "alpha_max": float(arr.max()),
        "positive_only": positive_only,
    }
