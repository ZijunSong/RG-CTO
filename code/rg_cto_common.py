#!/usr/bin/env python3
"""Reliability-Gated CTO (RG-CTO) estimators from paper/rg_cto_method.md.

Support:
    m(e, y) = I(Match(e, y) > tau_match)
    u(e)    = (1/N) sum_i m(e, y_i)

Locality:
    l(e, q) = rho(e, q) * rho(e, E_+)
    rho(a,b) = cosine(psi(a), psi(b))

Conflict:
    c(e) = p_theta(conflict | T_conf)   # frozen-model Yes/No probability

Reliability:
    w(e) = clip(u(e)^lambda_u * l(e,q)^lambda_l * (1-c(e)), 0, 1)
    keep e iff w(e) >= delta
    g^(r) = mean_{kept} w(e)   (0 if none kept)
    alpha_r = alpha_0 * g^(r)
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

import experience_drop_common as edc

_EMB_ST: Dict[str, Any] = {}

_CONFLICT_SYSTEM_PROMPT = (
    "You estimate whether suppressing a negative reasoning pattern would conflict "
    "with useful positive evidence. Reply with a single token: YES or NO."
)

_CONFLICT_USER_TEMPLATE = """Positive evidence T_conf:
{t_conf}

Negative experience e:
{pitfall}

Would suppressing this negative experience conflict with a reasoning direction
that is strongly supported by the positive evidence?
Answer YES if it would conflict (the negative item may actually be a valid path).
Answer NO if suppression is safe.
"""


def unique_texts(items: Sequence[Any]) -> List[str]:
    return list(
        dict.fromkeys(p.strip() for p in items if isinstance(p, str) and p.strip())
    )


def set_embedder(embed_model_path: str, embedder: Any) -> None:
    """Inject an embedder that implements encode(texts, normalize_embeddings=True)."""
    _EMB_ST[embed_model_path] = embedder


def clear_embedder_cache() -> None:
    _EMB_ST.clear()


def _get_embedder(embed_model_path: str):
    st = _EMB_ST.get(embed_model_path)
    if st is not None:
        return st
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        return None
    st = SentenceTransformer(embed_model_path)
    _EMB_ST[embed_model_path] = st
    return st


def _encode(st: Any, texts: List[str]) -> np.ndarray:
    if not texts:
        return np.zeros((0, 1), dtype=np.float64)
    return np.asarray(st.encode(texts, normalize_embeddings=True), dtype=np.float64)


def _snippet(text: str, max_chars: int = 1500) -> str:
    t = (text or "").strip()
    if len(t) <= max_chars:
        return t
    head = t[: max_chars // 2]
    tail = t[-max_chars // 2 :]
    return head + "\n...\n" + tail


def _clip01(x: float) -> float:
    return float(max(0.0, min(1.0, x)))


def rho_vecs(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity of already-normalized vectors, clipped to [0, 1]."""
    if a.size == 0 or b.size == 0:
        return 0.0
    return _clip01(float(np.dot(a, b)))


def extract_records_pitfalls(
    records: List[Dict[str, Any]],
) -> Tuple[List[str], Dict[str, Dict[str, Any]]]:
    pits, _props = edc.extract_strings_from_records(records)
    tally = edc.tally_support(pits)
    unique = [v["text"] for v in tally.values()]
    return unique, tally


def load_previous_trajectories(answer_dir: Optional[str], original_idx: int) -> List[str]:
    """Load previous rollouts y_1..y_N from a standard sampling/CTO result JSON."""
    if not answer_dir:
        return []
    path = Path(answer_dir) / f"{original_idx}.json"
    if not path.exists():
        return []
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return []
    texts: List[str] = []
    for comp in data.get("completions") or []:
        if not isinstance(comp, dict):
            continue
        reason = (comp.get("reasoning_content") or "").strip()
        answer = (comp.get("text") or "").strip()
        body = reason if len(reason) > 50 else (reason + "\n" + answer).strip()
        if not body:
            body = answer
        if body:
            texts.append(body)
    return texts


def compute_support_scores(
    pitfalls: Sequence[str],
    trajectories: Sequence[str],
    *,
    embed_model_path: str,
    tau_match: float,
    pitfall_vecs: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """u(e) = (1/N) sum_i I(Match(e, y_i) > tau_match).

    Returns (u_scores [E], match_matrix [E, N]).
    """
    e_list = [p.strip() for p in pitfalls]
    n_e = len(e_list)
    y_snips = [_snippet(y) for y in trajectories if (y or "").strip()]
    n_y = len(y_snips)
    if n_e == 0:
        return np.zeros((0,), dtype=np.float64), np.zeros((0, n_y), dtype=np.float64)
    if n_y == 0:
        return np.zeros((n_e,), dtype=np.float64), np.zeros((n_e, 0), dtype=np.float64)

    st = _get_embedder(embed_model_path)
    if st is None:
        return np.zeros((n_e,), dtype=np.float64), np.zeros((n_e, n_y), dtype=np.float64)

    ev = pitfall_vecs if pitfall_vecs is not None else _encode(st, e_list)
    yv = _encode(st, y_snips)
    sims = ev @ yv.T
    matches = (sims > float(tau_match)).astype(np.float64)
    u_scores = matches.mean(axis=1)
    return u_scores, matches


def compute_locality_scores(
    pitfalls: Sequence[str],
    question_text: str,
    positive_evidence: Sequence[str],
    *,
    embed_model_path: str,
    pitfall_vecs: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """l(e,q) = rho(e,q) * rho(e, E_+). Returns (l, rho_q, rho_eplus)."""
    e_list = [p.strip() for p in pitfalls]
    n_e = len(e_list)
    zeros = np.zeros((n_e,), dtype=np.float64)
    if n_e == 0:
        return zeros, zeros, zeros

    st = _get_embedder(embed_model_path)
    if st is None:
        # Neutral fallback when embeddings are unavailable.
        ones = np.ones((n_e,), dtype=np.float64)
        return ones * 0.55, ones * 0.55, ones

    ev = pitfall_vecs if pitfall_vecs is not None else _encode(st, e_list)
    qv = _encode(st, [_snippet(question_text, max_chars=4000)])[0]
    rho_q = np.clip(ev @ qv, 0.0, 1.0)

    eplus = [p.strip() for p in positive_evidence if isinstance(p, str) and p.strip()]
    if eplus:
        pv = _encode(st, [_snippet(p, max_chars=800) for p in eplus])
        eplus_vec = pv.mean(axis=0)
        norm = float(np.linalg.norm(eplus_vec))
        if norm > 0:
            eplus_vec = eplus_vec / norm
        rho_plus = np.clip(ev @ eplus_vec, 0.0, 1.0)
    else:
        rho_plus = np.ones((n_e,), dtype=np.float64)

    locality = rho_q * rho_plus
    return locality, rho_q, rho_plus


def build_conflict_prompt(pitfall: str, positive_evidence: Sequence[str], *, max_items: int = 16) -> str:
    items = [p.strip() for p in positive_evidence if isinstance(p, str) and p.strip()][:max_items]
    if items:
        t_conf = "\n".join(f"- {_snippet(p, max_chars=400)}" for p in items)
    else:
        t_conf = "(none)"
    return _CONFLICT_USER_TEMPLATE.format(t_conf=t_conf, pitfall=_snippet(pitfall, max_chars=600))


def parse_conflict_probability(text: str, first_step_logprobs: Any = None) -> float:
    """Map frozen-model output to c(e) in [0, 1]. Prefer YES/NO token logprobs."""
    yes_lp: Optional[float] = None
    no_lp: Optional[float] = None
    if first_step_logprobs:
        items = first_step_logprobs.items() if isinstance(first_step_logprobs, dict) else []
        for tok, info in items:
            token = str(tok).strip().upper()
            lp = getattr(info, "logprob", None)
            if lp is None and isinstance(info, dict):
                lp = info.get("logprob")
            if lp is None:
                continue
            if token in {"YES", "Y"} or token.endswith("YES"):
                yes_lp = float(lp) if yes_lp is None else max(yes_lp, float(lp))
            if token in {"NO", "N"} or token.endswith("NO"):
                no_lp = float(lp) if no_lp is None else max(no_lp, float(lp))
        if yes_lp is not None and no_lp is not None:
            m = max(yes_lp, no_lp)
            ey = math.exp(yes_lp - m)
            en = math.exp(no_lp - m)
            return float(ey / max(ey + en, 1e-12))
        if yes_lp is not None:
            return 1.0
        if no_lp is not None:
            return 0.0

    raw = (text or "").strip().upper()
    raw = re.sub(r"[^A-Z]", " ", raw).strip()
    if raw.startswith("YES"):
        return 1.0
    if raw.startswith("NO"):
        return 0.0
    return 0.0


def score_conflict_probs_with_llm(
    llm: Any,
    tokenizer: Any,
    pitfalls: Sequence[str],
    positive_evidence: Sequence[str],
    *,
    max_model_len: Optional[int] = None,
    vllm_score_batch_size: int = 8,
) -> List[float]:
    """c(e) = p_theta(conflict | T_conf) via frozen-model YES/NO generation."""
    del max_model_len
    from vllm import SamplingParams

    if not pitfalls or llm is None or tokenizer is None:
        return [0.0] * len(pitfalls)

    prompts: List[str] = []
    for pit in pitfalls:
        messages = [
            {"role": "system", "content": _CONFLICT_SYSTEM_PROMPT},
            {"role": "user", "content": build_conflict_prompt(pit, positive_evidence)},
        ]
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        prompts.append(prompt)

    params = SamplingParams(
        max_tokens=8,
        n=1,
        temperature=0.0,
        top_p=1.0,
        top_k=-1,
        logprobs=8,
    )
    probs: List[float] = []
    bs = max(1, int(vllm_score_batch_size))
    for start in range(0, len(prompts), bs):
        batch = prompts[start : start + bs]
        outs = llm.generate(batch, params)
        for out in outs:
            gen = out.outputs[0] if getattr(out, "outputs", None) else out
            text = getattr(gen, "text", "") or ""
            lp0 = None
            logprobs = getattr(gen, "logprobs", None)
            if logprobs:
                lp0 = logprobs[0]
            probs.append(parse_conflict_probability(text, lp0))
    return probs


def compute_item_weights(
    records: List[Dict[str, Any]],
    question_text: str,
    *,
    embed_model_path: str,
    trajectories: Optional[Sequence[str]] = None,
    positive_evidence: Optional[Sequence[str]] = None,
    pitfalls: Optional[Sequence[str]] = None,
    conflict_probs: Optional[Sequence[float]] = None,
    tau_match: float = 0.8,
    delta: float = 0.4,
    lambda_u: float = 0.5,
    lambda_l: float = 0.5,
    min_support: int = 2,
    pilot_meta: Optional[Dict[str, Any]] = None,
    pilot_risk_weight: float = 0.35,
) -> Dict[str, Any]:
    """Item-level w(e), filtered negatives, and round-level gate g^(r).

    Unused legacy kwargs (min_support, pilot_meta, pilot_risk_weight) are
    accepted for call-site compatibility and ignored by the paper formulas.
    """
    del min_support, pilot_meta, pilot_risk_weight

    if pitfalls is None:
        pitfalls, _tally = extract_records_pitfalls(records)
        conflict_probs = None if conflict_probs is None else list(conflict_probs)
    else:
        raw = [p.strip() for p in pitfalls if isinstance(p, str) and p.strip()]
        if conflict_probs is not None and len(conflict_probs) == len(raw):
            merged: Dict[str, float] = {}
            for text, c_e in zip(raw, conflict_probs):
                if text not in merged:
                    merged[text] = float(c_e)
            pitfalls = list(merged.keys())
            conflict_probs = [merged[t] for t in pitfalls]
        else:
            pitfalls = unique_texts(raw)
    if positive_evidence is None:
        _pits, props = edc.extract_strings_from_records(records)
        positive_evidence = unique_texts(props)
    eplus = unique_texts(positive_evidence or [])
    trajs = list(trajectories or [])

    st = _get_embedder(embed_model_path)
    pitfall_vecs = _encode(st, pitfalls) if st is not None and pitfalls else None

    u_scores, match_matrix = compute_support_scores(
        pitfalls,
        trajs,
        embed_model_path=embed_model_path,
        tau_match=tau_match,
        pitfall_vecs=pitfall_vecs,
    )
    l_scores, rho_q, rho_plus = compute_locality_scores(
        pitfalls,
        question_text,
        eplus,
        embed_model_path=embed_model_path,
        pitfall_vecs=pitfall_vecs,
    )

    if conflict_probs is None:
        c_scores = np.zeros((len(pitfalls),), dtype=np.float64)
    else:
        c_scores = np.array(
            [float(conflict_probs[i]) if i < len(conflict_probs) else 0.0 for i in range(len(pitfalls))],
            dtype=np.float64,
        )
        c_scores = np.clip(c_scores, 0.0, 1.0)

    item_details: List[Dict[str, Any]] = []
    weights: List[float] = []
    n_y = int(match_matrix.shape[1]) if match_matrix.size else 0
    for i, text in enumerate(pitfalls):
        u_e = float(u_scores[i]) if i < len(u_scores) else 0.0
        l_e = float(l_scores[i]) if i < len(l_scores) else 0.0
        c_e = float(c_scores[i]) if i < len(c_scores) else 0.0
        u_term = u_e ** float(lambda_u) if u_e > 0.0 else 0.0
        l_term = l_e ** float(lambda_l) if l_e > 0.0 else 0.0
        w_e = _clip01(u_term * l_term * (1.0 - c_e))
        kept = w_e >= float(delta)
        n_matched = int(match_matrix[i].sum()) if n_y else 0
        item_details.append(
            {
                "text": text,
                "support_count": n_matched,
                "n_matched_trajectories": n_matched,
                "n_trajectories": n_y,
                "u": u_e,
                "l": l_e,
                "rho_q": float(rho_q[i]) if i < len(rho_q) else 0.0,
                "rho_eplus": float(rho_plus[i]) if i < len(rho_plus) else 0.0,
                "c": c_e,
                "tau_match": float(tau_match),
                "lambda_u": float(lambda_u),
                "lambda_l": float(lambda_l),
                "w": w_e,
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
        "tau_match": float(tau_match),
        "lambda_u": float(lambda_u),
        "lambda_l": float(lambda_l),
        "n_trajectories": n_y,
        "n_pitfalls_total": int(len(pitfalls)),
        "n_pitfalls_kept": int(len(filtered_pitfalls)),
        "filtered_pitfalls": filtered_pitfalls,
        "item_details": item_details,
        "pilot_risk": 0.0,
        "pilot_detail": {},
        "n_records": int(len(records)),
    }


def effective_alpha(
    alpha_0: float,
    gate_factors: Dict[str, Any],
    *,
    alpha_floor: float = 0.02,
) -> float:
    """alpha_r = alpha_0 * g^(r); treat tiny values as positive-only."""
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
