#!/usr/bin/env python3
"""Unit tests for paper/rg_cto_method.md reliability formulas."""

from __future__ import annotations

import json
import math
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import rg_cto_common as rgc


class FakeEmbedder:
    """Deterministic unit-vector lookup table; unknown texts get a tiny residual dim."""

    def __init__(self, table):
        self.table = {k: np.asarray(v, dtype=np.float64) for k, v in table.items()}
        self.dim = len(next(iter(self.table.values())))

    def encode(self, texts, normalize_embeddings=True):
        rows = []
        for t in texts:
            if t in self.table:
                v = self.table[t].copy()
            else:
                v = np.zeros(self.dim, dtype=np.float64)
                v[-1] = 1.0
            if normalize_embeddings:
                n = float(np.linalg.norm(v))
                if n > 0:
                    v = v / n
            rows.append(v)
        return np.stack(rows, axis=0)


class ReliabilityFormulaTests(unittest.TestCase):
    def setUp(self):
        rgc.clear_embedder_cache()
        self.emb_path = "fake-embedder"
        self.table = {
            "pit_a": np.array([1.0, 0.0, 0.0]),
            "pit_b": np.array([0.0, 1.0, 0.0]),
            "traj_match": np.array([1.0, 0.0, 0.0]),
            "traj_other": np.array([0.0, 1.0, 0.0]),
            "question": np.array([1.0, 0.0, 0.0]),
            "eplus": np.array([0.6, 0.8, 0.0]),
        }
        rgc.set_embedder(self.emb_path, FakeEmbedder(self.table))

    def tearDown(self):
        rgc.clear_embedder_cache()

    def test_support_is_match_fraction_above_tau(self):
        u, matches = rgc.compute_support_scores(
            ["pit_a"],
            ["traj_match", "traj_match", "traj_other"],
            embed_model_path=self.emb_path,
            tau_match=0.5,
        )
        self.assertEqual(matches.shape, (1, 3))
        np.testing.assert_allclose(matches[0], [1.0, 1.0, 0.0])
        np.testing.assert_allclose(u, [2.0 / 3.0])

        u_strict, _ = rgc.compute_support_scores(
            ["pit_a"],
            ["traj_match", "traj_match", "traj_other"],
            embed_model_path=self.emb_path,
            tau_match=0.99,
        )
        np.testing.assert_allclose(u_strict, [2.0 / 3.0])

        u_none, _ = rgc.compute_support_scores(
            ["pit_a"],
            ["traj_other", "traj_other"],
            embed_model_path=self.emb_path,
            tau_match=0.5,
        )
        np.testing.assert_allclose(u_none, [0.0])

    def test_locality_is_product_of_cosines(self):
        l, rho_q, rho_plus = rgc.compute_locality_scores(
            ["pit_a"],
            "question",
            ["eplus"],
            embed_model_path=self.emb_path,
        )
        expected_rho_q = 1.0
        expected_rho_plus = 0.6
        np.testing.assert_allclose(rho_q, [expected_rho_q], atol=1e-8)
        np.testing.assert_allclose(rho_plus, [expected_rho_plus], atol=1e-8)
        np.testing.assert_allclose(l, [expected_rho_q * expected_rho_plus], atol=1e-8)

        l_empty, _, rho_empty = rgc.compute_locality_scores(
            ["pit_a"],
            "question",
            [],
            embed_model_path=self.emb_path,
        )
        np.testing.assert_allclose(rho_empty, [1.0])
        np.testing.assert_allclose(l_empty, [1.0])

    def test_weight_formula_and_delta_filter(self):
        out = rgc.compute_item_weights(
            [],
            "question",
            embed_model_path=self.emb_path,
            trajectories=["traj_match", "traj_other"],
            positive_evidence=["eplus"],
            pitfalls=["pit_a", "pit_b"],
            conflict_probs=[0.2, 0.0],
            tau_match=0.5,
            delta=0.4,
            lambda_u=0.5,
            lambda_l=0.5,
        )
        # pit_a: u=1/2, l=1.0*0.6=0.6, c=0.2
        w_a = (0.5**0.5) * (0.6**0.5) * (1.0 - 0.2)
        # pit_b: u=1/2, l=0*0.8=0, c=0 -> w=0
        details = {d["text"]: d for d in out["item_details"]}
        self.assertAlmostEqual(details["pit_a"]["u"], 0.5)
        self.assertAlmostEqual(details["pit_a"]["l"], 0.6)
        self.assertAlmostEqual(details["pit_a"]["c"], 0.2)
        self.assertAlmostEqual(details["pit_a"]["w"], w_a, places=6)
        self.assertAlmostEqual(details["pit_b"]["w"], 0.0)
        self.assertTrue(details["pit_a"]["kept"])
        self.assertFalse(details["pit_b"]["kept"])
        self.assertAlmostEqual(out["gate"], w_a)
        self.assertEqual(out["filtered_pitfalls"], ["pit_a"])

        alpha_r = rgc.effective_alpha(0.55, out, alpha_floor=0.02)
        self.assertAlmostEqual(alpha_r, 0.55 * w_a)

    def test_empty_filter_gives_zero_gate(self):
        out = rgc.compute_item_weights(
            [],
            "question",
            embed_model_path=self.emb_path,
            trajectories=["traj_other"],
            positive_evidence=["eplus"],
            pitfalls=["pit_a"],
            conflict_probs=[0.0],
            tau_match=0.5,
            delta=0.4,
            lambda_u=0.5,
            lambda_l=0.5,
        )
        # u=0 => w=0, filtered empty => g=0
        self.assertEqual(out["filtered_pitfalls"], [])
        self.assertEqual(out["gate"], 0.0)
        self.assertEqual(rgc.effective_alpha(0.55, out), 0.0)

    def test_conflict_probs_align_after_dedup(self):
        out = rgc.compute_item_weights(
            [],
            "question",
            embed_model_path=self.emb_path,
            trajectories=["traj_match", "traj_match"],
            positive_evidence=[],
            pitfalls=["pit_a", "pit_a", "pit_b"],
            conflict_probs=[0.3, 0.3, 0.9],
            tau_match=0.5,
            delta=0.0,
            lambda_u=1.0,
            lambda_l=1.0,
        )
        details = out["item_details"]
        self.assertEqual([d["text"] for d in details], ["pit_a", "pit_b"])
        self.assertAlmostEqual(details[0]["c"], 0.3)
        self.assertAlmostEqual(details[1]["c"], 0.9)


class ConflictParseTests(unittest.TestCase):
    def test_text_yes_no(self):
        self.assertEqual(rgc.parse_conflict_probability("YES"), 1.0)
        self.assertEqual(rgc.parse_conflict_probability("NO"), 0.0)
        self.assertEqual(rgc.parse_conflict_probability("yes."), 1.0)
        self.assertEqual(rgc.parse_conflict_probability("unknown"), 0.0)

    def test_logprob_softmax(self):
        lp = {
            "YES": SimpleNamespace(logprob=math.log(0.8)),
            "NO": SimpleNamespace(logprob=math.log(0.2)),
        }
        p = rgc.parse_conflict_probability("maybe", lp)
        self.assertAlmostEqual(p, 0.8, places=6)

    def test_logprob_yes_only(self):
        lp = {"YES": {"logprob": -0.1}}
        self.assertEqual(rgc.parse_conflict_probability("", lp), 1.0)


class TrajectoryLoadTests(unittest.TestCase):
    def test_load_previous_trajectories(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "3.json"
            path.write_text(
                json.dumps(
                    {
                        "completions": [
                            {
                                "reasoning_content": "long enough reasoning " * 8,
                                "text": "42",
                            },
                            {"reasoning_content": "", "text": "short answer"},
                        ]
                    }
                ),
                encoding="utf-8",
            )
            trajs = rgc.load_previous_trajectories(td, 3)
            self.assertEqual(len(trajs), 2)
            self.assertIn("long enough reasoning", trajs[0])
            self.assertEqual(trajs[1], "short answer")
            self.assertEqual(rgc.load_previous_trajectories(td, 99), [])


if __name__ == "__main__":
    unittest.main()
