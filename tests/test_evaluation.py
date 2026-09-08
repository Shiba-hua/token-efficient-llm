"""Test statistical failure modes, not live benchmark reproduction."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))
from evaluation_lab import (Call, total_tokens, aucoaa, harmonic_score,
                            pareto_indices, appworld_metrics, paired_bootstrap)


class EvaluationContracts(unittest.TestCase):
    def test_cache_is_subset_and_missing_usage_fails(self):
        self.assertEqual(total_tokens([Call(100, 20, 80)]), 120)
        with self.assertRaises(ValueError):
            total_tokens([Call(100, None)])
        with self.assertRaises(ValueError):
            Call(100, 20, 101)

    def test_deployment_selector_is_counted_grader_separate(self):
        calls = [Call(100, 20), Call(30, 5), Call(50, 5, account="evaluation")]
        self.assertEqual(total_tokens(calls), 155)
        self.assertEqual(total_tokens(calls, "evaluation"), 55)

    def test_aucoaa_keeps_accuracy_length_pairing(self):
        self.assertEqual(aucoaa([1, 0], [0, 1000]), .5)
        self.assertEqual(aucoaa([0, 1], [0, 1000]), 0)
        self.assertEqual(aucoaa([1], [1000]), 0)
        self.assertEqual(harmonic_score(0, 0), 0)

    def test_pareto_does_not_trade_quality_for_cost(self):
        self.assertEqual(pareto_indices([(100, .8), (80, .7), (120, .6)]), [0, 1])

    def test_scenario_requires_all_variants_and_binary_task_score(self):
        expected = {"s": ["s_1", "s_2", "s_3"]}
        self.assertEqual(appworld_metrics({"s_1": True, "s_2": True, "s_3": False}, expected), (2/3, 0))
        with self.assertRaises(ValueError):
            appworld_metrics({"s_1": True}, expected)
        with self.assertRaises(ValueError):
            appworld_metrics({"s_1": .8, "s_2": True, "s_3": True}, expected)

    def test_bootstrap_pairs_by_id_and_rejects_dropped_failures(self):
        self.assertEqual(paired_bootstrap({"a": 1, "b": 0}, {"b": 0, "a": 1})[0], 0)
        with self.assertRaises(ValueError):
            paired_bootstrap({"a": 1, "b": 0}, {"a": 1})


if __name__ == "__main__":
    unittest.main()
