#!/usr/bin/env python3
"""Deterministic synthetic agent budgets and routing; Python standard library.

Run: python3 examples/agent_budget_lab.py
All costs/success labels are handcrafted teaching assumptions, not LLM results.
"""
from __future__ import annotations

import json
from dataclasses import dataclass


@dataclass
class Budget:
    limit: int
    spent: int = 0
    final_reserve: int = 100

    def admit(self, input_units: int, max_output_units: int) -> bool:
        """Check BEFORE dispatch; positive output allowance is required."""
        if input_units < 0 or max_output_units <= 0:
            raise ValueError("invalid call reservation")
        return self.spent + input_units + max_output_units + self.final_reserve <= self.limit

    def charge(self, input_units: int, actual_output_units: int) -> None:
        if input_units < 0 or actual_output_units < 0:
            raise ValueError("negative usage is invalid")
        if self.spent + input_units + actual_output_units > self.limit:
            raise RuntimeError("budget exceeded; do not silently clamp usage")
        self.spent += input_units + actual_output_units


def budget_demo() -> dict:
    budget = Budget(limit=1000)
    events = []
    for name, input_units, output_cap, actual_output in [
        ("search", 200, 150, 80),
        ("inspect", 300, 150, 90),
        ("retry", 250, 150, 70),
    ]:
        admitted = budget.admit(input_units, output_cap)
        events.append({"call": name, "admitted": admitted, "spent_before": budget.spent})
        if admitted:
            if actual_output > output_cap:
                raise RuntimeError("backend violated output reservation")
            budget.charge(input_units, actual_output)
    budget.charge(70, 30)  # A final call whose full input+output fits the reserve.
    assert budget.spent == 770
    return {"events": events, "total_units": budget.spent,
            "limit": budget.limit, "final_status": "stopped_with_partial_answer"}


def routing_demo() -> list[dict]:
    # Observable difficulty feature is synthetic; it is NOT the success label.
    tasks = [{"feature": i / 10, "weak_ok": i < 6, "strong_ok": i < 9} for i in range(10)]
    result = []
    for threshold in (0.0, 0.3, 0.6, 0.9, 1.0):
        total_units, success, strong_calls = 0, 0, 0
        for task in tasks:
            use_strong = task["feature"] >= threshold
            total_units += 20 + (500 if use_strong else 200)  # Router call included.
            strong_calls += use_strong
            success += task["strong_ok"] if use_strong else task["weak_ok"]
        result.append({"threshold": threshold, "strong_calls": strong_calls,
                       "success_rate": success / len(tasks),
                       "mean_total_units": total_units / len(tasks)})
    return result


def communication_demo(workers: int = 4, rounds: int = 3, message_units: int = 100) -> dict:
    # Each sender generates ONE message per round. Every other worker reads it
    # exactly once; no history replay, coordinator, or work tokens are included.
    generated = workers * rounds * message_units
    read = workers * (workers - 1) * rounds * message_units
    # Star: coordinator sends one distinct assignment and receives one result
    # per worker per round. Every message has one generation and one read.
    star_total = 4 * workers * rounds * message_units
    return {"workers": workers, "rounds": rounds, "message_units": message_units,
            "all_to_all_generated_units": generated,
            "all_to_all_read_units": read,
            "all_to_all_total_units": generated + read,
            "star_total_units": star_total,
            "caution": "does not compare task quality; full replay adds more input"}


def filtered_tool_demo() -> dict:
    rows = [{"id": i, "status": "failed" if i in (7, 81) else "passed",
             "log": "verbose detail " * 20} for i in range(100)]
    result = {"total": len(rows), "failed": [r["id"] for r in rows if r["status"] == "failed"]}
    full = len(json.dumps(rows).split())
    compact = len(json.dumps(result).split())
    assert result["failed"] == [7, 81]
    return {"raw_whitespace_units": full, "filtered_whitespace_units": compact,
            "result": result, "method": "exact programmatic aggregation"}


def resource_demo() -> dict:
    # One-position speculative sampling, computed exactly (not Monte Carlo).
    target, draft = [0.1, 0.6, 0.3], [0.5, 0.3, 0.2]
    accepted_mass = [min(p, q) for p, q in zip(target, draft)]
    residual = [max(p - q, 0.0) for p, q in zip(target, draft)]
    rejection_mass = 1.0 - sum(accepted_mass)
    recovered = [a + rejection_mass * r / sum(residual)
                 for a, r in zip(accepted_mass, residual)]
    assert all(abs(a - b) < 1e-12 for a, b in zip(target, recovered))
    length, diffusion_steps = 128, 16
    parameters = 1_000_000_000
    return {
        "speculative_target_distribution": target,
        "speculative_recovered_distribution": recovered,
        "draft_acceptance_probability": sum(accepted_mass),
        "final_output_positions_both_methods": length,
        "ar_new_position_visits_ignoring_prefill": length,
        "full_sequence_diffusion_position_visits": length * diffusion_steps,
        "diffusion_forward_calls": diffusion_steps,
        "warning": "position visits are not FLOPs or wall-clock latency",
        "one_billion_weight_bytes_16bit": parameters * 16 // 8,
        "one_billion_weight_bytes_4bit_idealized": parameters * 4 // 8,
    }


def main() -> None:
    print(json.dumps({"label": "SYNTHETIC DEMONSTRATION; NOT AGENT BENCHMARK RESULTS",
                      "budget": budget_demo(), "routing": routing_demo(),
                      "communication": communication_demo(),
                      "tool_filter": filtered_tool_demo(),
                      "resources": resource_demo()}, indent=2))


if __name__ == "__main__":
    main()
