"""CPU teaching lab: accounting and metrics, with synthetic outcomes only.

Run from the repository root: python3 examples/evaluation_lab.py
No model/API call is made. This is not an implementation of an official harness.
"""
from dataclasses import dataclass
from math import isfinite, log1p
from random import Random
from statistics import mean


@dataclass(frozen=True)
class Call:
    input_tokens: int | None
    output_tokens: int | None
    cached_input_tokens: int = 0
    account: str = "method"

    def __post_init__(self):
        for value in (self.input_tokens, self.output_tokens, self.cached_input_tokens):
            if value is not None and (type(value) is not int or value < 0):
                raise ValueError("Token counts must be nonnegative integers or unknown.")
        if self.account not in {"method", "development", "evaluation"}:
            raise ValueError("Unknown account.")
        if self.input_tokens is not None and self.cached_input_tokens > self.input_tokens:
            raise ValueError("Cached input is a subset of input, not extra tokens.")


def total_tokens(calls, account="method"):
    if account not in {"method", "development", "evaluation"}:
        raise ValueError("Unknown account.")
    selected = [c for c in calls if c.account == account]
    if any(c.input_tokens is None or c.output_tokens is None for c in selected):
        raise ValueError("Missing usage: report unknown instead of silently using zero.")
    return sum(c.input_tokens + c.output_tokens for c in selected)


def _unit_scores(values):
    values = list(values)
    if not values or any(not isfinite(x) or not 0 <= x <= 1 for x in values):
        raise ValueError("Scores must be a nonempty sequence in [0, 1].")
    return values


def ock_score(accuracy, average_tokens):
    """Explicit convention: accuracy in [0,1], output-token mean for paper formula."""
    _unit_scores([accuracy])
    if not isfinite(average_tokens) or average_tokens < 0:
        raise ValueError("Invalid token mean.")
    return 100 * accuracy - 10 * log1p(average_tokens / 10000)


def aucoaa(correct, thinking_lengths, t_max=1000):
    """Per-response pairing; differs from multiplying two per-task means."""
    correct = _unit_scores(correct)
    thinking_lengths = list(thinking_lengths)
    if len(correct) != len(thinking_lengths) or not isfinite(t_max) or t_max <= 0:
        raise ValueError("Require aligned responses and positive threshold.")
    if any(not isfinite(x) or x < 0 for x in thinking_lengths):
        raise ValueError("Lengths cannot be negative or missing.")
    return mean(c * max(0, 1 - length / t_max)
                for c, length in zip(correct, thinking_lengths))


def harmonic_score(a, u):
    _unit_scores([a, u])
    return 0.0 if a + u == 0 else 2 * a * u / (a + u)


def pareto_indices(points):
    """Indices of nondominated (cost, score) observations; no interpolation."""
    for cost, score in points:
        if not isfinite(cost) or cost < 0:
            raise ValueError("Invalid cost.")
        _unit_scores([score])
    return [i for i, (cost, score) in enumerate(points)
            if not any(c <= cost and s >= score and (c < cost or s > score)
                       for c, s in points)]


def appworld_metrics(success_by_task, expected_scenarios):
    """Explicit scenario membership prevents scoring an incomplete easy subset."""
    members = [task for tasks in expected_scenarios.values() for task in tasks]
    if (not members or any(not tasks for tasks in expected_scenarios.values())
            or len(members) != len(set(members))
            or set(members) != set(success_by_task)):
        raise ValueError("Provide every expected task exactly once.")
    if any(type(value) is not bool for value in success_by_task.values()):
        raise ValueError("Use final all-tests-pass booleans, not partial test fractions.")
    tgc = mean(success_by_task.values())
    sgc = mean(all(success_by_task[t] for t in tasks)
               for tasks in expected_scenarios.values())
    return tgc, sgc


def paired_bootstrap(a_by_task, b_by_task, repeats=2000, seed=7):
    """Percentile interval for B-A, resampling aligned tasks.

    Assumes tasks are independent sampling units. Cluster variants by scenario
    BEFORE calling, or use a cluster bootstrap in a real experiment.
    """
    if not a_by_task or set(a_by_task) != set(b_by_task):
        raise ValueError("Both methods must cover the identical task set.")
    if type(repeats) is not int or repeats < 2:
        raise ValueError("Need at least two resamples.")
    ids = sorted(a_by_task)
    a = _unit_scores(a_by_task[t] for t in ids)
    b = _unit_scores(b_by_task[t] for t in ids)
    differences = [y - x for x, y in zip(a, b)]
    rng = Random(seed)
    draws = sorted(mean(rng.choices(differences, k=len(ids))) for _ in range(repeats))
    return mean(differences), (draws[int(.025 * (repeats-1))],
                               draws[int(.975 * (repeats-1))])


def main():
    print("SYNTHETIC TEACHING EXAMPLES — no LLM performance measured")
    calls = [Call(2000, 500, 1500), Call(1000, 100), Call(800, 200),
             Call(900, 80, account="evaluation")]
    print("Method total:", total_tokens(calls), "grader:", total_tokens(calls, "evaluation"))
    print("OckScore(80%, 1000 output tokens):", round(ock_score(.8, 1000), 4))
    print("AUCOAA, paired correct/length:", aucoaa([1, 0], [0, 1000]))
    print("Product of means (different statistic):", .5 * (1 - 500/1000))
    print("OTB harmonic score:", harmonic_score(.5, .8))
    points = [(1000, .7), (800, .68), (1200, .65), (1400, .75)]
    print("Pareto indices:", pareto_indices(points))
    print("TGC/SGC:", appworld_metrics({"s_1": True, "s_2": True, "s_3": False},
                                         {"s": ["s_1", "s_2", "s_3"]}))
    a = {str(i): int(i < 12) for i in range(20)}
    b = {str(i): int(i < 14) for i in range(20)}
    print("Paired difference and illustrative 95% interval:", paired_bootstrap(a, b))
    # Failures still count: costs 100,100,1000 with success 1,1,0.
    print("All-task cost:", mean([100, 100, 1000]), "success-only cost:", mean([100, 100]))


if __name__ == "__main__":
    main()
