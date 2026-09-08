"""Synthetic arithmetic for the efficient reasoning reading module; no LLM calls."""

from math import ceil

def retain_units(units, importance, keep_fraction):
    if len(units) != len(importance) or not 0 < keep_fraction <= 1:
        raise ValueError("invalid lengths or retention fraction")
    k = ceil(len(units) * keep_fraction)
    selected = sorted(range(len(units)), key=lambda i: (-importance[i], i))[:k]
    return [units[i] for i in sorted(selected)]

units = ["首先", "速度=3", "然后", "时间=4", "因此", "位移=12"]
assert retain_units(units, [0, 1, 0, 1, 0, 1], 0.5) == ["速度=3", "时间=4", "位移=12"]


import math
from statistics import mean, pstdev

def length_rewards(lengths, correct, alpha=0.2, eps=1e-8):
    if len(lengths) != len(correct) or not 0 <= alpha < 1 or not eps > 0:
        raise ValueError("invalid arguments")
    if any(not math.isfinite(n) or n < 0 for n in lengths):
        raise ValueError("lengths must be finite and nonnegative")
    if any(c not in (0, 1) for c in correct):
        raise ValueError("correctness must be binary")
    good = [n for n, c in zip(lengths, correct) if c]
    if not good:
        return [0.0] * len(lengths)
    center, scale = mean(good), pstdev(good) + eps
    rewards = []
    for n, c in zip(lengths, correct):
        z = (n - center) / scale
        sigmoid = 1 / (1 + math.exp(-z)) if z >= 0 else math.exp(z) / (1 + math.exp(z))
        rewards.append(c * (1 - alpha * sigmoid))
    return rewards

r = length_rewards([100, 300, 1], [1, 1, 0])
assert r[0] > r[1] > r[2] == 0
assert length_rewards([1, 100], [0, 0]) == [0.0, 0.0]
assert length_rewards([100, 100], [1, 1]) == [0.9, 0.9]


print("retained units:", retain_units(units, [0, 1, 0, 1, 0, 1], 0.5))
print("correctness-gated relative-length rewards:", [round(x, 6) for x in r])
print("Synthetic arithmetic checks passed; no model performance was measured.")
