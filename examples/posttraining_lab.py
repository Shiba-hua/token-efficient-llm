#!/usr/bin/env python3
"""Synthetic post-training arithmetic lab; standard library, CPU, no model/API.

Run: python3 examples/posttraining_lab.py --self-test --plots
Figures illustrate equations, NOT measured LLM performance.
"""
import argparse
import html
import json
import math
from pathlib import Path


def softmax(logits):
    top = max(logits)
    values = [math.exp(x - top) for x in logits]
    total = sum(values)
    return [x / total for x in values]


def kl(p, q):
    """Discrete KL(p||q), strictly positive normalized probabilities."""
    if len(p) != len(q) or any(x <= 0 for x in p + q):
        raise ValueError("KL requires equally sized, positive distributions")
    if not all(math.isclose(sum(x), 1.0) for x in (p, q)):
        raise ValueError("probabilities must sum to one")
    return sum(a * math.log(a / b) for a, b in zip(p, q))


def forward_kl_grad(logits, teacher):
    """d KL(teacher||student) / d student logits, fixed one-step state."""
    return [p - q for p, q in zip(softmax(logits), teacher)]


def reverse_kl_grad(logits, teacher):
    """Exact gradient at one fixed state, not a multi-step OPD trainer."""
    student = softmax(logits)
    divergence = kl(student, teacher)
    return [p * (math.log(p / q) - divergence)
            for p, q in zip(student, teacher)]


def dpo_loss_grad(margin, beta=0.5):
    """margin = log(pi_w/ref_w) - log(pi_l/ref_l).

    Return loss and derivative w.r.t. this scalar margin.
    """
    if beta <= 0:
        raise ValueError("beta must be positive")
    x = beta * margin
    loss = max(-x, 0.0) + math.log1p(math.exp(-abs(x)))
    sigmoid_minus_x = (math.exp(-x) / (1 + math.exp(-x))
                       if x >= 0 else 1 / (1 + math.exp(x)))
    return loss, -beta * sigmoid_minus_x


def group_advantages(rewards, eps_std=1e-8):
    if not rewards or eps_std <= 0:
        raise ValueError("nonempty group and positive eps_std required")
    if max(rewards) == min(rewards):
        return [0.0] * len(rewards)
    mean = sum(rewards) / len(rewards)
    std = math.sqrt(sum((r - mean) ** 2 for r in rewards) / len(rewards))
    return [(r - mean) / (std + eps_std) for r in rewards]


def ppo_surrogate(ratio, advantage, eps_clip=0.2):
    """Clipped objective to MAXIMIZE, not a hard constraint on ratio."""
    if ratio <= 0 or not 0 < eps_clip < 1:
        raise ValueError("positive ratio and 0 < eps_clip < 1 required")
    clipped = min(max(ratio, 1 - eps_clip), 1 + eps_clip)
    return min(ratio * advantage, clipped * advantage)


def gated_reward(correct, cost, budget=100.0, weight=0.2):
    """Toy gate: every success scores above every failure if 0 <= weight < 1."""
    if cost < 0 or budget <= 0 or not 0 <= weight < 1:
        raise ValueError("invalid cost, budget, or weight")
    return float(correct) * (1 - weight * min(cost / budget, 1.0))


def sft_demo():
    # Every target token has probability exp(-1), so every token NLL is 1.
    lengths = [2, 8]
    nlls = [2.0, 8.0]
    logits, teacher = [0.4, -0.2, 0.0], [0.7, 0.2, 0.1]
    student = softmax(logits)
    return {"synthetic": True, "target_lengths": lengths,
            "token_average_nll": sum(nlls) / sum(lengths),
            "sequence_average_nll": sum(n / t for n, t in zip(nlls, lengths)) / 2,
            "long_example_token_weight": 0.8,
            "long_example_sequence_weight": 0.5,
            "student": student, "teacher": teacher,
            "forward_kl_teacher_student": kl(teacher, student),
            "reverse_kl_student_teacher": kl(student, teacher),
            "forward_gradient": forward_kl_grad(logits, teacher),
            "reverse_gradient": reverse_kl_grad(logits, teacher)}


def dpo_demo():
    margin = 0.0
    history = []
    for step in range(6):
        loss, grad = dpo_loss_grad(margin)
        history.append({"step": step, "margin": margin, "loss": loss,
                        "gradient_wrt_margin": grad})
        margin -= 0.5 * grad
    return {"synthetic": True, "scalar_margin_updates": history,
            "clip_examples": [{"advantage": a, "ratio": r,
                               "surrogate": ppo_surrogate(r, a)}
                              for a in [-1, 1] for r in [0.5, 1, 1.5]]}


def grpo_demo():
    correctness, costs = [True, True, False, False], [30, 100, 1, 50]
    gated = [gated_reward(y, c) for y, c in zip(correctness, costs)]
    return {"synthetic": True,
            "binary_mixed": group_advantages([1, 1, 0, 0]),
            "binary_all_correct": group_advantages([1, 1, 1, 1]),
            "binary_all_wrong": group_advantages([0, 0, 0, 0]),
            "gated_rewards": gated, "gated_advantages": group_advantages(gated),
            "all_correct_different_cost_advantages":
                group_advantages([gated_reward(True, c) for c in [10, 40, 70, 100]]),
            "unsafe_ungated_reward": {"long_correct": 1 - 0.02 * 100,
                                      "short_wrong": 0 - 0.02 * 1}}


def self_test():
    z, q = [0.4, -0.2, 0.0], [0.7, 0.2, 0.1]
    h = 1e-6
    for direction, gradient in [("forward", forward_kl_grad), ("reverse", reverse_kl_grad)]:
        for j, analytical in enumerate(gradient(z, q)):
            lo, hi = z.copy(), z.copy()
            lo[j] -= h
            hi[j] += h
            fn = (lambda v: kl(q, softmax(v))) if direction == "forward" else (lambda v: kl(softmax(v), q))
            numerical = (fn(hi) - fn(lo)) / (2 * h)
            assert math.isclose(analytical, numerical, abs_tol=1e-7)
    for margin in [-3.0, 0.0, 3.0]:
        numerical = (dpo_loss_grad(margin + h)[0] - dpo_loss_grad(margin - h)[0]) / (2 * h)
        assert math.isclose(numerical, dpo_loss_grad(margin)[1], abs_tol=1e-7)
    assert group_advantages([1, 1, 1]) == [0, 0, 0]
    assert group_advantages([0, 0, 0]) == [0, 0, 0]
    assert group_advantages([0.1, 0.1, 0.1]) == [0, 0, 0]
    assert math.isclose(ppo_surrogate(1.5, 1), 1.2)
    assert math.isclose(ppo_surrogate(1.5, -1), -1.5)
    assert math.isclose(ppo_surrogate(0.5, -1), -0.8)
    assert gated_reward(True, 1e9) > gated_reward(False, 0)
    # The toy scalar DPO gradient descent should decrease its objective.
    hist = dpo_demo()["scalar_margin_updates"]
    assert hist[-1]["loss"] < hist[0]["loss"]
    return "PASS: finite-difference KL/DPO gradients, clipping signs, constant groups, correctness gate"


def svg_plot(path, title, subtitle, series, xmin, xmax, ymin, ymax, xlabel, ylabel, xticks=None):
    """Small deterministic line-chart renderer: no third-party dependencies."""
    width, height = 900, 540
    left, top, right, bottom = 85, 96, 865, 412
    esc = html.escape
    px = lambda x: left + (x - xmin) / (xmax - xmin) * (right - left)
    py = lambda y: bottom - (y - ymin) / (ymax - ymin) * (bottom - top)
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img">',
             f'<title>{esc(title)}</title><desc>{esc(subtitle)}</desc>',
             '<rect width="900" height="540" fill="#ffffff"/>',
             '<g font-family="Arial, sans-serif" fill="#172033">',
             f'<text x="35" y="35" font-size="24" font-weight="700">{esc(title)}</text>',
             f'<text x="35" y="65" font-size="15">{esc(subtitle)}</text>']
    for k in range(5):
        y = ymin + (ymax - ymin) * k / 4
        parts += [f'<path d="M {left} {py(y):.2f} H {right}" stroke="#e3e8ef"/>',
                  f'<text x="{left-12}" y="{py(y)+5:.2f}" text-anchor="end" font-size="14">{y:.2g}</text>']
    for x in (xticks if xticks is not None else [xmin + (xmax - xmin) * k / 4 for k in range(5)]):
        parts.append(f'<text x="{px(x):.2f}" y="438" text-anchor="middle" font-size="14">{x:g}</text>')
    parts += [f'<path d="M {left} {top} V {bottom} H {right}" fill="none" stroke="#49556b"/>',
              f'<text x="475" y="466" text-anchor="middle" font-size="16">{esc(xlabel)}</text>',
              f'<text transform="translate(22 254) rotate(-90)" text-anchor="middle" font-size="16">{esc(ylabel)}</text>']
    for idx, (label, color, points) in enumerate(series):
        coords = " ".join(f'{px(x):.2f},{py(y):.2f}' for x, y in points)
        parts.append(f'<polyline points="{coords}" fill="none" stroke="{color}" stroke-width="3"/>')
        if len(points) < 20:
            parts.extend(f'<circle cx="{px(x):.2f}" cy="{py(y):.2f}" r="5" fill="{color}"/>' for x, y in points)
        lx = 60 + idx * 280
        parts += [f'<path d="M {lx} 502 h 24" stroke="{color}" stroke-width="3"/>',
                  f'<text x="{lx+32}" y="507" font-size="14">{esc(label)}</text>']
    parts.append('</g></svg>')
    path.write_text("\n".join(parts), encoding="utf-8")


def make_plots():
    dest = Path(__file__).resolve().parents[1] / "assets" / "plots"
    dest.mkdir(parents=True, exist_ok=True)
    q = [0.8, 0.2]
    xs = [0.02 + i * 0.0096 for i in range(101)]
    svg_plot(dest / "posttraining-kl.svg", "Same teacher, two KL directions",
             "Synthetic 2-action distribution; teacher probability q(A)=0.8. No LLM experiment.",
             [("KL(teacher || student)", "#1864c6", [(x, kl(q, [x, 1-x])) for x in xs]),
              ("KL(student || teacher)", "#c64032", [(x, kl([x, 1-x], q)) for x in xs])],
             0, 1, 0, 3, "Student probability p(A)", "KL divergence (nats)", [0, 0.2, 0.4, 0.6, 0.8, 1])
    ratios = [0.4 + i * 0.006 for i in range(201)]
    svg_plot(dest / "posttraining-clip.svg", "PPO clipping depends on the advantage sign",
             "Synthetic objective to maximize; eps_clip=0.2. Ratios are not forcibly bounded.",
             [("A=+1, clipped objective", "#1864c6", [(r, ppo_surrogate(r, 1)) for r in ratios]),
              ("A=-1, clipped objective", "#c64032", [(r, ppo_surrogate(r, -1)) for r in ratios])],
             0.4, 1.6, -1.8, 1.8, "Probability ratio: current / rollout policy", "Surrogate objective", [0.4, 0.8, 1, 1.2, 1.6])
    adv = group_advantages([gated_reward(True, c) for c in [10, 40, 70, 100]])
    svg_plot(dest / "posttraining-grpo.svg", "Correctness and relative advantage are different",
             "All 4 answers are correct. Adding a bounded cost term changes their relative rewards.",
             [("Binary reward only", "#1864c6", list(zip([10, 40, 70, 100], [0]*4))),
              ("Correctness-gated cost reward", "#c64032", list(zip([10, 40, 70, 100], adv)))],
             0, 110, -1.7, 1.7, "Synthetic deployment token cost", "Group-relative advantage", [10, 40, 70, 100])
    return [str(dest / f"posttraining-{x}.svg") for x in ["kl", "clip", "grpo"]]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--section", choices=["all", "sft", "dpo", "grpo"], default="all")
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--plots", action="store_true")
    args = parser.parse_args()
    out = {"notice": "Synthetic arithmetic only; no LLM, optimizer framework, rollout, or performance claim."}
    if args.self_test:
        out["self_test"] = self_test()
    demos = {"sft": sft_demo, "dpo": dpo_demo, "grpo": grpo_demo}
    out.update({k: fn() for k, fn in demos.items() if args.section in ["all", k]})
    if args.plots:
        out["plots"] = make_plots()
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
