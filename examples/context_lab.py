#!/usr/bin/env python3
"""Synthetic context accounting; no model, provider tokenizer, API, or GPU.

Run: python3 examples/context_lab.py
A whitespace-separated unit is a TEACHING UNIT, not a real LLM token.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass


def units(text: str) -> int:
    return len(text.split())


def pad(text: str, length: int) -> str:
    if units(text) > length:
        raise ValueError("initial text exceeds requested synthetic length")
    return " ".join([text] + ["filler"] * (length - units(text)))


@dataclass(frozen=True)
class Turn:
    action: str
    observation: str


def experiment(strategy: str, steps: int = 8, window: int = 2) -> dict:
    """Replay an identical fixed trajectory; only the context view changes.

    Summary is a handwritten deterministic rule, NOT a learned summarizer.
    It incurs a synthetic compression call reading the old history, and
    preserves one planted fact. This does not estimate real task accuracy.
    """
    if steps < 2 or window < 0:
        raise ValueError("steps >= 2 and window >= 0 are required")
    base = pad("instruction question", 100)
    history: list[Turn] = []
    memory = ""
    input_units = output_units = compressor_units = 0
    final_context = ""
    for index in range(steps):
        if strategy == "summary" and index and index % 4 == 0:
            old = " ".join([memory] + [t.action + " " + t.observation for t in history])
            memory = pad("warehouse_code=K7", 35)
            # Synthetic summarizer input = 30 instruction units + old context.
            compressor_units += 30 + units(old) + units(memory)
            history = []
        chunks = [base, memory]
        for position, turn in enumerate(history):
            observation = turn.observation
            if strategy == "mask" and position < len(history) - window:
                observation = "[observation omitted]"
            chunks.extend([turn.action, observation])
        final_context = " ".join(chunks)
        input_units += units(final_context)
        action = pad(f"action_{index}", 20)
        observation = pad("warehouse_code=K7" if index == 0 else f"observation_{index}", 200)
        output_units += units(action)
        history.append(Turn(action, observation))
    # Last tool observation has not been sent back to a model yet: no input fee.
    return {
        "strategy": strategy,
        "main_input_units": input_units,
        "main_output_units": output_units,
        "compressor_input_plus_output_units": compressor_units,
        "total_units": input_units + output_units + compressor_units,
        "planted_fact_in_final_context": "warehouse_code=K7" in final_context,
    }


def retrieval_demo() -> dict:
    documents = [
        "warehouse code K7 closes at 18:00",
        "garden flowers need water each morning",
        "warehouse returns require the invoice code R2",
        "the library closes at 20:00 on weekdays",
    ]
    query = "warehouse closes"
    query_words = set(query.split())
    ranked = sorted(enumerate(documents),
                    key=lambda pair: (-len(query_words & set(pair[1].split())), pair[0]))
    selected = ranked[0][1]
    return {
        "all_document_units": units(" ".join(documents)),
        "selected_document_units": units(selected),
        "selected": selected,
        "relevant_fact_retained": "18:00" in selected,
        "method": "handwritten lexical overlap; not dense RAG or LLMLingua-2",
    }


def cache_demo() -> dict:
    # Same 1,000-unit input twice; second request reuses a 900-unit prefix.
    input_units, cached_units, output_units = 2000, 900, 100
    uncached_units = input_units - cached_units
    # Arbitrary relative price weights; no real provider price is implied.
    price = uncached_units * 1.0 + cached_units * 0.1 + output_units * 3.0
    return {"input_units": input_units, "cached_units": cached_units,
            "uncached_units": uncached_units, "output_units": output_units,
            "total_units": input_units + output_units,
            "synthetic_weighted_cost": price}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--window", type=int, default=2)
    args = parser.parse_args()
    runs = [experiment(s, args.steps, args.window) for s in ("raw", "mask", "summary")]
    assert cache_demo()["cached_units"] + cache_demo()["uncached_units"] == 2000
    if args.steps == 8 and args.window == 2:
        assert [r["total_units"] for r in runs] == [7120, 4150, 4685]
        assert [r["planted_fact_in_final_context"] for r in runs] == [True, False, True]
    print(json.dumps({"label": "SYNTHETIC TEACHING UNITS; NOT LLM RESULTS",
                      "trajectory": runs, "retrieval": retrieval_demo(),
                      "cache": cache_demo()}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
