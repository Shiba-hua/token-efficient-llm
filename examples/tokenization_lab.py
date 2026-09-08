#!/usr/bin/env python3
"""Standard-library lessons: simplified BPE, NTP, and domain mixtures.

python3 examples/tokenization_lab.py --mode all
All corpora are synthetic. The bigram models are count estimates, not LLMs.
"""
import argparse
from collections import Counter, defaultdict
import math


def merge_pair(tokens, pair):
    result, i = [], 0
    while i < len(tokens):
        if i + 1 < len(tokens) and (tokens[i], tokens[i + 1]) == pair:
            result.append(tokens[i] + tokens[i + 1])
            i += 2
        else:
            result.append(tokens[i])
            i += 1
    return result


def train_bpe(texts, max_merges):
    """Character BPE with cross-space merges; not a production tokenizer."""
    sequences = [list(text) for text in texts]
    merges = []
    for _ in range(max_merges):
        counts = Counter(pair for seq in sequences for pair in zip(seq, seq[1:]))
        if not counts:
            break
        # Lexical tie-break makes results independent of hash randomization.
        pair = min(counts, key=lambda p: (-counts[p], p))
        if counts[pair] < 2:
            break
        merges.append(pair)
        sequences = [merge_pair(seq, pair) for seq in sequences]
    return merges


def encode(text, merges):
    tokens = list(text)
    for pair in merges:
        tokens = merge_pair(tokens, pair)
    assert "".join(tokens) == text
    return tokens


def bpe_lab():
    english = ["the model predicts the next token", "the model reads the next text"] * 4
    chinese = ["模型预测下一个词元", "模型读取下一个文本"] * 4
    heldout = ["the model predicts text", "模型预测文本", "def f(x): return x + 1"]
    tokenizers = {
        "char": [],
        "english_bpe": train_bpe(english, 24),
        "balanced_bpe": train_bpe(english + chinese, 24),
    }
    print("BPE: learned only on synthetic TRAIN text; evaluated on held-out strings")
    for text in heldout:
        print(f"text={text!r}, chars={len(text)}, utf8_bytes={len(text.encode('utf-8'))}")
        for name, merges in tokenizers.items():
            tokens = encode(text, merges)
            print(f"  {name:14s} tokens={len(tokens):2d} pieces={tokens}")
    print("Fewer pieces here measure representation length, not task accuracy or speed.")


class Bigram:
    def __init__(self, texts, vocab, weights=None, alpha=0.5):
        if alpha <= 0 or not vocab:
            raise ValueError("Need positive smoothing and a non-empty vocabulary")
        self.vocab = set(vocab)
        self.alpha = alpha
        self.counts = defaultdict(Counter)
        self.totals = Counter()
        if weights is None:
            weights = [1.0] * len(texts)
        if len(weights) != len(texts) or any(w < 0 for w in weights):
            raise ValueError("Weights must be nonnegative and match texts")
        for words, weight in zip(texts, weights):
            tokens = ["<bos>"] + words + ["<eos>"]
            for previous, target in zip(tokens, tokens[1:]):
                if target not in self.vocab:
                    raise ValueError("Training target outside fixed vocabulary")
                self.counts[previous][target] += weight
                self.totals[previous] += weight

    def probability(self, previous, target):
        if target not in self.vocab:
            raise ValueError("Unknown evaluation token; specify a vocabulary policy")
        return ((self.counts[previous][target] + self.alpha) /
                (self.totals[previous] + self.alpha * len(self.vocab)))

    def nll(self, texts):
        losses = []
        for words in texts:
            tokens = ["<bos>"] + words + ["<eos>"]
            losses.extend(-math.log(self.probability(p, t))
                          for p, t in zip(tokens, tokens[1:]))
        if not losses:
            raise ValueError("No evaluation targets")
        return sum(losses) / len(losses)


def synthetic_corpora():
    general = ["the model reads text", "the model predicts text", "the model reads words"] * 6
    domain = ["the model proves claims", "the model proves results", "the proof uses lemmas"] * 6
    # Different sequences, while every word is already in the training vocabulary.
    general_eval = ["the model predicts words"]
    domain_eval = ["the model proves lemmas", "the proof uses results"]
    convert = lambda corpus: [s.split() for s in corpus]
    result = tuple(map(convert, (general, domain, general_eval, domain_eval)))
    train_sequences = {tuple(words) for words in result[0] + result[1]}
    assert not any(tuple(words) in train_sequences for words in result[2] + result[3])
    return result


def ntp_lab():
    general, domain, general_eval, _ = synthetic_corpora()
    vocab = {w for words in general + domain for w in words} | {"<eos>"}
    model = Bigram(general, vocab)
    print("NTP: smoothed bigram probability conditioned on ONE previous token")
    for target in ["reads", "predicts", "proves"]:
        print(f"p({target!r} | 'model') = {model.probability('model', target):.6f}")
    total = sum(model.probability("model", t) for t in vocab)
    assert abs(total - 1.0) < 1e-12
    loss = model.nll(general_eval)
    print(f"heldout_nll_nats_per_token={loss:.6f}, perplexity={math.exp(loss):.6f}")
    print("MTP targets for prefix ['the', 'model']: k=1: 'reads'; k=2: 'text'")
    print("A second marginal head does not condition on the sampled first target.")


def adaptation_lab():
    general, domain, general_eval, domain_eval = synthetic_corpora()
    vocab = {w for words in general + domain for w in words} | {"<eos>"}
    print("DOMAIN MIXTURE: exact count refits; NOT gradient-based continued pretraining")
    print("domain_weight,general_nll,domain_nll")
    for domain_weight in [0.0, 0.25, 0.50, 0.75, 1.0]:
        weights = ([1.0 - domain_weight] * len(general) +
                   [domain_weight] * len(domain))
        model = Bigram(general + domain, vocab, weights=weights)
        print(f"{domain_weight:.2f},{model.nll(general_eval):.6f},{model.nll(domain_eval):.6f}")
    print("The mixture holds weighted training mass fixed and exposes a distribution trade-off.")
    print("It illustrates adaptation/retention, not the dynamics or magnitude of LLM forgetting.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["all", "bpe", "ntp", "adaptation"], default="all")
    args = parser.parse_args()
    print("SYNTHETIC CPU LESSON — no LLM training or deployment result\n")
    for name, fn in [("bpe", bpe_lab), ("ntp", ntp_lab), ("adaptation", adaptation_lab)]:
        if args.mode in ("all", name):
            fn()
            print()


if __name__ == "__main__":
    main()
