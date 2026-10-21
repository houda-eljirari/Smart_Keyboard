"""Lab 02 - evaluation.   Run:  python -m src.evaluate"""
import json, os, random

from src.pipeline import (NGramModel, InterpolatedModel, load_documents, split_by_document,
                          top_k_accuracy, backoff_top_k, EOS)

MIN_COUNT = 2
KS = [0.001, 0.01, 0.03, 0.1, 0.3, 1.0]
LAMBDAS = [(0.6, 0.3, 0.1), (0.5, 0.3, 0.2), (0.4, 0.4, 0.2), (0.7, 0.2, 0.1)]


def backoff_accuracy(tri, bi, uni, sentences, k=3):
    hits = total = 0
    for s in sentences:
        for i, w in enumerate(s + [EOS]):
            if w not in tri.vocab:
                continue
            guess = [x for x, _ in backoff_top_k(tri, bi, uni, s[:i], k)[0]]
            total += 1
            hits += w in guess
    return hits / total


def main():
    docs = load_documents()
    train, test = split_by_document(docs)
    ids = sorted(docs)
    dev_docs = {i: docs[i] for i in ids[:int(len(ids) * 0.8)]}
    fit, val = split_by_document(dev_docs)          # validation = last 20 % of the train documents
    n_tr = sum(len(s) for s in train); n_te = sum(len(s) for s in test)
    print(f"Split BY DOCUMENT: {len(ids)} docs -> train = {len(train)} sentences / {n_tr} tokens, "
          f"test = {len(test)} sentences / {n_te} tokens")

    # ---- hyper-parameters chosen on the VALIDATION split only
    sweep, best_k = {}, {}
    for n in (1, 2, 3):
        sweep[n] = [NGramModel(fit, n, k, MIN_COUNT).perplexity(val)[0] for k in KS]
        best_k[n] = KS[sweep[n].index(min(sweep[n]))]
    best_i = (float("inf"), None, None)
    for k in (0.01, 0.03, 0.1):
        m = InterpolatedModel(fit, k=k, min_count=MIN_COUNT)
        for lam in LAMBDAS:
            m.lambdas = lam
            ppl = m.perplexity(val)[0]
            if ppl < best_i[0]:
                best_i = (ppl, k, lam)
    print("\nBest k on validation:", {f"{n}-gram": best_k[n] for n in best_k})
    print(f"Best interpolation on validation: k={best_i[1]}, lambdas={best_i[2]}")
    print("Perplexity vs k (validation):")
    for n in (1, 2, 3):
        print(f"  {n}-gram: " + ", ".join(f"k={k}: {p:.1f}" for k, p in zip(KS, sweep[n])))

    # ---- final models: train on the full train set, report on TEST once
    uni, bi, tri = (NGramModel(train, n, best_k[n], MIN_COUNT) for n in (1, 2, 3))
    tri_k1, tri_k0 = NGramModel(train, 3, 1.0, MIN_COUNT), NGramModel(train, 3, 0.0, MIN_COUNT)
    interp = InterpolatedModel(train, k=best_i[1], lambdas=best_i[2], min_count=MIN_COUNT)
    unk_rate = sum(w not in tri.vocab for s in test for w in s) / n_te
    print(f"\nVocabulary V = {tri.V} (min_count={MIN_COUNT}); <unk> share of test words = {unk_rate:.1%}")
    print(f"(top-3 accuracy ignores positions whose true word is <unk>)")

    rows = []
    def add(name, model):
        ppl, zero = model.perplexity(test)
        acc, _ = top_k_accuracy(model, test, 3)
        rows.append({"model": name, "perplexity": ppl, "zero_prob_share": zero, "top3": acc})

    add(f"unigram (k={best_k[1]})", uni)
    add(f"bigram (k={best_k[2]})", bi)
    add(f"trigram (k={best_k[3]})", tri)
    add("trigram (k=1)", tri_k1)
    add("trigram unsmoothed (k=0)", tri_k0)
    add(f"interpolated (k={best_i[1]})", interp)

    print("\nmodel                               perplexity   zero-prob tokens   top-3 accuracy")
    for r in rows:
        ppl = "inf" if r["perplexity"] == float("inf") else f"{r['perplexity']:.2f}"
        print(f"{r['model']:<36}{ppl:>10}{r['zero_prob_share']:>16.1%}{r['top3']:>17.1%}")
    bo = backoff_accuracy(tri, bi, uni, test)
    print(f"\nHard back-off (3->2->1-gram) top-3 accuracy: {bo:.1%}")

    # ---- failure analysis
    _, misses = top_k_accuracy(tri, test, 3)
    rng = random.Random(0)
    unseen = [" ".join(s[max(0, i - 2):i]) or "<start>"
              for s in test for i in range(len(s) + 1) if not tri.seen_context(s[:i])]
    lines = ["Trigram top-3 misses (context -> true word | suggestions):"]
    lines += [f"  '{c}' -> {w!r} | {g}" for c, w, g in rng.sample(misses, min(8, len(misses)))]
    lines += [f"\nTest positions whose trigram context never appears in training: {len(unseen)}",
              f"  examples: {rng.sample(sorted(set(unseen)), min(8, len(set(unseen))))}"]
    print("\n" + "\n".join(lines))

    os.makedirs("results", exist_ok=True)
    with open("results/failures.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    with open("results/metrics.json", "w") as f:
        json.dump({"rows": rows, "backoff_top3": bo, "ks": KS,
                   "val_sweep": {str(n): sweep[n] for n in sweep},
                   "best_k": {str(n): best_k[n] for n in best_k},
                   "interp_k": best_i[1], "interp_lambdas": list(best_i[2]),
                   "train_tokens": n_tr, "test_tokens": n_te, "vocab": tri.V,
                   "unk_rate_test": unk_rate, "unseen_trigram_contexts": len(unseen)}, f, indent=2)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    for n, c in ((1, "#bbbbbb"), (2, "#8a8a8a"), (3, "#333333")):
        axes[0].plot(KS, sweep[n], marker="o", color=c, label=f"{n}-gram")
    axes[0].set_xscale("log"); axes[0].set_yscale("log")
    axes[0].set_xlabel("smoothing constant k (log scale)")
    axes[0].set_ylabel("validation perplexity"); axes[0].legend()
    axes[0].set_title("Smoothing: too little or too much hurts")
    ok = [r for r in rows if r["perplexity"] != float("inf")]
    axes[1].barh([r["model"].replace(" (", "\n(") for r in ok], [r["top3"] for r in ok], color="#333333")
    axes[1].invert_yaxis(); axes[1].set_xlim(0, 1); axes[1].set_xlabel("top-3 accuracy (test)")
    axes[1].tick_params(axis="y", labelsize=7); axes[1].set_title("Next-word suggestion quality")
    fig.tight_layout(); fig.savefig("results/ngram_results.png", dpi=160)
    print("\nSaved results/metrics.json, results/failures.txt, results/ngram_results.png")


if __name__ == "__main__":
    main()