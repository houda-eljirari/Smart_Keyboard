"""Lab 02 - Smart Keyboard V2: n-gram language model with add-k smoothing."""
import csv, heapq, math, re
from collections import Counter, defaultdict

BOS, EOS, UNK = "<s>", "</s>", "<unk>"
TOKEN_RE = re.compile(r"[a-z0-9]+(?:'[a-z]+)?")


def tokenize(text):
    return TOKEN_RE.findall(text.lower())


def load_documents(path="data/corpus.csv"):
    docs = defaultdict(list)
    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            docs[int(row["doc_id"])].append(row["sentence"].split())
    return dict(docs)


def split_by_document(docs, test_fraction=0.2):
    """Fixed split BY DOCUMENT (the last 20 % of documents are the test set)."""
    ids = sorted(docs)
    cut = int(len(ids) * (1 - test_fraction))
    train = [s for i in ids[:cut] for s in docs[i]]
    test = [s for i in ids[cut:] for s in docs[i]]
    return train, test


class NGramModel:
    def __init__(self, sentences, n=3, k=1.0, min_count=1):
        self.n, self.k, self.min_count = n, k, min_count
        freq = Counter(w for s in sentences for w in s)
        self.vocab = {w for w, c in freq.items() if c >= min_count} | {EOS, UNK}
        self.V = len(self.vocab)
        self._sorted_vocab = sorted(self.vocab - {UNK})
        self.ctx_counts = defaultdict(Counter)
        for s in sentences:
            padded = [BOS] * (n - 1) + self.clean(s) + [EOS]
            for i in range(n - 1, len(padded)):
                self.ctx_counts[tuple(padded[i - n + 1:i])][padded[i]] += 1
        self.ctx_totals = {c: sum(cnt.values()) for c, cnt in self.ctx_counts.items()}
        self._cache = {}

    def clean(self, words):
        return [w if w in self.vocab else UNK for w in words]

    def context(self, words):
        words = [BOS] * (self.n - 1) + self.clean(words)
        return tuple(words[-(self.n - 1):]) if self.n > 1 else ()

    def prob(self, ctx, word):
        counts = self.ctx_counts.get(ctx)
        total = self.ctx_totals.get(ctx, 0)
        num = (counts[word] if counts else 0) + self.k
        den = total + self.k * self.V
        return num / den if den > 0 else 0.0

    def _top_for_ctx(self, ctx, k):
        key = (ctx, k)
        if key not in self._cache:
            counts = self.ctx_counts.get(ctx, {})
            best = heapq.nsmallest(k, ((w, c) for w, c in counts.items() if w != UNK),
                                   key=lambda x: (-x[1], x[0]))
            chosen = [w for w, _ in best]
            if len(chosen) < k:                      # unseen context: alphabetical filler
                for w in self._sorted_vocab:
                    if w not in chosen:
                        chosen.append(w)
                        if len(chosen) == k:
                            break
            self._cache[key] = [(w, self.prob(ctx, w)) for w in chosen]
        return self._cache[key]

    def top_k(self, words, k=3):
        return self._top_for_ctx(self.context(words), k)

    def seen_context(self, words):
        return self.context(words) in self.ctx_counts

    def perplexity(self, sentences):
        """Returns (perplexity, share_of_zero_probability_tokens); EOS included."""
        log_sum, n_tok, zeros = 0.0, 0, 0
        for s in sentences:
            padded = [BOS] * (self.n - 1) + self.clean(s) + [EOS]
            for i in range(self.n - 1, len(padded)):
                p = self.prob(tuple(padded[i - self.n + 1:i]), padded[i])
                n_tok += 1
                if p <= 0:
                    zeros += 1
                else:
                    log_sum += math.log(p)
        if zeros:
            return float("inf"), zeros / n_tok
        return math.exp(-log_sum / n_tok), 0.0


class InterpolatedModel:
    """p = l3 * p_trigram + l2 * p_bigram + l1 * p_unigram"""

    def __init__(self, sentences, k=0.1, lambdas=(0.6, 0.3, 0.1), min_count=1):
        self.models = [NGramModel(sentences, n, k, min_count) for n in (3, 2, 1)]
        self.lambdas = lambdas
        self.vocab, self.V, self.n = self.models[0].vocab, self.models[0].V, 3
        uni = self.models[2].ctx_counts[()]
        self._uni_ranked = sorted(self.vocab - {UNK}, key=lambda w: (-uni[w], w))
        self._cache = {}

    def clean(self, words):
        return self.models[0].clean(words)

    def _prob(self, c3, c2, word):
        l3, l2, l1 = self.lambdas
        tri, bi, uni = self.models
        return l3 * tri.prob(c3, word) + l2 * bi.prob(c2, word) + l1 * uni.prob((), word)

    def prob_words(self, history, word):
        history = [BOS, BOS] + history
        return self._prob(tuple(history[-2:]), (history[-1],), word)

    def top_k(self, words, k=3):
        hist = [BOS, BOS] + self.clean(words)
        c3, c2 = tuple(hist[-2:]), (hist[-1],)
        key = (c3, k, self.lambdas)
        if key not in self._cache:
            tri, bi, _ = self.models
            cands = set(tri.ctx_counts.get(c3, {})) | set(bi.ctx_counts.get(c2, {}))
            extra = 0          # words unseen in both contexts: only the unigram term differs
            for w in self._uni_ranked:
                if w not in cands:
                    cands.add(w)
                    extra += 1
                    if extra >= k:
                        break
            cands.discard(UNK)
            dist = {w: self._prob(c3, c2, w) for w in cands}
            self._cache[key] = sorted(dist.items(), key=lambda x: (-x[1], x[0]))[:k]
        return self._cache[key]

    def perplexity(self, sentences):
        log_sum, n_tok = 0.0, 0
        for s in sentences:
            words = self.clean(s)
            for i, w in enumerate(words + [EOS]):
                log_sum += math.log(self.prob_words(words[:i], w))
                n_tok += 1
        return math.exp(-log_sum / n_tok), 0.0


def backoff_top_k(tri, bi, uni, words, k=3):
    """Hard back-off: use the highest order whose context was actually seen."""
    for model in (tri, bi, uni):
        if model.seen_context(words) or model.n == 1:
            return model.top_k(words, k), f"{model.n}-gram"
    return [], "none"


def top_k_accuracy(model, sentences, k=3):
    """Top-k accuracy; positions whose true word is <unk> are skipped."""
    hits = total = 0
    failures = []
    for s in sentences:
        for i, true_w in enumerate(s + [EOS]):
            if true_w not in model.vocab:
                continue
            guess = [w for w, _ in model.top_k(s[:i], k)]
            total += 1
            if true_w in guess:
                hits += 1
            elif len(failures) < 2000:
                failures.append((" ".join(s[:i]) or "<start>", true_w, guess))
    return hits / total, failures