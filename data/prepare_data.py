"""Build data/corpus.csv from Kaggle "A Million News Headlines".
Put abcnews-date-text.csv in data/raw/ then run:  python data/prepare_data.py
One DOCUMENT = one publication day; doc ids are numbered in random order, so the
"last 20 % of documents" is a random set of days (no chronological drift)."""
import csv, os, random, re
from collections import defaultdict

RAW, OUT = "data/raw/abcnews-date-text.csv", "data/corpus.csv"
SEED, N_DOCS, PER_DAY = 42, 400, 100
MIN_WORDS, MAX_WORDS = 3, 15
TOKEN_RE = re.compile(r"[a-z0-9]+(?:'[a-z]+)?")      # same rule as src.pipeline.tokenize


def main():
    if not os.path.exists(RAW):
        raise SystemExit(f"Missing {RAW}: download it from Kaggle first.")
    by_day, seen = defaultdict(list), set()
    with open(RAW, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            toks = TOKEN_RE.findall(row["headline_text"].lower())
            text = " ".join(toks)
            if MIN_WORDS <= len(toks) <= MAX_WORDS and text not in seen:  # drop exact duplicates
                seen.add(text)
                by_day[row["publish_date"]].append(text)
    days = sorted(d for d, hs in by_day.items() if len(hs) >= PER_DAY)
    rng = random.Random(SEED)
    rng.shuffle(days)
    days = days[:N_DOCS]
    with open(OUT, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["doc_id", "sentence"])
        for doc_id, day in enumerate(days):
            for h in rng.sample(by_day[day], PER_DAY):
                w.writerow([doc_id, h])
    print(f"wrote {OUT}: {len(days) * PER_DAY} headlines in {len(days)} documents")


if __name__ == "__main__":
    main()