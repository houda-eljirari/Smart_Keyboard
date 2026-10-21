import math
import random
import re
from collections import Counter, defaultdict
import nltk
import numpy as np
import pandas as pd

# -----------------------------------------------------------------------------
# PARTIE 1 — N-GRAM : corpus français fourni dans le sujet
# -----------------------------------------------------------------------------
CORPUS = [
    "L'apprentissage automatique est un sous-domaine de l'intelligence artificielle qui permet aux ordinateurs d'apprendre à partir de données.",
    "Le Maroc est un pays situé en Afrique du Nord. Sa capitale est Rabat. Elle possède une riche histoire.",
    "Rabat est la capitale du Maroc et est connue pour sa culture, son histoire et sa gastronomie.",
    "L'informatique est la science du traitement automatique de l'information par des machines.",
]
TOKEN_RE_FR = re.compile(r"[^a-zàâçéèêëîïôûùüÿñ0-9\s]")
BOS, EOS = "<s>", "</s>"


def tokenize(text):
    text = text.lower()
    text = TOKEN_RE_FR.sub(" ", text)
    return [t for t in text.split() if t.strip()]


def split_sentences(corpus):
    return [s.strip() for doc in corpus for s in doc.split(".") if s.strip()]


def train_ngram(sentences, n=2):
    ngram_counts, context_counts, vocab = Counter(), Counter(), set()
    for sentence in sentences:
        words = tokenize(sentence)
        vocab.update(words)
        padded = [BOS] * (n - 1) + words + [EOS]
        for i in range(n - 1, len(padded)):
            ng = tuple(padded[i-n+1:i+1])
            ctx = tuple(padded[i-n+1:i])
            ngram_counts[ng] += 1
            context_counts[ctx] += 1
    return ngram_counts, context_counts, vocab


def make_ngram_helpers(n, ngram_counts, context_counts, vocab, alpha=1.0):
    # V inclut le symbole de fin de phrase, comme dans le sujet.
    prediction_vocab = sorted(vocab) + [EOS]
    V = len(vocab) + 1

    def prob(context, word):
        context = tuple(context)
        return (ngram_counts[context + (word,)] + alpha) / (context_counts[context] + alpha * V)

    def predict_next(context, topk=5):
        if isinstance(context, str):
            context = tokenize(context)
        context = ([BOS] * (n - 1) + context)[-(n - 1):] if n > 1 else []
        return sorted(((w, prob(context, w)) for w in prediction_vocab), key=lambda x: -x[1])[:topk]

    def generate(max_words=15, seed=None, greedy=False):
        rng = random.Random(seed)
        context = [BOS] * (n - 1)
        output = []
        for _ in range(max_words):
            ctx = tuple(context[-(n-1):]) if n > 1 else ()
            choices = [(w, prob(ctx, w)) for w in prediction_vocab]
            if greedy:
                word = max(choices, key=lambda x: x[1])[0]
            else:
                word = rng.choices([x[0] for x in choices], weights=[x[1] for x in choices], k=1)[0]
            if word == EOS:
                break
            output.append(word)
            if n > 1:
                context.append(word)
        return " ".join(output)

    def perplexity(sentence):
        tokens = [BOS] * (n - 1) + tokenize(sentence) + [EOS]
        log_p = 0.0
        count = 0
        for i in range(n - 1, len(tokens)):
            ctx, word = tuple(tokens[i-n+1:i]), tokens[i]
            log_p += math.log(prob(ctx, word))
            count += 1
        return math.exp(-log_p / max(count, 1))

    return V, prediction_vocab, prob, predict_next, generate, perplexity


def run_ngram():
    print("\\n" + "=" * 72 + "\nPARTIE 1 — N-GRAM\n" + "=" * 72)
    sentences = split_sentences(CORPUS)
    print(f"1. Nombre total de phrases : {len(sentences)}")
    print("Tokenisation de 'l'intelligence' :", tokenize("l'intelligence"))

    ngram_counts, context_counts, vocab = train_ngram(sentences, n=2)
    V, next_words, prob, predict_next, generate, perplexity = make_ngram_helpers(
        2, ngram_counts, context_counts, vocab, alpha=1.0)
    print(f"\\n2. Taille du vocabulaire V : {V} (à comparer au résultat attendu du sujet)")
    print("Bigrammes les plus fréquents :", ngram_counts.most_common(10))

    print("\\n3. Probabilités avec lissage de Laplace (alpha=1)")
    print("P(maroc | le) =", prob(("le",), "maroc"))
    print("P(est | le)   =", prob(("le",), "est"))
    c_le = context_counts[("le",)]
    print("Sans lissage P(maroc|le) =", ngram_counts[("le", "maroc")] / c_le if c_le else 0)
    print("Bigramme inédit avec lissage :", prob(("le",), "ordinateur"))
    print("Sans lissage, un bigramme inédit a une probabilité nulle.")

    contexts = [BOS] + sorted(vocab)
    matrix = pd.DataFrame([[prob((h,), w) for w in next_words] for h in contexts],
                          index=contexts, columns=next_words)
    print("\\n4. Matrice des transitions :", matrix.shape)
    print("Sommes des lignes (doivent être proches de 1) :", matrix.sum(axis=1).round(6).unique())
    print(matrix.loc[[BOS, "le", "rabat"], ["maroc", "est", EOS]])
    print("Prédiction après 'le' :", predict_next(["le"]))

    print("\\n5. Cinq mots les plus probables au début :", predict_next([], topk=5))
    print("\\n6. Générations aléatoires :")
    for seed in range(3):
        print(f"  {seed+1}. {generate(seed=seed)}")
    print("Décodage glouton :", generate(seed=0, greedy=True))
    print("Le choix glouton peut répéter les choix localement les plus probables et réduire la diversité.")

    print("\\n7. Perplexité :")
    examples = ["Rabat est la capitale du Maroc", "Maroc capitale la est Rabat"]
    for phrase in examples:
        print(f"  {phrase!r} -> {perplexity(phrase):.2f}")
    print("La phrase la plus proche des séquences apprises devrait généralement avoir la perplexité la plus faible.")

    print("\\n8. Influence du lissage et de n")
    for alpha in (1.0, 0.01):
        _, _, _, _, gen, pp = make_ngram_helpers(2, ngram_counts, context_counts, vocab, alpha=alpha)
        print(f"  alpha={alpha}: génération={gen(seed=7)!r}; perplexité={pp(examples[0]):.2f}")
    tri_counts, tri_contexts, tri_vocab = train_ngram(sentences, n=3)
    tri = make_ngram_helpers(3, tri_counts, tri_contexts, tri_vocab, alpha=1.0)
    print("  trigramme, génération :", tri[4](seed=7))
    print("Conclusion : un contexte plus long peut améliorer la cohérence locale, mais les contextes rares deviennent plus fréquents.")

# -----------------------------------------------------------------------------
# PARTIE 2 — HMM : corpus Treebank et universal tagset de NLTK
# -----------------------------------------------------------------------------
def run_hmm():
    print("\\n" + "=" * 72 + "\nPARTIE 2 — HMM\n" + "=" * 72)
    nltk.download("treebank", quiet=True)
    nltk.download("universal_tagset", quiet=True)
    data = list(nltk.corpus.treebank.tagged_sents(tagset="universal"))
    print("3. Première phrase étiquetée :", data[0])
    tagged_words = [pair for sent in data for pair in sent]
    print("4. Nombre initial de mots étiquetés :", len(tagged_words))
    tags_before = sorted({tag for _, tag in tagged_words})
    print("5. Tags uniques :", len(tags_before), tags_before)
    print("6. Taille du vocabulaire initial :", len({word for word, _ in tagged_words}))
    print("7. Exemples de traces X :", [w for w, t in tagged_words if t == "X"][:10])

    def is_trace(word, tag):
        return tag == "X" and (word.startswith("*") or word == "0")

    data = [[(w, t) for w, t in sent if not is_trace(w, t)] for sent in data]
    data = [sent for sent in data if sent]
    tagged_words = [pair for sent in data for pair in sent]
    print("   Nombre après nettoyage :", len(tagged_words))
    vocab = sorted({w for w, _ in tagged_words})
    tags = sorted({t for _, t in tagged_words})

    # 8. Probabilités d'émission P(mot | tag)
    tag_counts = Counter(t for _, t in tagged_words)
    pair_counts = Counter(tagged_words)
    words_by_tag = defaultdict(list)
    for (word, tag), count in pair_counts.items():
        words_by_tag[tag].append((word, count))

    def emission(word, tag):
        return pair_counts[(word, tag)] / tag_counts[tag] if tag_counts[tag] else 0.0

    print("\\n8. P(the|DET) =", emission("the", "DET"), "; P(the|NOUN) =", emission("the", "NOUN"))

    # 9. Probabilités de transition P(tag suivant | tag courant), dans chaque phrase.
    trans, depart = Counter(), Counter()
    for sent in data:
        ts = [t for _, t in sent]
        for t1, t2 in zip(ts, ts[1:]):
            trans[(t1, t2)] += 1
            depart[t1] += 1
    A = pd.DataFrame(
        [[trans[(t1, t2)] / depart[t1] if depart[t1] else 0.0 for t2 in tags] for t1 in tags],
        index=tags, columns=tags)
    print("\\n9. Matrice de transition A :")
    print(A.round(2))
    if "DET" in A.index:
        print("Tag suivant le plus fréquent après DET :", A.loc["DET"].idxmax())
    if "ADJ" in A.index:
        print("Tag suivant le plus fréquent après ADJ :", A.loc["ADJ"].idxmax())
    print("Sommes des lignes non nulles :", A.sum(axis=1).round(6).unique())

    # 10. Probabilités initiales par étiquette de la première position de phrase.
    starts = Counter(sent[0][1] for sent in data)
    total_sents = len(data)
    pi = {tag: starts[tag] / total_sents for tag in tags}
    print("\\n10. Probabilités initiales pi :", pi)

    # Échantillonnage d'émission pondéré par la fréquence du couple (mot, tag).
    # Construire une liste pondérée : un mot apparaît autant de fois que sa fréquence pour ce tag.
    emission_words = {tag: [w for (w, t), c in pair_counts.items() if t == tag for _ in range(c)] for tag in tags}
    def generate_hmm(max_words=20, seed=None):
        rng = random.Random(seed)
        current = rng.choices(tags, weights=[pi[t] for t in tags], k=1)[0]
        out_words, out_tags = [], []
        for _ in range(max_words):
            candidates = emission_words.get(current, [])
            if not candidates:
                break
            word = rng.choice(candidates)
            out_words.append(word)
            out_tags.append(current)
            if word == ".":
                break
            row = A.loc[current].to_dict()
            next_tags = list(row)
            weights = [max(0.0, row[t]) for t in next_tags]
            if sum(weights) <= 0:
                break
            current = rng.choices(next_tags, weights=weights, k=1)[0]
        return " ".join(out_words), out_tags

    print("\\n11. Texte HMM généré :")
    text, generated_tags = generate_hmm(seed=42)
    print("Texte :", text)
    print("Tags  :", generated_tags)
    print("\\n12. Conclusion : le HMM modélise les transitions entre catégories grammaticales et les émissions mot/tag.")
    print("La syntaxe peut être partiellement plausible, mais le sens reste souvent incohérent car le modèle utilise une hypothèse de Markov et des émissions conditionnelles au tag, pas le sens global de la phrase.")


if __name__ == "__main__":
    run_ngram()
    run_hmm()
