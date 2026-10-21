"""Smart Keyboard - Streamlit UI.  Run:  streamlit run app.py"""
import json

import streamlit as st

from src.pipeline import (NGramModel, InterpolatedModel, EOS, load_documents,
                          backoff_top_k, tokenize)

MIN_COUNT = 2
st.set_page_config(page_title="Smart Keyboard", page_icon="⌨️", layout="centered")

st.markdown("""
<style>
#MainMenu, footer {visibility: hidden;}
.block-container {max-width: 780px; padding-top: 2rem;}

.hero {background: linear-gradient(135deg, #6366f1 0%, #a855f7 50%, #ec4899 100%);
       border-radius: 24px; padding: 26px 30px; color: white; margin-bottom: 1.2rem;
       box-shadow: 0 12px 32px rgba(99, 102, 241, .35);}
.hero h1 {margin: 0; padding: 0; font-size: 2.3rem; color: white;}
.hero p {margin: .4rem 0 0; opacity: .92; font-size: .98rem;}

.stTextInput input {font-size: 1.35rem; padding: 16px 18px; border-radius: 18px;}
.stTextInput > div > div {border-radius: 18px;}

div.stButton > button {border-radius: 999px; padding: .55rem 1rem; font-weight: 600;
       border: 1px solid rgba(127,127,127,.35); transition: all .15s ease; width: 100%;}
div.stButton > button:hover {transform: translateY(-2px); border-color: #a855f7;
       box-shadow: 0 6px 16px rgba(168, 85, 247, .25);}
button[kind="primary"], button[data-testid="stBaseButton-primary"] {
       background: linear-gradient(135deg, #6366f1, #ec4899) !important;
       color: white !important; border: none !important;}

.badge {display: inline-block; padding: 4px 12px; border-radius: 999px; font-size: .8rem;
        font-weight: 600; margin-right: 6px;}
.badge-model {background: rgba(99,102,241,.18); color: #818cf8;}
.badge-warn {background: rgba(245,158,11,.18); color: #f59e0b;}

.bar-row {display: flex; align-items: center; margin: 7px 0; font-size: .95rem;}
.bar-label {width: 150px; font-weight: 600; overflow: hidden; text-overflow: ellipsis;}
.bar-track {flex: 1; height: 12px; border-radius: 99px; background: rgba(127,127,127,.18);}
.bar-fill {height: 100%; border-radius: 99px;
           background: linear-gradient(90deg, #6366f1, #ec4899);}
.bar-pct {width: 62px; text-align: right; opacity: .75; font-variant-numeric: tabular-nums;}
.section-title {font-size: .8rem; letter-spacing: .08em; text-transform: uppercase;
                opacity: .6; margin: 1.4rem 0 .4rem; font-weight: 700;}
</style>
""", unsafe_allow_html=True)


def load_params():
    try:
        with open("results/metrics.json") as f:
            m = json.load(f)
        return m["best_k"], m["interp_k"], tuple(m["interp_lambdas"])
    except Exception:
        return {"1": 0.1, "2": 0.03, "3": 0.01}, 0.05, (0.6, 0.3, 0.1)


@st.cache_resource
def get_models():
    best_k, ik, lam = load_params()
    sentences = [s for d in load_documents().values() for s in d]
    uni, bi, tri = [NGramModel(sentences, n, best_k[str(n)], MIN_COUNT) for n in (1, 2, 3)]
    interp = InterpolatedModel(sentences, k=ik, lambdas=lam, min_count=MIN_COUNT)
    return tri, bi, uni, interp, len(sentences), best_k


tri, bi, uni, interp, n_sent, best_k = get_models()

# ------------------------------------------------------------- state & callbacks
if "text" not in st.session_state:
    st.session_state["text"] = "the prime minister"


def add_word(w):
    st.session_state["text"] = (st.session_state["text"].rstrip() + " " + w).strip()


def delete_word():
    st.session_state["text"] = " ".join(st.session_state["text"].split()[:-1])


def set_text(t):
    st.session_state["text"] = t


# ------------------------------------------------------------- header
st.markdown("""
<div class="hero">
  <h1>⌨️ Smart Keyboard</h1>
  <p>Next-word suggestions from n-gram language models trained on ABC News headlines.</p>
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.header("About")
    st.write("Models: unigram, bigram and trigram with add-k smoothing, "
             "hard back-off and linear interpolation.")
    st.write(f"Training data: **{n_sent:,}** headlines, vocabulary **{tri.V:,}** words "
             f"(words seen fewer than {MIN_COUNT} times become `<unk>`).")
    st.write(f"Smoothing k (tuned on validation): " + ", ".join(
        f"{n}-gram = {best_k[str(n)]}" for n in (1, 2, 3)))
    st.caption("Dataset: A Million News Headlines (Kaggle, R. Kulkarni).")

mode = st.radio("Model", ["Back-off (3 → 2 → 1-gram)", "Interpolated"], horizontal=True)

text = st.text_input("Type the beginning of a headline", key="text",
                     label_visibility="collapsed", placeholder="Start typing a headline...")
words = tokenize(text)

c1, c2, = st.columns([1, 1])
c1.button("⌫ Delete word", on_click=delete_word)
c2.button("✕ Clear", on_click=set_text, args=("",))


# ------------------------------------------------------------- predictions
if mode.startswith("Back-off"):
    suggestions, order = backoff_top_k(tri, bi, uni, words, 8)
    model_name = f"{order} used"
else:
    suggestions, model_name = interp.top_k(words, 8), "interpolated"

unknown = [w for w in words if w not in uni.vocab]
badges = f'<span class="badge badge-model">{model_name}</span>'
if unknown:
    badges += f'<span class="badge badge-warn">unknown → &lt;unk&gt;: {", ".join(unknown)}</span>'
st.markdown(f'<div class="section-title">Suggestions</div>{badges}', unsafe_allow_html=True)

cols = st.columns(3)
for i, (col, (w, p)) in enumerate(zip(cols, suggestions[:3])):
    if w == EOS:
        col.button(f"⏎ end · {p:.1%}", disabled=True, key=f"sugg_{i}")
    else:
        col.button(f"{w} · {p:.1%}", on_click=add_word, args=(w,), key=f"sugg_{i}",
                   type="primary" if i == 0 else "secondary")

st.markdown('<div class="section-title">Top 8 probabilities</div>', unsafe_allow_html=True)
top = max(p for _, p in suggestions) or 1.0
rows = ""
for w, p in suggestions:
    label = "⏎ end of headline" if w == EOS else w
    rows += (f'<div class="bar-row"><div class="bar-label">{label}</div>'
             f'<div class="bar-track"><div class="bar-fill" style="width:{p / top * 100:.0f}%">'
             f'</div></div><div class="bar-pct">{p:.1%}</div></div>')
st.markdown(rows, unsafe_allow_html=True)

st.caption("Click a suggestion to add it to your headline. Probabilities are model estimates "
           "on news headlines, not real-world typing statistics.")