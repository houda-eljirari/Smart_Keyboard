# Smart Keyboard : clavier intelligent basé sur les modèles N-gram

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://smartkeyboard-5adqetjt2ljeppjkunaugz.streamlit.app/)
![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-app-red)
![NLP](https://img.shields.io/badge/NLP-N--gram-green)

Application web de **prédiction du mot suivant** (et de génération de texte) construite à partir d'un modèle de langage **N-gramme avec lissage de Laplace**. Projet réalisé dans le cadre de l'**Atelier 2 : « Utiliser les ML N-gram et HMM pour générer un texte »** du module NLP (Semestre 3, ENSET Mohammedia, A.U 2026/2027).

**Démo en ligne : [smartkeyboard.streamlit.app](https://smartkeyboard-5adqetjt2ljeppjkunaugz.streamlit.app/)**

---

## Fonctionnalités

- **Suggestion du mot suivant** : à chaque mot saisi, l'application propose les mots les plus probables avec leur probabilité.
- **Génération automatique de texte** à partir du modèle entraîné (tirage aléatoire pondéré par les probabilités).
- **Paramètres réglables** : ordre du modèle `n` (bigramme, trigramme) et paramètre de lissage `alpha`.
- **Interface web simple** développée avec Streamlit, accessible sans installation.

---

## Installation et exécution en local

### Prérequis
- Python 3.10 ou plus
- `pip`

### Étapes

```bash
# 1. Cloner le dépôt
git clone https://github.com/houda-eljirari/Smart_Keyboard.git
cd Smart_Keyboard

# 2. Créer et activer un environnement virtuel
python -m venv .venv
# Windows (PowerShell)
.venv\Scripts\Activate.ps1
# Linux / macOS
source .venv/bin/activate

# 3. Installer les dépendances
pip install -r requirements.txt

# 4. Lancer l'application
streamlit run app.py
```

L'application s'ouvre sur `http://localhost:8501`.

---

## Dépendances principales

| Bibliothèque | Utilisation |
|---|---|
| `streamlit` | Interface web |
| `pandas` | Matrices de probabilités |
| `nltk` | Corpus Treebank (partie HMM) |
| `numpy` | Calculs numériques |

Les modules `re`, `math`, `random` et `collections` font partie de la bibliothèque standard de Python.

---

## Limites

- Le corpus d'entraînement est très petit : les prédictions restent peu fiables.
- Un lissage trop fort (alpha = 1) rend la génération presque aléatoire sur un petit corpus.
- Le problème des n-grammes inconnus s'aggrave quand `n` augmente (trigramme).
- Pistes d'amélioration : corpus plus grand, *backoff* / interpolation / Kneser-Ney, modèles neuronaux.
