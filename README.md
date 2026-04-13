# Chatbot Expert AI Act (UE 2024/1689)

Chatbot RAG pour interroger le Reglement europeen sur l'Intelligence Artificielle en langage naturel. Routage deterministe (code Python) + LLM pour la redaction. Recherche internet automatique via DuckDuckGo si l'information n'est pas dans le AI Act.

Tout tourne en local sauf la recherche web.

## Stack technique

| Composant | Outil | Detail |
|---|---|---|
| Embeddings | `paraphrase-multilingual-mpnet-base-v2` | 278M parametres, 768 dimensions, 50+ langues |
| Vector store | FAISS | Index persistant sur disque, recherche avec seuil de score |
| LLM | Qwen 2.5 3B via Ollama | Inference locale, ~2 Go RAM |
| Recherche web | DuckDuckGo (ddgs) | Fallback quand l'info n'est pas dans le AI Act |
| Memoire | InMemoryChatMessageHistory | Memoire conditionnelle (activee sur les questions de suivi) |
| Interface | Streamlit | Chat web avec historique |

## Structure du projet

```
RAG_project/
├── requirements.txt                      # Dependances Python
├── chunker.py                            # Parsing structurel du AI Act (641 chunks)
├── build_index.py                        # Construction de l'index vectoriel FAISS
├── app.py                                # Application Streamlit (3 modes + memoire)
├── chatbot_ai_act.ipynb                  # Notebook tout-en-un
├── L-202401689FR.000101.fmx.xml.md      # Texte source du AI Act (FR)
└── .gitignore
```

## Prerequis

- Python 3.10+
- [Ollama](https://ollama.com/download) installe et lance
- ~2 Go de RAM libre (pour Qwen 2.5 3B)

## Installation

```bash
git clone <url-du-depot>
cd RAG_project

python -m venv .venv
source .venv/bin/activate        # Linux/Mac
# .venv\Scripts\activate         # Windows

pip install -r requirements.txt

ollama pull qwen2.5:3b
```

## Utilisation

```bash
# 1. Construire l'index (une seule fois)
python build_index.py

# 2. Lancer le chatbot
streamlit run app.py
```

Ou ouvrir `chatbot_ai_act.ipynb` dans Jupyter.

## 3 modes automatiques

Le routage est fait par du **code Python** (regex + score FAISS), pas par le LLM. Le LLM sert uniquement a rediger la reponse.

### Mode DIRECT (0 appel LLM, instantane)

Regex detecte un article ou considerant precis → restitution du texte officiel mot a mot via `docstore_lookup()` (bypass vectoriel).

```
"Donne-moi l'article 5"       → texte integral (8 paragraphes)
"Que dit le considerant 12 ?" → texte exact
```

### Mode RAG (1 appel LLM)

FAISS recherche les passages pertinents (seuil de score 0.35, top-8). Le LLM redige a partir du contexte.

```
"Je recrute par IA, suis-je conforme ?" → recherche semantique + reponse citant les articles
"Quelles sont les sanctions ?"          → trouve Article 99 + reponse structuree
```

### Mode WEB (1 appel LLM)

Quand FAISS ne trouve rien (score < 0.35), DuckDuckGo est appele automatiquement (2 recherches : FR + EN). Le LLM repond en precisant que la source est internet.

```
"Qui a gagne Paris-Roubaix ?" → DuckDuckGo → reponse avec source internet
```

## Memoire conversationnelle

La memoire utilise `InMemoryChatMessageHistory` (LangChain natif, non deprecated). Elle est **conditionnelle** :

- **Question independante** ("Quelles sanctions ?") → le LLM recoit uniquement le contexte FAISS, pas l'historique. Evite la pollution par les echanges precedents.
- **Question de suivi** ("Resume ci-dessus", "Explique en detail") → l'historique des 3 derniers echanges est inclus dans le prompt.

La detection de suivi se fait par 2 signaux :
1. References explicites : "ci-dessus", "precedent", "ta reponse", "ces articles"
2. Verbes d'action en debut de phrase : "resume", "explique", "continue", "et pour..."

## Architecture

```
Question
    |
    v
Regex detecte article/considerant ?
    |
   oui → docstore_lookup → texte mot a mot (0 LLM)
    |
   non
    |
    v
FAISS retriever (score > 0.35) ?
    |
   oui → contexte + [historique si suivi] → LLM redige (1 LLM)
    |
   non
    |
    v
DuckDuckGo (FR + EN)
    |
    v
contexte web + [historique si suivi] → LLM redige (1 LLM)
```

## Configuration

| Parametre | Defaut | Description |
|---|---|---|
| `LLM_MODEL` | `"qwen2.5:3b"` | Modele Ollama |
| `SCORE_THRESHOLD` | `0.35` | Seuil de pertinence FAISS |
| `TOP_K` | `8` | Nombre max de documents en mode RAG |

## Exemples

Questions AI Act :
- Donne-moi l'article 5
- Que dit le considerant 12 ?
- Je recrute par IA, suis-je conforme ?
- Quelles sont les sanctions ?

Questions de suivi :
- Resume les points ci-dessus
- Explique en detail
- Et pour le recrutement ?

Questions hors AI Act :
- Qui a gagne Paris-Roubaix ?
- Qui etait Jacques Chirac ?

## Licence

Le texte du Reglement (UE) 2024/1689 est un document officiel de l'Union europeenne accessible sur [EUR-Lex](https://eur-lex.europa.eu/legal-content/FR/TXT/HTML/?uri=OJ:L_202401689).
