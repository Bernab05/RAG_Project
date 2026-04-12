# Chatbot Expert AI Act (UE 2024/1689)

Pipeline RAG (Retrieval-Augmented Generation) pour interroger le Reglement europeen sur l'Intelligence Artificielle en langage naturel. Tout tourne en local, aucune donnee n'est envoyee en ligne.

## Stack technique

| Composant | Outil | Detail |
|---|---|---|
| Embeddings | `paraphrase-multilingual-mpnet-base-v2` | 278M parametres, 768 dimensions, 50+ langues |
| Vector store | FAISS (Facebook AI Similarity Search) | Index persistant sur disque |
| LLM | Gemma 3 1B via Ollama | Inference locale |
| Framework | LangChain (LCEL) | Orchestration de la chaine RAG |
| Interface | Streamlit | Chat web avec historique |

## Structure du projet

```
RAG_project/
├── requirements.txt                      # Dependances Python
├── chunker.py                            # Parsing structurel du AI Act (641 chunks)
├── build_index.py                        # Construction de l'index vectoriel FAISS
├── app.py                                # Interface Streamlit (chat web)
├── chatbot_ai_act.ipynb                  # Notebook tout-en-un (alternative a app.py)
├── L-202401689FR.000101.fmx.xml.md      # Texte source du AI Act (FR)
└── .gitignore
```

## Description des fichiers

**`chunker.py`** — Parse le Markdown du reglement et produit 641 chunks structures :
- 180 considerants (motivations legislatives)
- 461 chunks d'articles (113 articles, decoupes par paragraphe si > 2000 caracteres)
- Chaque chunk contient un prefixe hierarchique et 8 champs de metadonnees

**`build_index.py`** — Encode les 641 chunks avec sentence-transformers et construit l'index FAISS persistant sauvegarde sur disque.

**`app.py`** — Application Streamlit avec deux modes de fonctionnement :
- Mode direct : restitution mot a mot via recherche dans le docstore (bypass vectoriel)
- Mode RAG : recherche semantique avec seuil de score + generation LLM
- Trois mecanismes anti-hallucinations integres

**`chatbot_ai_act.ipynb`** — Version notebook autonome avec le meme pipeline.

## Prerequis

- Python 3.10+
- [Ollama](https://ollama.com/download) installe et lance

## Installation

```bash
git clone <url-du-depot>
cd RAG_project

python -m venv .venv
source .venv/bin/activate        # Linux/Mac
# .venv\Scripts\activate         # Windows

pip install -r requirements.txt

ollama pull gemma3:1b
```

## Utilisation

### 1. Construire l'index (une seule fois)

```bash
python build_index.py
```

L'index est persistant sur disque (`faiss_index/`), recharge sans re-encodage.

### 2. Lancer le chatbot

```bash
streamlit run app.py
```

Ou ouvrir `chatbot_ai_act.ipynb` dans Jupyter.

## Deux modes de fonctionnement

### Mode direct (filtrage metadata)

Mentionnez un article, chapitre ou considerant pour obtenir le texte officiel mot a mot, sans appel au LLM.

```
"Donne-moi l'article 5"          → texte integral de l'article 5 (9 paragraphes)
"Que dit le considerant 12 ?"    → texte exact du considerant 12
"Articles du chapitre III"       → liste des articles du chapitre III
"Article 6 paragraphe 3"         → paragraphe precis
```

Le mode direct utilise `docstore_lookup()` qui itere sur les 641 documents du docstore et filtre par metadata en Python. Cette approche bypass la recherche vectorielle car FAISS fait du post-filtrage (recherche vectorielle d'abord, filtrage ensuite), ce qui peut manquer des documents quand la question est semantiquement eloignee du contenu.

### Mode RAG (recherche semantique)

Pour les questions libres, le pipeline RAG classique s'active :

```
"Quelles sont les pratiques d'IA interdites ?"  → recherche semantique + LLM
"Qui est responsable de la conformite ?"         → recherche semantique + LLM
```

## Mecanismes anti-hallucinations

1. **Seuil de score** (0.4) — Le retriever ne retourne que les documents au-dessus du seuil de similarite. Les questions hors-sujet obtiennent des scores < 0.3.

2. **Garde-fou pre-LLM** — Si 0 documents pertinents, le LLM n'est jamais appele. Un refus poli est retourne directement.

3. **Prompt systeme strict** — Six regles explicites contraignent le LLM. Phrases de refus imposees pour les cas hors contexte.

## Configuration

Parametres modifiables en haut de `app.py` :

| Parametre | Defaut | Description |
|---|---|---|
| `LLM_MODEL` | `"gemma3:1b"` | Modele Ollama |
| `SCORE_THRESHOLD` | `0.4` | Seuil de pertinence (augmenter = plus strict) |
| `TOP_K` | `5` | Nombre max de documents en mode RAG |

Modeles LLM compatibles :

| Modele | RAM | Precision |
|---|---|---|
| `gemma3:1b` | ~1.5 Go | Correcte |
| `gemma3:4b` | ~3 Go | Bonne |
| `mistral` | ~5 Go | Tres bonne |

## Exemples de questions

Questions couvertes :
- Quelles sont les pratiques d'IA interdites ?
- Qu'est-ce qu'un systeme d'IA a haut risque ?
- Donne-moi l'article 60
- Que dit le considerant 176 ?

Questions hors-sujet (refus poli) :
- Qu'est-ce que le Bitcoin ?
- Quelle est la capitale de la France ?

## Licence

Le texte du Reglement (UE) 2024/1689 est un document officiel de l'Union europeenne accessible sur [EUR-Lex](https://eur-lex.europa.eu/legal-content/FR/TXT/HTML/?uri=OJ:L_202401689).
