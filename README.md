# Chatbot Expert AI Act (UE 2024/1689)

Pipeline RAG (Retrieval-Augmented Generation) pour interroger le Reglement europeen sur l'Intelligence Artificielle en langage naturel. Le systeme parse le texte officiel, le decoupe en chunks semantiques, les indexe avec FAISS, puis genere des reponses contextualisees via un LLM local.

Aucune donnee n'est envoyee en ligne : tout tourne en local.

## Stack technique

| Composant | Outil | Detail |
|---|---|---|
| Embeddings | `paraphrase-multilingual-mpnet-base-v2` | 278M parametres, 768 dimensions, 50+ langues |
| Vector store | FAISS (Facebook AI Similarity Search) | Recherche par similarite cosinus avec seuil de score |
| LLM | Gemma 3 1B via Ollama | Inference locale |
| Framework | LangChain (LCEL) | Orchestration de la chaine RAG |
| Interface | Streamlit | Interface chat web |

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

### Description des fichiers

**`chunker.py`** — Parse le Markdown du reglement et produit 641 chunks structures :
- 180 considerants (motivations legislatives)
- 461 chunks d'articles (113 articles, decoupes par paragraphe si trop longs)
- Chaque chunk contient un prefixe hierarchique (`Chapitre > Section > Article`) et des metadonnees riches (type, numero, titre, chapitre, section, paragraphe)

**`build_index.py`** — Encode les 641 chunks avec `sentence-transformers` et construit l'index FAISS persistant, sauvegarde sur disque dans `faiss_index/`.

**`app.py`** — Application Streamlit avec interface chat. Integre trois mecanismes anti-hallucinations :
- Seuil de score sur le retriever (seuls les documents pertinents sont retournes)
- Garde-fou pre-LLM (si aucun document pertinent, refus poli sans appeler le LLM)
- Prompt systeme strict avec regles de refus explicites

**`chatbot_ai_act.ipynb`** — Version notebook autonome du pipeline complet (parsing, indexation, recherche, generation). Peut etre utilise a la place de `app.py`.

## Prerequis

- Python 3.10+
- [Ollama](https://ollama.com/download) installe et lance

## Installation

```bash
# Cloner le depot
git clone <url-du-depot>
cd RAG_project

# Creer un environnement virtuel
python -m venv .venv
source .venv/bin/activate        # Linux/Mac
# .venv\Scripts\activate         # Windows

# Installer les dependances
pip install -r requirements.txt

# Telecharger le modele LLM
ollama pull gemma3:1b
```

## Utilisation

### 1. Construire l'index (une seule fois)

```bash
python build_index.py
```

Cree le dossier `faiss_index/` contenant les vecteurs et les documents indexes.
L'index est persistant : il est sauvegarde sur disque et recharge automatiquement aux lancements suivants sans re-encodage.

### 2a. Lancer l'interface Streamlit

```bash
streamlit run app.py
```

Ouvre une interface chat dans le navigateur sur `http://localhost:8501`.

### 2b. Alternative : utiliser le notebook

Ouvrir `chatbot_ai_act.ipynb` dans Jupyter ou VS Code et executer les cellules dans l'ordre.

## Architecture du pipeline RAG

```
Question utilisateur
    |
    v
[Retriever FAISS]  -- similarity_score_threshold (seuil: 0.4)
    |
    |-- 0 documents pertinents --> Refus poli (LLM jamais appele)
    |
    |-- 1 a 5 documents  -------> [Prompt systeme + contexte + question]
                                        |
                                        v
                                   [LLM Gemma 3 1B]
                                        |
                                        v
                                   Reponse + Sources
```

### Strategie de chunking

Le chunking exploite la structure hierarchique du reglement plutot qu'un decoupage aveugle par taille fixe :

| Element | Quantite | Strategie |
|---|---|---|
| Considerants | 180 | 1 chunk par considerant |
| Articles courts | ~70 | 1 chunk par article |
| Articles longs (> 2000 car.) | ~43 | Decoupe par paragraphe numerote |
| **Total** | **641 chunks** | |

Chaque chunk est prefixe avec son contexte hierarchique :
```
Chapitre III - SYSTEMES D'IA A HAUT RISQUE > Section 1 > Article 6 : Regles relatives a la classification...
```

Ce chunking structurel a ete prefere au `MarkdownHeaderTextSplitter` de LangChain car le document ne contient aucun header Markdown (`#`, `##`). La structure est encodee en texte brut (`CHAPITRE`, `Article`, `|(N)|`), ce que le splitter standard ne detecte pas.

### Mecanismes anti-hallucinations

Trois couches de protection empechent le chatbot d'inventer des reponses :

1. **Seuil de score sur le retriever** — Le retriever utilise `similarity_score_threshold` au lieu de retourner systematiquement k documents. Si aucun chunk n'a un score superieur a 0.4, la liste retournee est vide. Les questions hors-sujet (ex: "Qu'est-ce que le Bitcoin ?") obtiennent des scores < 0.3 et sont filtrees.

2. **Garde-fou pre-LLM** — Si le retriever retourne 0 documents, le LLM n'est jamais appele. Un message de refus pre-ecrit est retourne directement, sans cout de calcul ni risque d'hallucination.

3. **Prompt systeme strict** — Six regles explicites contraignent le LLM a ne repondre qu'a partir du contexte fourni, avec des phrases de refus imposees pour les cas ou l'information n'est pas disponible.

## Configuration

Les parametres modifiables sont en haut de `app.py` (ou cellule 3 du notebook) :

| Parametre | Valeur par defaut | Description |
|---|---|---|
| `LLM_MODEL` | `"gemma3:1b"` | Modele Ollama |
| `SCORE_THRESHOLD` | `0.4` | Seuil de pertinence (augmenter = plus strict) |
| `TOP_K` | `5` | Nombre max de documents retournes |
| `MODEL_NAME` | `"sentence-transformers/paraphrase-multilingual-mpnet-base-v2"` | Modele d'embeddings |

Modeles LLM compatibles (Ollama) :

| Modele | Parametres | RAM requise | Precision |
|---|---|---|---|
| `gemma3:1b` | 1B | ~1.5 Go | Correcte |
| `gemma3:4b` | 4B | ~3 Go | Bonne |
| `mistral` | 7B | ~5 Go | Tres bonne |
| `llama3.2:3b` | 3B | ~2.5 Go | Bonne |

## Exemples de questions

Questions couvertes par l'AI Act :
- Quelles sont les pratiques d'IA interdites ?
- Qu'est-ce qu'un systeme d'IA a haut risque ?
- Quelles sont les obligations de transparence ?
- Qui est responsable de la conformite d'un systeme d'IA ?
- Quelles sanctions sont prevues en cas de non-conformite ?

Questions hors-sujet (refus poli attendu) :
- Qu'est-ce que le Bitcoin ?
- Quelle est la capitale de la France ?

## Licence

Le texte du Reglement (UE) 2024/1689 est un document officiel de l'Union europeenne, librement accessible sur [EUR-Lex](https://eur-lex.europa.eu/legal-content/FR/TXT/HTML/?uri=OJ:L_202401689).
