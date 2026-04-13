# Chatbot Expert AI Act (UE 2024/1689)

Chatbot RAG pour interroger le Reglement europeen sur l'Intelligence Artificielle. Routage deterministe + LLM pour la redaction. Recherche internet automatique via DuckDuckGo si l'information n'est pas dans le AI Act.

Fonctionne en local (Ollama) et sur Streamlit Cloud (HuggingFace Inference API).

## Stack technique

| Composant | Outil | Detail |
|---|---|---|
| Embeddings | `paraphrase-multilingual-mpnet-base-v2` | 278M parametres, 768 dimensions, 50+ langues |
| Vector store | FAISS | Index persistant, recherche avec seuil de score |
| LLM (local) | Qwen 2.5 3B via Ollama | ~2 Go RAM, detection automatique |
| LLM (cloud) | Mistral 7B Instruct via HuggingFace | Cle API gratuite (HF_TOKEN) |
| Recherche web | DuckDuckGo (ddgs) | Fallback quand l'info n'est pas dans le AI Act |
| Memoire | InMemoryChatMessageHistory | Conditionnelle (activee sur les questions de suivi) |
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
├── .gitignore
└── .streamlit/
    └── secrets.toml                      # Cle HF_TOKEN (non committe)
```

## Installation locale

```bash
git clone <url-du-depot>
cd RAG_project

python -m venv .venv
source .venv/bin/activate        # Linux/Mac
# .venv\Scripts\activate         # Windows

pip install -r requirements.txt

# Construire l'index (une seule fois)
python build_index.py

# Installer le modele LLM local
ollama pull qwen2.5:3b

# Lancer
streamlit run app.py
```

## Deploiement sur Streamlit Cloud

1. Poussez le code sur GitHub (incluant `faiss_index/` dans le repo)
2. Connectez le repo a Streamlit Cloud
3. Ajoutez le secret `HF_TOKEN` :
   - Dashboard → Settings → Secrets
   - `HF_TOKEN = "hf_votre_cle_ici"`
   - Cle gratuite sur : https://huggingface.co/settings/tokens
4. L'app detecte automatiquement l'absence d'Ollama et bascule sur HuggingFace

## Detection automatique Local / Cloud

```
Demarrage de l'app
    |
    v
Ollama repond sur localhost:11434 ?
    |
   oui → ChatOllama (Qwen 2.5 3B, local, gratuit)
    |
   non → ChatHuggingFace (Mistral 7B Instruct, API, gratuit avec HF_TOKEN)
```

L'import `langchain-ollama` est fait dynamiquement : il n'est jamais execute sur le cloud, donc pas de `ModuleNotFoundError`.

## 3 modes automatiques

Le routage est fait par du **code Python** (regex + score FAISS), pas par le LLM.

### Mode DIRECT (0 appel LLM)

```
"Donne-moi l'article 5"       → texte integral (regex + docstore_lookup)
"Que dit le considerant 12 ?" → texte exact
```

### Mode RAG (1 appel LLM)

```
"Je recrute par IA, conforme ?" → FAISS top-8 + LLM redige avec citations
"Quelles sanctions ?"           → Article 99 + reponse structuree
```

### Mode WEB (1 appel LLM)

```
"Qui a gagne Paris-Roubaix ?"  → DuckDuckGo (FR + EN) + LLM redige
```

## Memoire conversationnelle

`InMemoryChatMessageHistory` (LangChain natif). Conditionnelle :

- **Question independante** → pas d'historique (evite la pollution)
- **Question de suivi** ("resume ci-dessus", "explique", "et pour...") → 3 derniers echanges inclus

## Architecture

```
Question
    |
    v
Regex article/considerant ?
   oui → docstore_lookup → texte mot a mot (0 LLM)
    |
   non → FAISS (score > 0.35) ?
          oui → contexte + [historique si suivi] → LLM (1 appel)
           |
          non → DuckDuckGo (FR + EN) → LLM (1 appel)
```

## Configuration

| Parametre | Defaut | Description |
|---|---|---|
| `OLLAMA_MODEL` | `"qwen2.5:3b"` | Modele local (Ollama) |
| `HF_MODEL` | `"mistralai/Mistral-7B-Instruct-v0.3"` | Modele cloud (HuggingFace) |
| `SCORE_THRESHOLD` | `0.35` | Seuil de pertinence FAISS |
| `TOP_K` | `8` | Nombre max de documents en mode RAG |

## Licence

Le texte du Reglement (UE) 2024/1689 est un document officiel de l'Union europeenne accessible sur [EUR-Lex](https://eur-lex.europa.eu/legal-content/FR/TXT/HTML/?uri=OJ:L_202401689).
