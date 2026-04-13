# Chatbot Expert AI Act (UE 2024/1689)

Chatbot RAG pour interroger le Reglement europeen sur l'Intelligence Artificielle. Routage deterministe + LLM pour la redaction. Le LLM utilise d'abord ses propres connaissances, puis recherche sur internet uniquement s'il ne sait pas.

Fonctionne en local (Ollama) et sur Streamlit Cloud (Groq).

## Stack technique

| Composant | Outil | Detail |
|---|---|---|
| Embeddings | `paraphrase-multilingual-mpnet-base-v2` | 278M parametres, 768 dimensions, 50+ langues |
| Vector store | FAISS | 796 chunks indexes (articles + considerants + annexes) |
| LLM (local) | Qwen 2.5 3B via Ollama | ~2 Go RAM, detection automatique |
| LLM (cloud) | Llama 3.1 8B Instant via Groq | Cle API gratuite (GROQ_API_KEY) |
| Recherche web | DuckDuckGo (ddgs) | Dernier recours si le LLM ne sait pas |
| Memoire | InMemoryChatMessageHistory | Systematique (6 derniers messages toujours inclus) |
| Interface | Streamlit | Chat web avec historique |

## Structure du projet

```
RAG_project/
├── requirements.txt                              # Dependances Python
├── chunker.py                                    # Parsing structurel du AI Act (796 chunks)
├── build_index.py                                # Construction de l'index vectoriel FAISS
├── app.py                                        # Application Streamlit (4 modes + memoire)
├── chatbot_ai_act.ipynb                          # Notebook tout-en-un
├── OJ_L_202401689_FR_TXTavec annexes.md         # Texte source complet (articles + 13 annexes)
├── faiss_index/                                  # Index FAISS (committe pour Streamlit Cloud)
├── .gitignore
└── .streamlit/
    └── secrets.toml                              # Cle GROQ_API_KEY (non committe)
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
3. Ajoutez le secret `GROQ_API_KEY` :
   - Dashboard → Settings → Secrets
   - `GROQ_API_KEY = "gsk_votre_cle_ici"`
   - Cle gratuite sur : https://console.groq.com
4. L'app detecte automatiquement l'absence d'Ollama et bascule sur Groq

## Detection automatique Local / Cloud

```
Demarrage de l'app
    |
    v
Ollama repond sur localhost:11434 ?
    |
   oui → ChatOllama (Qwen 2.5 3B, local, gratuit)
    |
   non → ChatGroq (Llama 3.1 8B Instant, API Groq, gratuit)
```

## 4 modes automatiques

Le routage est fait par du **code Python** (regex + score FAISS + detection de suivi), pas par le LLM.

### Mode 1 — DIRECT (0 ou 1 appel LLM)

Regex detecte un article, considerant ou annexe. Retourne le texte integral, ou le passe au LLM si un resume/explication est demande.

```
"Donne-moi l'article 5"        → texte integral (0 LLM)
"Articles 5 et 8"              → texte des deux articles (0 LLM)
"Articles 5 a 8"               → plage d'articles (0 LLM)
"Considerants 1 et 2"          → texte des deux considerants (0 LLM)
"Annexe III"                   → texte integral de l'annexe (0 LLM)
"Resume l'article 5"           → texte passe au LLM pour synthese (1 LLM)
"Explique le considerant 12"   → texte passe au LLM pour explication (1 LLM)
```

### Mode 2 — RAG (1 appel LLM)

FAISS trouve des documents pertinents → le LLM redige avec le contexte.

```
"Je recrute par IA, conforme ?" → FAISS top-8 + LLM redige avec citations
"Quelles sanctions ?"           → Article 99 + reponse structuree
```

### Mode 2bis — CONVERSATION (1 appel LLM, pas de recherche)

Question de suivi detectee (pronoms, verbes d'action, phrase courte). Le LLM repond avec l'historique seul, sans polluer avec DuckDuckGo.

```
"Etait-il un grand peintre ?"  → utilise l'historique (pas de web)
"Resume le"                    → resume le dernier contenu discute
"Comment je m'appelle ?"       → retrouve le prenom dans l'historique
"Repete la blague"             → utilise l'historique
```

### Mode 3 — LLM SEUL (1 appel LLM)

Ni le RAG ni la conversation ne donnent de resultat. Le LLM repond avec ses propres connaissances. S'il ne sait pas, il emet le marqueur `[RECHERCHE_WEB]` qui declenche le mode 4.

```
"Qui est Emmanuel Macron ?"    → reponse directe du LLM
"Raconte une blague de Toto"   → reponse directe du LLM
```

### Mode 4 — WEB (1 appel LLM)

Declenche uniquement si le LLM a repondu `[RECHERCHE_WEB]` (il ne sait pas). DuckDuckGo (FR + EN) + LLM redige.

```
"Qui a gagne Paris-Roubaix ?"  → LLM ne sait pas → DuckDuckGo → reponse
```

## Memoire conversationnelle

`InMemoryChatMessageHistory` (LangChain natif). Les 6 derniers messages sont **toujours** envoyes au LLM.

La detection de suivi conversationnel utilise 2 signaux :
- **Pronoms/references** : il, elle, son, sa, ce, cette, quel, sur quel, a-t-il, est-elle, ci-dessus, ta reponse...
- **Verbes d'action en debut de phrase** : resume, explique, detaille, continue, repete, redis...

Cela evite que "etait-il un peintre aussi ?" parte sur DuckDuckGo et retourne un resultat sans rapport.

## Architecture

```
Question
    |
    v
1. Regex article/considerant/annexe ?
   oui → docstore_lookup → texte integral (0 LLM)
    |                       ou synthese LLM si resume demande (1 LLM)
    v
2. FAISS (score > 0.35) ?
   oui → contexte AI Act + historique → LLM (1 appel)
    |
    v
3. Question de suivi ? (pronoms + phrase courte, OU verbe d'action)
   oui → LLM + historique seul (0 recherche)
    |
    v
4. LLM repond avec ses connaissances propres
   → Si [RECHERCHE_WEB] → DuckDuckGo (FR + EN) → LLM (1 appel)
   → Sinon → reponse directe
```

## Donnees indexees

| Type | Nombre | Source |
|---|---|---|
| Considerants | 236 | (1) a (180+) |
| Articles | 465 | Article premier a Article 113 (decoupes par paragraphe) |
| Annexes | 95 | Annexe I a XIII (13 annexes, decoupees par section) |
| **Total** | **796 chunks** | |

## Configuration

| Parametre | Defaut | Description |
|---|---|---|
| `OLLAMA_MODEL` | `"qwen2.5:3b"` | Modele local (Ollama) |
| `GROQ_MODEL` | `"llama-3.1-8b-instant"` | Modele cloud (Groq) |
| `SCORE_THRESHOLD` | `0.35` | Seuil de pertinence FAISS |
| `TOP_K` | `8` | Nombre max de documents en mode RAG |

## Licence

Le texte du Reglement (UE) 2024/1689 est un document officiel de l'Union europeenne accessible sur [EUR-Lex](https://eur-lex.europa.eu/legal-content/FR/TXT/HTML/?uri=OJ:L_202401689).
