# Chatbot Expert AI Act (UE 2024/1689)

Chatbot RAG pour interroger le Règlement européen sur l'Intelligence Artificielle.

**Deux versions disponibles :**
- **`app.py`** — Routage déterministe (regex + FAISS + code Python). Fonctionne en local (Ollama) et sur Streamlit Cloud (Groq).
- **`app_agent.py`** — Agent LangGraph avec tool calling. Le LLM décide lui-même quel outil appeler. Cloud uniquement (Groq).

## Stack technique

| Composant | app.py (déterministe) | app_agent.py (agent) |
|---|---|---|
| Routage | Code Python (regex, FAISS, pronoms) | LLM décide via tool calling |
| LLM local | Qwen 2.5 3B via Ollama | — (cloud uniquement) |
| LLM cloud | Llama 3.1 8B via Groq | Llama 4 Scout 17B via Groq |
| Framework | LangChain (invoke) | LangGraph (create_react_agent) |
| Outils | Fonctions internes | 5 outils @tool déclarés |
| Embeddings | paraphrase-multilingual-mpnet-base-v2 (278M, 768 dim) | idem |
| Vector store | FAISS (796 chunks) | idem |
| Recherche web | DuckDuckGo (dernier recours) | idem |
| Mémoire | 6 derniers messages systématiques | idem |
| Interface | Streamlit | Streamlit (+ affichage des outils appelés) |

## Structure du projet

```
RAG_project/
├── app.py                                        # Version déterministe (5 modes de routage)
├── app_agent.py                                  # Version agent LangGraph (5 outils @tool)
├── chatbot_ai_act.ipynb                          # Notebook version déterministe
├── chatbot_agent.ipynb                           # Notebook version agent (annoté)
├── chunker.py                                    # Parsing structurel du AI Act (796 chunks)
├── build_index.py                                # Construction de l'index vectoriel FAISS
├── OJ_L_202401689_FR_TXTavec annexes.md         # Texte source complet (articles + 13 annexes)
├── faiss_index/                                  # Index FAISS (committé pour Streamlit Cloud)
├── requirements.txt                              # Dépendances Python
├── DOCUMENTATION_TECHNIQUE.md                    # Documentation technique détaillée
├── Presentation_Projet_RAG_v3.pptx              # Présentation PowerPoint
├── .gitignore
└── .streamlit/
    └── secrets.toml                              # Clé GROQ_API_KEY (non committée)
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

# Installer le modèle LLM local (pour app.py uniquement)
ollama pull qwen2.5:3b

# Lancer la version déterministe
streamlit run app.py

# Lancer la version agent
streamlit run app_agent.py
```

## Déploiement sur Streamlit Cloud

1. Poussez le code sur GitHub (incluant `faiss_index/` dans le repo)
2. Connectez le repo à Streamlit Cloud
3. Ajoutez le secret `GROQ_API_KEY` :
   - Dashboard → Settings → Secrets
   - `GROQ_API_KEY = "gsk_votre_clé_ici"`
   - Clé gratuite sur : https://console.groq.com
4. Choisissez le **Main file path** :
   - `app.py` pour la version déterministe
   - `app_agent.py` pour la version agent

## Les deux architectures

### app.py — Routage déterministe

Le routage est fait par du **code Python** (regex + score FAISS + détection de pronoms), pas par le LLM.

```
Question
    │
    ▼
① Regex article/considérant/annexe ?
   oui → docstore_lookup → texte intégral (0 LLM)
    │                       ou synthèse LLM si résumé demandé
    ▼
② FAISS (score > 0.35) ?
   oui → contexte AI Act + historique → LLM (1 appel)
    │
    ▼
③ Question de suivi ? (pronoms + phrase courte, ou verbe d'action)
   oui → LLM + historique seul (0 recherche)
    │
    ▼
④ LLM répond avec ses connaissances propres
   → Si [RECHERCHE_WEB] → DuckDuckGo → LLM
   → Sinon → réponse directe
```

### app_agent.py — Agent LangGraph

Le **LLM décide lui-même** quel outil appeler via le mécanisme de **tool calling** (boucle ReAct).

```
Question
    │
    ▼
Agent LangGraph (create_react_agent)
Le LLM reçoit 5 outils et décide :
    │
    ├── recherche_article("5")        → docstore_lookup → texte intégral
    ├── recherche_considerant("12")   → docstore_lookup → texte intégral
    ├── recherche_annexe("III")       → docstore_lookup → texte intégral
    ├── recherche_ia_act("obligations")→ FAISS retriever → chunks pertinents
    │
    │  Si aucun outil AI Act ne trouve :
    ├── Connaissances internes du modèle (données d'entraînement)
    │   → Si suffisant : réponse directe
    │   → Si insuffisant ↓
    └── recherche_web("actualité")    → DuckDuckGo (indique "source internet")
    │
    ▼
Réponse rédigée par le LLM
```

**Logique de priorité (définie dans le prompt système) :**
1. Outils AI Act (base FAISS locale) → pas d'accès internet
2. Connaissances internes du modèle → pas d'accès internet
3. `recherche_web` (DuckDuckGo) → accès internet, **toujours signalé dans la réponse**

**5 outils déclarés avec le décorateur `@tool` :**

```python
@tool
def recherche_article(article_num: str) -> str:
    """Recherche le texte intégral d'un article du AI Act par son numéro."""
    docs = docstore_lookup(db, {"article": article_num})
    return format_docs(docs)
```

Le LLM lit les **docstrings** pour décider quand appeler chaque outil. Il ne voit jamais le code Python.

## Comparaison des deux approches

| Critère | app.py (déterministe) | app_agent.py (agent) |
|---|---|---|
| Prédictibilité | Même question → même mode, toujours | Peut varier selon le contexte |
| Rapidité | Regex + FAISS = millisecondes | Plusieurs appels LLM (plus lent) |
| Coût en tokens | 0 token pour le routage | ~500 tokens par décision |
| Déboguabilité | Mode affiché (Direct/RAG/Web) | Outils appelés affichés |
| Multi-articles | Regex "articles 5 et 8" | L'agent appelle l'outil 2 fois |
| Résumé | Regex wants_analysis | L'agent résume naturellement |
| Flexibilité | Limitée aux regex prédéfinies | Le LLM s'adapte à toute formulation |
| LLM requis | Fonctionne avec 3B (routage par code) | Nécessite ≥ 8B (tool calling fiable) |

## Mémoire conversationnelle

`InMemoryChatMessageHistory` (LangChain natif). Les 6 derniers messages sont **toujours** envoyés au LLM dans les deux versions.

**app.py** ajoute une détection de suivi conversationnel (pronoms + verbes d'action) pour éviter les recherches web parasites.

**app_agent.py** laisse l'agent gérer naturellement les questions de suivi grâce à l'historique envoyé dans les messages. Pour les questions hors AI Act, l'agent cherche d'abord dans les connaissances internes du modèle, puis utilise `recherche_web` si insuffisant — et indique toujours que l'information vient d'internet.

## Données indexées

| Type | Nombre | Source |
|---|---|---|
| Considérants | 236 | (1) à (180+) |
| Articles | 465 | Article premier à Article 113 (découpés par paragraphe) |
| Annexes | 95 | Annexe I à XIII (13 annexes, découpées par section) |
| **Total** | **796 chunks** | |

## Configuration

| Paramètre | app.py | app_agent.py | Description |
|---|---|---|---|
| `OLLAMA_MODEL` | `qwen2.5:3b` | — | Modèle local |
| `GROQ_MODEL` | `llama-3.1-8b-instant` | `llama-4-scout-17b-16e-instruct` | Modèle cloud |
| `SCORE_THRESHOLD` | `0.35` | `0.35` | Seuil de pertinence FAISS |
| `TOP_K` | `8` | `8` | Nombre max de documents RAG |

**Pourquoi Llama 4 Scout pour l'agent ?**
L'agent fait plusieurs appels LLM par question. Llama 3.1 8B a un quota de 6K tokens/min → dépassé en 2-3 questions. Llama 4 Scout 17B a **30K tokens/min** (5× plus), suffisant pour un agent.

## Licence

Le texte du Règlement (UE) 2024/1689 est un document officiel de l'Union européenne accessible sur [EUR-Lex](https://eur-lex.europa.eu/legal-content/FR/TXT/HTML/?uri=OJ:L_202401689).
