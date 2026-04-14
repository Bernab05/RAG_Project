"""
Chatbot Expert AI Act — Version AGENT LangGraph

Utilise un vrai agent LangGraph (create_react_agent) avec 5 outils.
Le LLM (Llama 3.1 8B via Groq) décide lui-même quel outil appeler.

Mode cloud uniquement (Groq). Pour la version locale/déterministe → app.py

Outils disponibles pour l'agent :
1. recherche_article   — texte intégral d'un article par numéro
2. recherche_considerant — texte d'un considérant par numéro
3. recherche_annexe    — texte d'une annexe par chiffres romains
4. recherche_ia_act    — recherche sémantique FAISS dans tout le AI Act
5. recherche_web       — recherche internet DuckDuckGo (dernier recours)
"""

import os
from pathlib import Path

import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.tools import tool
from langchain_groq import ChatGroq
from langchain_community.tools import DuckDuckGoSearchRun
from langgraph.prebuilt import create_react_agent

# =============================================
# Configuration
# =============================================
INDEX_DIR       = Path(__file__).parent / "faiss_index"
MODEL_NAME      = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
SCORE_THRESHOLD = 0.35
TOP_K           = 8
GROQ_MODEL      = "meta-llama/llama-4-scout-17b-16e-instruct"

# =============================================
# Prompt système pour l'agent
# =============================================
AGENT_PROMPT = """\
Tu es un assistant expert du Règlement européen sur l'Intelligence Artificielle \
(AI Act, Règlement UE 2024/1689). Réponds en français.

Tu as 5 outils à ta disposition :
- recherche_article : retourne le texte intégral d'un article par son numéro (ex: "5", "66")
- recherche_considerant : retourne le texte d'un considérant par son numéro (ex: "12", "55")
- recherche_annexe : retourne le texte d'une annexe par son numéro en chiffres romains (ex: "III", "XI")
- recherche_ia_act : recherche sémantique dans tout le AI Act pour les questions générales
- recherche_web : recherche sur internet via DuckDuckGo

RÈGLES D'UTILISATION DES OUTILS :
1. Quand l'utilisateur demande un article précis (ex: "article 5", "art. 66"), \
   utilise recherche_article avec le numéro.
2. Quand l'utilisateur demande un considérant (ex: "considérant 12"), \
   utilise recherche_considerant avec le numéro.
3. Quand l'utilisateur demande une annexe (ex: "annexe III"), \
   utilise recherche_annexe avec le numéro romain.
4. Pour les questions générales sur le AI Act (ex: "obligations IA haut risque", \
   "pratiques interdites"), utilise recherche_ia_act.
5. Si l'utilisateur demande PLUSIEURS articles (ex: "articles 5 et 8"), \
   appelle recherche_article PLUSIEURS FOIS, une fois par numéro.
6. Si la question ne concerne PAS le AI Act, cherche d'abord dans tes connaissances \
   propres (données d'entraînement du modèle). Si tes connaissances sont suffisantes, \
   réponds directement.
7. Si tes connaissances propres ne suffisent PAS (information trop récente, trop \
   spécifique, ou incertaine), utilise recherche_web. Quand tu utilises recherche_web, \
   PRÉCISE TOUJOURS dans ta réponse que l'information provient d'une recherche internet.

RÈGLES DE RÉPONSE :
1. Cite toujours les articles et considérants exacts (ex: "Article 6, paragraphe 2").
2. Si le contexte contient des obligations ou interdictions, LISTE-LES précisément.
3. Structure ta réponse avec des titres et des puces si nécessaire.
4. Si l'information vient d'internet (recherche_web), INDIQUE-LE clairement \
   en début de réponse : "D'après une recherche internet :".
5. Ne dis JAMAIS "consultez le texte complet". Utilise ce que tu as.
6. Tu as accès à l'historique de conversation. Si l'utilisateur fait référence \
   à un échange précédent (un prénom, un sujet, une personne), utilise l'historique.
7. Si l'utilisateur demande un RÉSUMÉ, fournis un résumé synthétique et non le texte intégral.
"""

# =============================================
# Chargement des ressources (cache Streamlit)
# =============================================

@st.cache_resource
def load_vectorstore():
    embeddings = HuggingFaceEmbeddings(
        model_name=MODEL_NAME,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    return FAISS.load_local(str(INDEX_DIR), embeddings, allow_dangerous_deserialization=True)


@st.cache_resource
def load_llm():
    groq_api_key = (
        os.environ.get("GROQ_API_KEY")
        or st.secrets.get("GROQ_API_KEY", "")
    )
    if not groq_api_key:
        st.error(
            "Clé Groq requise pour le mode agent.\n\n"
            "1. Créez un compte gratuit sur https://console.groq.com\n"
            "2. Générez une clé API\n"
            "3. Ajoutez dans Streamlit Cloud → Settings → Secrets :\n"
            '   `GROQ_API_KEY = "gsk_..."`'
        )
        st.stop()
    return ChatGroq(model=GROQ_MODEL, api_key=groq_api_key, temperature=0.1)


# =============================================
# Fonctions de base (réutilisées de app.py)
# =============================================

def docstore_lookup(db, filters):
    """Recherche exacte dans le docstore FAISS par métadonnées."""
    results = []
    for doc_id in db.index_to_docstore_id.values():
        doc = db.docstore.search(doc_id)
        if all(doc.metadata.get(k) == v for k, v in filters.items()):
            results.append(doc)
    results.sort(key=lambda d: int(d.metadata.get("paragraph", "0") or "0"))
    return results


def format_docs(docs):
    """Concatène les contenus des documents avec un séparateur."""
    return "\n\n---\n\n".join(doc.page_content for doc in docs)


def get_sources(docs):
    """Extrait les labels de sources lisibles pour l'affichage."""
    sources = []
    for doc in docs:
        m = doc.metadata
        if m.get("type") == "article":
            label = f"Article {m['article']} : {m['title']}"
            if m.get("chapter"):
                label = f"Chapitre {m['chapter']} > {label}"
        elif m.get("type") == "annexe":
            label = f"Annexe {m['annexe']} : {m['title']}"
        else:
            label = m.get("title", "Considérant")
        sources.append(label)
    return sources


def get_chat_history() -> InMemoryChatMessageHistory:
    """Retourne l'historique de conversation stocké dans la session Streamlit."""
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = InMemoryChatMessageHistory()
    return st.session_state.chat_history


# =============================================
# Initialisation
# =============================================

db = load_vectorstore()
llm = load_llm()
retriever = db.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={"k": TOP_K, "score_threshold": SCORE_THRESHOLD},
)

# =============================================
# Définition des 5 outils (@tool)
# Les docstrings sont lues par le LLM pour décider quel outil appeler.
# =============================================

@tool
def recherche_article(article_num: str) -> str:
    """Recherche le texte intégral d'un article du AI Act par son numéro.
    Utilise cet outil quand l'utilisateur demande un article précis.

    Args:
        article_num: numéro de l'article sous forme de chaîne (ex: "5", "66", "1")
    """
    # Normaliser : "premier" → "1"
    num = "1" if article_num.lower().strip() in ("premier", "1er") else article_num.strip()
    docs = docstore_lookup(db, {"article": num})
    if not docs:
        return f"Article {num} non trouvé dans la base AI Act."
    sources = get_sources(docs)
    # Stocker les sources pour affichage
    if "last_sources" not in st.session_state:
        st.session_state.last_sources = []
    st.session_state.last_sources.extend(sources)
    return format_docs(docs)


@tool
def recherche_considerant(numero: str) -> str:
    """Recherche le texte d'un considérant du AI Act par son numéro.
    Utilise cet outil quand l'utilisateur demande un considérant précis.

    Args:
        numero: numéro du considérant sous forme de chaîne (ex: "12", "55")
    """
    num = numero.strip().strip("()")
    docs = docstore_lookup(db, {"type": "considerant", "numero": f"({num})"})
    if not docs:
        return f"Considérant {num} non trouvé dans la base AI Act."
    sources = get_sources(docs)
    if "last_sources" not in st.session_state:
        st.session_state.last_sources = []
    st.session_state.last_sources.extend(sources)
    return format_docs(docs)


@tool
def recherche_annexe(annexe_num: str) -> str:
    """Recherche le texte intégral d'une annexe du AI Act par son numéro romain.
    Utilise cet outil quand l'utilisateur demande une annexe.

    Args:
        annexe_num: numéro de l'annexe en chiffres romains (ex: "III", "XI", "I")
    """
    num = annexe_num.strip().upper()
    docs = docstore_lookup(db, {"type": "annexe", "annexe": num})
    if not docs:
        return f"Annexe {num} non trouvée dans la base AI Act."
    sources = get_sources(docs)
    if "last_sources" not in st.session_state:
        st.session_state.last_sources = []
    st.session_state.last_sources.extend(sources)
    return format_docs(docs)


@tool
def recherche_ia_act(query: str) -> str:
    """Recherche sémantique dans le AI Act pour les questions générales.
    Utilise cet outil pour les questions sur les obligations, interdictions,
    sanctions, conformité, etc. — tout ce qui n'est pas un numéro d'article précis.

    Args:
        query: la question ou les mots-clés à rechercher dans le AI Act
    """
    docs = retriever.invoke(query)
    if not docs:
        return "Aucun passage pertinent trouvé dans le AI Act pour cette requête."
    sources = get_sources(docs)
    if "last_sources" not in st.session_state:
        st.session_state.last_sources = []
    st.session_state.last_sources.extend(sources)
    source_list = "\n".join(f"- {s}" for s in sources)
    context = format_docs(docs)
    if len(context) > 4000:
        context = context[:4000] + "\n\n[... tronqué ...]"
    return f"Sources AI Act :\n{source_list}\n\nExtraits officiels :\n{context}"


@tool
def recherche_web(query: str) -> str:
    """Recherche sur internet via DuckDuckGo.
    Utilise cet outil UNIQUEMENT si les 4 autres outils n'ont pas trouvé
    l'information ET que tes propres connaissances ne suffisent pas.

    Args:
        query: la question à rechercher sur internet
    """
    search = DuckDuckGoSearchRun()
    web_parts = []
    try:
        r_fr = search.invoke(query)
        if r_fr:
            web_parts.append(r_fr)
    except Exception:
        pass
    try:
        r_en = search.invoke(query + " results 2025")
        if r_en:
            web_parts.append(r_en)
    except Exception:
        pass
    web_results = "\n\n".join(web_parts)
    if not web_results or len(web_results) < 50:
        return "Aucun résultat pertinent trouvé sur internet."
    if "last_sources" not in st.session_state:
        st.session_state.last_sources = []
    st.session_state.last_sources.append("Recherche internet (DuckDuckGo)")
    return f"Résultats internet (DuckDuckGo) :\n\n{web_results[:2500]}"


# =============================================
# Création de l'agent LangGraph
# =============================================

tools = [recherche_article, recherche_considerant, recherche_annexe, recherche_ia_act, recherche_web]

agent = create_react_agent(
    model=llm,
    tools=tools,
)

# =============================================
# Fonction principale : traitement d'une question
# =============================================

def process_question(question: str) -> dict:
    """
    Envoie la question à l'agent LangGraph.
    L'agent décide quel(s) outil(s) appeler et rédige la réponse.
    """
    # Réinitialiser les sources pour cette question
    st.session_state.last_sources = []

    # Construire les messages : prompt système + historique + question
    messages = [SystemMessage(content=AGENT_PROMPT)]

    # Ajouter l'historique (6 derniers messages)
    history = get_chat_history()
    if history.messages:
        for msg in history.messages[-6:]:
            content = msg.content[:800] + "..." if len(msg.content) > 800 else msg.content
            messages.append(type(msg)(content=content))

    # Ajouter la question
    messages.append(HumanMessage(content=question))

    # Invoquer l'agent
    try:
        result = agent.invoke({"messages": messages})
    except Exception as e:
        error_msg = str(e)
        if "401" in error_msg or "authentication" in error_msg.lower() or "invalid_api_key" in error_msg.lower():
            return {
                "response": f"Erreur 401 : clé GROQ_API_KEY invalide.\nVérifiez le secret dans Streamlit Cloud.",
                "sources": [],
                "tools_called": [],
            }
        elif "429" in error_msg or "rate" in error_msg.lower():
            return {
                "response": f"Erreur 429 : quota Groq atteint. Réessayez dans quelques secondes.",
                "sources": [],
                "tools_called": [],
            }
        else:
            return {
                "response": f"Erreur agent : {error_msg}",
                "sources": [],
                "tools_called": [],
            }

    # Extraire la réponse finale (dernier message de l'agent)
    response_text = result["messages"][-1].content

    # Extraire les outils appelés (messages intermédiaires)
    tools_called = []
    for msg in result["messages"]:
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            for tc in msg.tool_calls:
                tools_called.append(f"{tc['name']}({tc['args']})")

    # Récupérer les sources collectées par les outils
    sources = list(dict.fromkeys(st.session_state.get("last_sources", [])))

    return {
        "response": response_text,
        "sources": sources,
        "tools_called": tools_called,
    }


# =============================================
# Interface Streamlit
# =============================================

st.set_page_config(page_title="Expert AI Act (Agent)", page_icon="🤖", layout="wide")
st.title("Expert AI Act — Mode Agent 🤖")
st.caption(f"Agent LangGraph + Groq — 5 outils")

with st.sidebar:
    st.header("Mode Agent LangGraph")

    # Bouton nouvelle conversation
    if st.button("🔄 Nouvelle conversation", use_container_width=True):
        st.session_state.messages = []
        st.session_state.chat_history = InMemoryChatMessageHistory()
        st.session_state.last_sources = []
        st.rerun()

    st.markdown("---")

    st.markdown(
        "L'agent **décide lui-même** quel outil appeler.\n\n"
        "**5 outils disponibles :**\n"
        "1. `recherche_article` — article par numéro\n"
        "2. `recherche_considerant` — considérant par numéro\n"
        "3. `recherche_annexe` — annexe par numéro romain\n"
        "4. `recherche_ia_act` — recherche sémantique\n"
        "5. `recherche_web` — DuckDuckGo (dernier recours)\n\n"
        "---\n"
        "**Exemples :**\n"
        "- *Donne-moi l'article 5*\n"
        "- *Articles 5 et 8*\n"
        "- *Résume le considérant 12*\n"
        "- *Annexe III*\n"
        "- *Je recrute par IA, suis-je conforme ?*\n\n"
        "---\n"
        "**Différence avec app.py :**\n"
        "- app.py = routage déterministe (regex)\n"
        "- app\\_agent.py = agent LLM (tool calling)"
    )

if not INDEX_DIR.exists():
    st.error("Index FAISS introuvable. Exécutez :\n\n```\npython build_index.py\n```")
    st.stop()

# Historique d'affichage
if "messages" not in st.session_state:
    st.session_state.messages = []

# Entrée utilisateur (en haut, avant l'historique)
if question := st.chat_input("Posez votre question..."):
    # Traitement par l'agent
    with st.spinner("L'agent réfléchit..."):
        result = process_question(question)

    # Sauvegarder dans l'affichage Streamlit (la dernière question + réponse en premier)
    st.session_state.messages.insert(0, {"role": "assistant", "content": result["response"],
                                         "tools_called": result.get("tools_called", []),
                                         "sources": result.get("sources", [])})
    st.session_state.messages.insert(0, {"role": "user", "content": question})

    # Sauvegarder dans la mémoire LLM
    history = get_chat_history()
    history.add_message(HumanMessage(content=question))
    history.add_message(AIMessage(content=result["response"]))

    st.rerun()

# Afficher l'historique (dernier échange en haut)
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        # Afficher les outils appelés (si message assistant)
        if msg["role"] == "assistant" and msg.get("tools_called"):
            with st.expander(f"🔧 Outils appelés ({len(msg['tools_called'])})"):
                for tc in msg["tools_called"]:
                    st.code(tc, language="python")
        # Afficher les sources (si message assistant)
        if msg["role"] == "assistant" and msg.get("sources"):
            with st.expander(f"📚 Sources ({len(msg['sources'])})"):
                for s in msg["sources"]:
                    st.markdown(f"- {s}")
