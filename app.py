"""
Chatbot Expert AI Act — Routage deterministe + LLM pour la redaction.

Le routage est fait par du code Python (regex + score FAISS), pas par le LLM.
Le LLM sert UNIQUEMENT a rediger la reponse a partir du contexte recupere.

3 modes automatiques :
- DIRECT : regex detecte article/considerant → texte mot a mot (0 appel LLM)
- RAG : FAISS trouve des passages pertinents → LLM redige (1 appel LLM)
- WEB : FAISS ne trouve rien → DuckDuckGo + LLM redige (1 appel LLM)
"""

import re
from pathlib import Path

import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_community.tools import DuckDuckGoSearchRun

# =============================================
# Configuration
# =============================================
INDEX_DIR       = Path(__file__).parent / "faiss_index"
MODEL_NAME      = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
LLM_MODEL       = "qwen2.5:3b"
SCORE_THRESHOLD = 0.35
TOP_K           = 8

# =============================================
# Prompt unique : redaction de la reponse
# =============================================
RESPONSE_PROMPT = """\
Tu es un expert du Reglement europeen sur l'Intelligence Artificielle \
(AI Act, Reglement UE 2024/1689). Reponds en francais.

REGLES :
1. Base ta reponse UNIQUEMENT sur le contexte fourni ci-dessous.
2. Cite les articles et considerants exacts (ex: "Article 6, paragraphe 2").
3. Si le contexte contient des obligations ou interdictions, LISTE-LES precisement.
4. Si le contexte vient d'internet, PRECISE-LE clairement.
5. Ne dis JAMAIS "consultez le texte complet" ou "je n'ai pas assez d'info".
   Utilise ce que tu as et reponds du mieux possible.
6. Structure ta reponse avec des titres et des puces si necessaire.
"""

# =============================================
# Chargement ressources
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
    return ChatOllama(model=LLM_MODEL, temperature=0.1)

# =============================================
# Fonctions de base
# =============================================

def docstore_lookup(db, filters):
    results = []
    for doc_id in db.index_to_docstore_id.values():
        doc = db.docstore.search(doc_id)
        if all(doc.metadata.get(k) == v for k, v in filters.items()):
            results.append(doc)
    results.sort(key=lambda d: int(d.metadata.get("paragraph", "0") or "0"))
    return results

def format_docs(docs):
    return "\n\n---\n\n".join(doc.page_content for doc in docs)

def get_sources(docs):
    sources = []
    for doc in docs:
        m = doc.metadata
        if m.get("type") == "article":
            label = f"Article {m['article']} : {m['title']}"
            if m.get("chapter"):
                label = f"Chapitre {m['chapter']} > {label}"
        else:
            label = m.get("title", "Considerant")
        sources.append(label)
    return sources

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
# Routage deterministe + execution
# =============================================

def get_chat_history() -> InMemoryChatMessageHistory:
    """
    Retourne l'objet InMemoryChatMessageHistory stocke dans la session Streamlit.
    InMemoryChatMessageHistory est la classe officielle LangChain (non deprecated).
    """
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = InMemoryChatMessageHistory()
    return st.session_state.chat_history


def is_followup(question: str) -> bool:
    """
    Detecte si la question fait reference a la conversation precedente.
    Combine 2 signaux :
    1. Mots-cles de reference explicite (ci-dessus, precedent, ta reponse)
    2. Verbes d'action sur contenu implicite (resume, explique, continue)

    N'utilise PAS la longueur de la question (trop de faux positifs).
    """
    q = question.lower().strip()

    # Signal 1 : references explicites au contexte precedent
    explicit = [
        r"\bci.?dessus\b", r"\bprecedent\b", r"\bplus haut\b",
        r"\bta reponse\b", r"\bton analyse\b", r"\bce que tu\b",
        r"\bces articles\b", r"\bces resultats\b", r"\bces points\b",
        r"\bcette liste\b", r"\bce tableau\b", r"\bce texte\b",
    ]
    if any(re.search(p, q) for p in explicit):
        return True

    # Signal 2 : verbe d'action EN DEBUT de phrase sur contenu implicite
    action_start = [
        r"^(resume|syntheti|explique|detaille|developpe|precise|reformule|tradui)",
        r"^(continue|poursui|complete|approfondi)",
        r"^(et\s+(pour|qu|si|le|la|les|en)\b)",
    ]
    if any(re.search(p, q) for p in action_start):
        return True

    return False


def call_llm(question: str, context: str, context_label: str = "AI Act") -> str:
    """
    Appel LLM avec memoire CONDITIONNELLE :
    - Si la question est un suivi (ci-dessus, resume, etc.) → historique inclus
    - Sinon → juste le contexte (pas de pollution par l'historique)
    """
    messages = [SystemMessage(content=RESPONSE_PROMPT)]

    # Inclure l'historique SEULEMENT si c'est une question de suivi
    if is_followup(question):
        history = get_chat_history()
        recent = history.messages[-6:]
        for msg in recent:
            if len(msg.content) > 800:
                messages.append(type(msg)(content=msg.content[:800] + "..."))
            else:
                messages.append(msg)

    # Contexte + question
    messages.append(HumanMessage(
        content=f"Contexte ({context_label}) :\n{context}\n\nQuestion : {question}"
    ))

    return llm.invoke(messages).content


def process_question(question: str) -> dict:
    """
    Decide du traitement par du CODE PYTHON (pas par le LLM).

    Ordre :
    1. Regex detecte un article/considerant → restitution directe (0 LLM)
    2. FAISS trouve des docs pertinents → LLM redige avec contexte + historique (1 LLM)
    3. Rien trouve → DuckDuckGo + LLM avec historique (1 LLM)
    """
    q = question.lower()

    # ========================================
    # MODE 1 : DIRECT — regex article/considerant (0 appel LLM)
    # ========================================
    art = re.search(r"article\s+(premier|\d+)", q)
    if art:
        num = "1" if art.group(1) == "premier" else art.group(1)
        docs = docstore_lookup(db, {"article": num})
        if docs:
            return {
                "response": format_docs(docs),
                "sources": get_sources(docs),
                "mode": f"Article {num} (texte integral)",
            }

    cons = re.search(r"consid[ée]rant\s+(\d+)", q)
    if cons:
        docs = docstore_lookup(db, {"type": "considerant", "numero": f"({cons.group(1)})"})
        if docs:
            return {
                "response": docs[0].page_content,
                "sources": get_sources(docs),
                "mode": f"Considerant {cons.group(1)} (texte integral)",
            }

    # ========================================
    # MODE 2 : RAG — recherche semantique FAISS + LLM avec historique
    # ========================================
    docs = retriever.invoke(question)

    if docs:
        sources = get_sources(docs)
        context = format_docs(docs)
        source_list = "\n".join(f"- {s}" for s in sources)

        if len(context) > 4000:
            context = context[:4000] + "\n\n[... tronque ...]"

        full_context = f"Sources AI Act :\n{source_list}\n\nExtraits officiels :\n{context}"
        response_text = call_llm(question, full_context, "AI Act")
        return {
            "response": response_text,
            "sources": sources,
            "mode": f"RAG ({len(docs)} documents)",
        }

    # ========================================
    # MODE 3 : WEB — DuckDuckGo + LLM (1 appel LLM)
    # On fait 2 recherches (FR + EN) pour maximiser les resultats
    # ========================================
    search = DuckDuckGoSearchRun()
    web_parts = []
    try:
        r_fr = search.invoke(question)
        if r_fr:
            web_parts.append(r_fr)
    except Exception:
        pass
    try:
        # Recherche en anglais (souvent meilleurs resultats)
        r_en = search.invoke(question + " results 2025")
        if r_en:
            web_parts.append(r_en)
    except Exception:
        pass
    web_results = "\n\n".join(web_parts)

    if web_results and len(web_results) > 50:
        response_text = call_llm(
            question,
            f"INFORMATION D'INTERNET (pas du AI Act) :\n\n{web_results[:2500]}",
            "Internet (DuckDuckGo)",
        )
        return {
            "response": response_text,
            "sources": ["Recherche internet (DuckDuckGo)"],
            "mode": "Web (DuckDuckGo)",
        }

    # ========================================
    # FALLBACK : rien trouve nulle part
    # ========================================
    return {
        "response": "Je n'ai trouve aucune information pertinente, ni dans le AI Act ni sur internet.",
        "sources": [],
        "mode": "Aucun resultat",
    }

# =============================================
# Interface Streamlit
# =============================================

st.set_page_config(page_title="Expert AI Act", page_icon="EU", layout="wide")
st.title("Expert AI Act (UE 2024/1689)")
st.caption(f"RAG + DuckDuckGo — {LLM_MODEL}")

with st.sidebar:
    st.header("3 modes automatiques")
    st.markdown(
        "**Direct** : article ou considerant mot a mot\n\n"
        "**RAG** : recherche semantique + reponse IA\n\n"
        "**Web** : recherche internet si hors AI Act\n\n"
        "---\n"
        "Exemples :\n"
        "- *Donne-moi l'article 5*\n"
        "- *Que dit le considerant 12 ?*\n"
        "- *Je recrute par IA, suis-je conforme ?*\n"
        "- *Qui a gagne Paris-Roubaix ?*"
    )

if not INDEX_DIR.exists():
    st.error("Index FAISS introuvable.\n\n```\npython build_index.py\n```")
    st.stop()

# st.session_state.messages = affichage Streamlit (dicts role/content)
# st.session_state.chat_history = memoire LLM (objets HumanMessage/AIMessage)
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if question := st.chat_input("Posez votre question..."):
    # Ajouter a l'affichage Streamlit
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Recherche..."):
            result = process_question(question)

        st.markdown(result["response"])
        st.info(f"Mode : {result['mode']}")

        if result["sources"]:
            with st.expander(f"Sources ({len(result['sources'])})"):
                for s in result["sources"]:
                    st.markdown(f"- {s}")

    # Sauvegarder dans l'affichage Streamlit
    st.session_state.messages.append({"role": "assistant", "content": result["response"]})

    # Sauvegarder dans la memoire LLM (objets natifs LangChain)
    history = get_chat_history()
    history.add_message(HumanMessage(content=question))
    history.add_message(AIMessage(content=result["response"]))
