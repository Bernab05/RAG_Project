"""
Chatbot expert AI Act v2 — Interface Streamlit + RAG + filtrage metadata.

Deux modes de fonctionnement :
- MODE DIRECT : si la question mentionne un article/chapitre/considérant précis,
  recherche directe dans le docstore (bypass vectoriel) + restitution mot à mot.
- MODE RAG : recherche sémantique avec seuil de score + génération LLM.
"""

import re
from pathlib import Path

import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# --- Configuration ---
INDEX_DIR = Path(__file__).parent / "faiss_index"
MODEL_NAME = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
LLM_MODEL = "gemma3:1b"
SCORE_THRESHOLD = 0.4
TOP_K = 5

# --- Prompt système ---
SYSTEM_PROMPT = """\
Tu es un assistant juridique spécialisé UNIQUEMENT sur le Règlement européen sur \
l'Intelligence Artificielle (AI Act, Règlement UE 2024/1689).

RÈGLES STRICTES — tu DOIS les respecter à chaque réponse :
1. Réponds UNIQUEMENT à partir du contexte ci-dessous. JAMAIS avec tes connaissances.
2. Si le contexte ne contient PAS la réponse, tu DOIS répondre exactement :
   "Cette information ne figure pas dans les extraits de l'AI Act à ma disposition."
3. Ne complète JAMAIS une réponse avec des informations extérieures au contexte.
4. Cite TOUJOURS les articles ou considérants exacts (ex: "Article 6, paragraphe 2").
5. Si la question ne concerne PAS l'AI Act, refuse poliment en disant :
   "Ma compétence se limite au Règlement UE 2024/1689 (AI Act)."
6. Réponds en français, de manière structurée et concise.

Contexte (extraits officiels du Règlement UE 2024/1689) :
{context}
"""

NO_CONTEXT_RESPONSE = (
    "Je n'ai trouvé aucun passage pertinent dans l'AI Act pour répondre à cette question.\n\n"
    "Cela peut signifier que :\n"
    "- La question porte sur un sujet non couvert par le Règlement UE 2024/1689\n"
    "- La formulation est trop éloignée du vocabulaire juridique du texte\n\n"
    "Essayez de reformuler avec des termes du règlement "
    "(ex: *système d'IA à haut risque*, *pratiques interdites*, *obligations de transparence*)."
)

PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_PROMPT),
        ("human", "{question}"),
    ]
)


# =============================================
# Recherche directe dans le docstore (bypass vectoriel)
# =============================================

def docstore_lookup(db, filters: dict) -> list:
    """
    Recherche directe dans le docstore FAISS par metadata.
    Parcourt TOUS les documents (641) et filtre en Python.

    Pourquoi ne pas utiliser db.similarity_search(filter=...) ?
    → FAISS fait du POST-filtrage : recherche vectorielle d'abord (fetch_k résultats),
      puis filtre par metadata. Si le document cherché n'est pas dans les fetch_k plus
      proches sémantiquement de la question, il n'est jamais trouvé.
      Ex: "Que dit le considérant 12?" est sémantiquement loin du contenu du considérant 12.

    Cette fonction fait le contraire : elle filtre d'abord par metadata (exact match),
    puis retourne TOUS les documents qui matchent. Garanti de trouver le résultat.
    """
    results = []
    for doc_id in db.index_to_docstore_id.values():
        doc = db.docstore.search(doc_id)
        match = all(doc.metadata.get(k) == v for k, v in filters.items())
        if match:
            results.append(doc)

    # Trier par numéro de paragraphe pour les articles découpés
    def sort_key(d):
        p = d.metadata.get("paragraph", "")
        return int(p) if p.isdigit() else 0

    results.sort(key=sort_key)
    return results


# =============================================
# Parsing des filtres metadata depuis la question
# =============================================

def parse_metadata_filter(question: str) -> dict | None:
    """
    Détecte si la question mentionne un article, chapitre, section ou considérant.
    Retourne un dict de filtres ou None.
    """
    q = question.lower()
    filters = {}

    # Article
    art_match = re.search(r"article\s+(premier|\d+)", q)
    if art_match:
        art_num = "1" if art_match.group(1) == "premier" else art_match.group(1)
        filters["article"] = art_num

    # Chapitre (chiffres romains)
    chap_match = re.search(r"chapitre\s+([ivxlc]+)", q)
    if chap_match:
        filters["chapter"] = chap_match.group(1).upper()

    # Section
    sec_match = re.search(r"section\s+(\d+)", q)
    if sec_match:
        filters["section"] = sec_match.group(1)

    # Considérant
    cons_match = re.search(r"consid[ée]rant\s+(\d+)", q)
    if cons_match:
        filters["type"] = "considerant"
        filters["numero"] = f"({cons_match.group(1)})"

    # "articles du chapitre X" → force type=article
    if "chapter" in filters and "article" not in filters:
        if re.search(r"articles?\s+(du|de|dans)", q):
            filters["type"] = "article"

    # Paragraphe (si article détecté)
    para_match = re.search(r"paragraphe\s+(\d+)", q)
    if para_match and "article" in filters:
        filters["paragraph"] = para_match.group(1)

    return filters if filters else None


def is_verbatim_request(question: str) -> bool:
    """Détecte si l'utilisateur demande une restitution mot à mot."""
    patterns = [
        r"\b(lis|lire|texte|mot.?[àa].?mot|verbatim|intégral|complet)\b",
        r"\b(donne|montre|affiche|cite|recopie|reprodui)[s-]?\s*(moi|le|la|l'|les)\b",
        r"\bque\s+(dit|stipule|prévoit|dispose|énonce)\b",
        r"\bcontenu\s+(de|du|d')\b",
    ]
    q = question.lower()
    return any(re.search(p, q) for p in patterns)


# =============================================
# Chargement des ressources
# =============================================

@st.cache_resource
def load_vectorstore():
    embeddings = HuggingFaceEmbeddings(
        model_name=MODEL_NAME,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    return FAISS.load_local(
        str(INDEX_DIR), embeddings, allow_dangerous_deserialization=True
    )


@st.cache_resource
def load_llm():
    return Ollama(model=LLM_MODEL, temperature=0.1, timeout=120)


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
            if m.get("paragraph"):
                label += f" (§{m['paragraph']})"
        else:
            label = m.get("title", "Considérant")
        sources.append(label)
    return sources


# =============================================
# Interface Streamlit
# =============================================

st.set_page_config(page_title="Expert AI Act v2", page_icon="EU", layout="wide")
st.title("Expert AI Act v2 (UE 2024/1689)")
st.caption(f"Chatbot RAG + filtrage metadata — FAISS + {LLM_MODEL} + LangChain")

with st.sidebar:
    st.header("Modes de fonctionnement")
    st.markdown(
        "**Mode direct** : mentionnez un article, chapitre ou considérant "
        "pour obtenir le texte officiel mot à mot.\n\n"
        "**Mode RAG** : posez une question libre pour une réponse "
        "générée à partir des passages pertinents.\n\n"
        "---\n"
        "Exemples :\n"
        "- *Donne-moi l'article 5*\n"
        "- *Que dit le considérant 12 ?*\n"
        "- *Articles du chapitre III*\n"
        "- *Quelles pratiques sont interdites ?*"
    )

if not INDEX_DIR.exists():
    st.error("Index FAISS introuvable. Lancez d'abord :\n\n```\npython build_index.py\n```")
    st.stop()

db = load_vectorstore()
llm = load_llm()

retriever = db.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={"k": TOP_K, "score_threshold": SCORE_THRESHOLD},
)

chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | PROMPT
    | llm
    | StrOutputParser()
)

# Historique
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander("Sources"):
                for s in msg["sources"]:
                    st.markdown(f"- {s}")

if question := st.chat_input("Posez votre question sur l'AI Act..."):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Recherche dans l'AI Act..."):

            meta_filter = parse_metadata_filter(question)
            verbatim = is_verbatim_request(question)

            if meta_filter:
                # === MODE DIRECT : recherche dans le docstore (bypass vectoriel) ===
                docs = docstore_lookup(db, meta_filter)
                sources = get_sources(docs)

                if not docs:
                    response = (
                        f"Aucun document trouvé pour le filtre : `{meta_filter}`.\n\n"
                        "Vérifiez le numéro d'article, de chapitre ou de considérant."
                    )
                elif verbatim or len(docs) <= 5:
                    # Restitution mot à mot — tous les paragraphes dans l'ordre
                    response = "\n\n---\n\n".join(doc.page_content for doc in docs)
                else:
                    # Beaucoup de résultats → résumé par le LLM
                    context = format_docs(docs[:TOP_K])
                    response = llm.invoke(
                        PROMPT.format(context=context, question=question)
                    )

            else:
                # === MODE RAG : recherche sémantique ===
                docs = retriever.invoke(question)
                sources = get_sources(docs)

                if not docs:
                    response = NO_CONTEXT_RESPONSE
                else:
                    response = chain.invoke(question)

        # --- Affichage ---
        st.markdown(response)

        if meta_filter:
            st.info(f"Mode direct — Filtre : `{meta_filter}` — {len(docs)} document(s)")
        else:
            st.caption(f"Mode RAG — {len(docs)} document(s) pertinent(s)")

        if sources:
            with st.expander(f"Sources ({len(sources)})"):
                for s in sources:
                    st.markdown(f"- {s}")

    st.session_state.messages.append(
        {"role": "assistant", "content": response, "sources": sources}
    )
