"""
Chatbot expert AI Act — Interface Streamlit + RAG (FAISS + Gemma 3 1B via Ollama).
Avec seuil de pertinence et garde-fou anti-hallucinations.
"""

from pathlib import Path

import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

INDEX_DIR = Path(__file__).parent / "faiss_index"
MODEL_NAME = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
LLM_MODEL = "gemma3:1b"
SCORE_THRESHOLD = 0.4   # Seuil minimum de similarité (0-1). Augmenter = plus strict.
TOP_K = 5               # Nombre max de chunks récupérés

# --- Prompt système durci ---
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
   "Ma compétence se limite au Règlement UE 2024/1689 (AI Act). Je ne peux pas répondre à cette question."
6. Réponds en français, de manière structurée et concise.

Contexte (extraits officiels du Règlement UE 2024/1689) :
{context}
"""

NO_CONTEXT_RESPONSE = (
    "Je n'ai trouvé aucun passage pertinent dans l'AI Act pour répondre à cette question.\n\n"
    "Cela peut signifier que :\n"
    "- La question porte sur un sujet non couvert par le Règlement UE 2024/1689\n"
    "- La formulation de la question est trop éloignée du vocabulaire juridique du texte\n\n"
    "Essayez de reformuler votre question en utilisant des termes du règlement "
    "(ex: *système d'IA à haut risque*, *pratiques interdites*, *obligations de transparence*)."
)

PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_PROMPT),
        ("human", "{question}"),
    ]
)


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
    """Concatène les documents récupérés en un seul bloc de contexte."""
    return "\n\n---\n\n".join(doc.page_content for doc in docs)


def get_sources(docs):
    """Extrait une liste lisible des sources (pour affichage)."""
    sources = []
    for doc in docs:
        m = doc.metadata
        if m.get("type") == "article":
            label = f"Article {m['article']} : {m['title']}"
            if m.get("chapter"):
                label = f"Chapitre {m['chapter']} > {label}"
        else:
            label = m.get("title", "Considérant")
        sources.append(label)
    return sources


# --- Interface Streamlit ---
st.set_page_config(page_title="Expert AI Act", page_icon="EU", layout="wide")
st.title("Expert AI Act (UE 2024/1689)")
st.caption(f"Chatbot RAG — FAISS + {LLM_MODEL} + LangChain")

if not INDEX_DIR.exists():
    st.error(
        "Index FAISS introuvable. Lancez d'abord :\n\n"
        "```\npython build_index.py\n```"
    )
    st.stop()

db = load_vectorstore()
llm = load_llm()

# Retriever avec seuil de score : ne retourne QUE les documents au-dessus du seuil
retriever = db.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={"k": TOP_K, "score_threshold": SCORE_THRESHOLD},
)

# Chaîne RAG (utilisée seulement quand le retriever a trouvé des documents pertinents)
chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | PROMPT
    | llm
    | StrOutputParser()
)

# Historique de conversation
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
            # Garde-fou : vérifier si le retriever a trouvé des documents pertinents
            docs = retriever.invoke(question)
            sources = get_sources(docs)

            if not docs:
                # Aucun document au-dessus du seuil → refus sans appeler le LLM
                response = NO_CONTEXT_RESPONSE
            else:
                # Documents pertinents trouvés → générer la réponse via le LLM
                response = chain.invoke(question)

        st.markdown(response)
        if sources:
            with st.expander(f"Sources ({len(sources)} documents)"):
                for s in sources:
                    st.markdown(f"- {s}")

    st.session_state.messages.append(
        {"role": "assistant", "content": response, "sources": sources}
    )
