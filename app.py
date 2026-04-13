"""
Chatbot Expert AI Act — Routage deterministe + LLM pour la redaction.

Supporte 2 environnements :
- LOCAL : Ollama (qwen2.5:3b) — pas de cle API, tout en local
- CLOUD : HuggingFace Inference API — pour Streamlit Cloud (cle HF_TOKEN requise)

La detection est automatique : si Ollama repond sur localhost, on l'utilise.
Sinon, on bascule sur HuggingFace.
"""

import os
import re
from pathlib import Path

import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_community.tools import DuckDuckGoSearchRun

# =============================================
# Configuration
# =============================================
INDEX_DIR       = Path(__file__).parent / "faiss_index"
MODEL_NAME      = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
SCORE_THRESHOLD = 0.35
TOP_K           = 8

# Modele Groq pour le cloud (gratuit, rapide, Llama 3)
GROQ_MODEL      = "llama-3.1-8b-instant"
# Modele Ollama pour le local
OLLAMA_MODEL    = "qwen2.5:3b"

# =============================================
# Prompt unique : redaction de la reponse
# =============================================
RESPONSE_PROMPT = """\
Tu es un assistant expert du Reglement europeen sur l'Intelligence Artificielle \
(AI Act, Reglement UE 2024/1689). Reponds en francais.

REGLES :
1. Base ta reponse sur le contexte fourni ET sur l'historique de conversation.
2. Quand le contexte vient du AI Act, cite les articles et considerants exacts \
   (ex: "Article 6, paragraphe 2").
3. Si le contexte contient des obligations ou interdictions, LISTE-LES precisement.
4. Si le contexte vient d'internet, PRECISE-LE clairement et reponds normalement \
   sans forcer de lien avec le AI Act.
5. Ne dis JAMAIS "consultez le texte complet" ou "je n'ai pas assez d'info".
   Utilise ce que tu as et reponds du mieux possible.
6. Structure ta reponse avec des titres et des puces si necessaire.
7. Tu as acces a l'historique de conversation. Si l'utilisateur fait reference \
   a un echange precedent (un prenom, un sujet, une personne), utilise l'historique \
   pour repondre. Ne dis JAMAIS "je ne sais pas" si l'info est dans l'historique.
8. Si l'utilisateur demande un RESUME ou une EXPLICATION d'un article ou considerant, \
   fournis un resume synthetique, pas le texte integral.
9. Pour les questions sans rapport avec le AI Act (blagues, culture generale, etc.), \
   reponds normalement sans forcer de lien avec le reglement.
10. Si on te demande de repondre avec tes propres connaissances et que tu ne connais \
    PAS la reponse ou que tes informations sont trop vagues/incertaines, reponds \
    EXACTEMENT avec le marqueur [RECHERCHE_WEB] sur la premiere ligne, suivi d'une \
    breve explication de ce que tu cherches. Exemple :
    [RECHERCHE_WEB]
    Je n'ai pas d'information fiable sur ce sujet.
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

def is_ollama_available() -> bool:
    """Verifie si Ollama tourne sur localhost."""
    try:
        import requests
        r = requests.get("http://localhost:11434/api/tags", timeout=2)
        return r.status_code == 200
    except Exception:
        return False


@st.cache_resource
def load_llm():
    """
    Charge le LLM automatiquement :
    - Si Ollama est disponible (local) → ChatOllama (LangChain)
    - Sinon (Streamlit Cloud) → InferenceClient (huggingface_hub natif, sans wrapper LangChain)
    Retourne un dict {"type": "ollama"|"hf", "client": ...}
    """
    if is_ollama_available():
        from langchain_ollama import ChatOllama
        client = ChatOllama(model=OLLAMA_MODEL, temperature=0.1)
        return {"type": "ollama", "client": client}, f"Ollama ({OLLAMA_MODEL})"
    else:
        from langchain_groq import ChatGroq
        groq_api_key = (
            os.environ.get("GROQ_API_KEY")
            or st.secrets.get("GROQ_API_KEY", "")
        )
        if not groq_api_key:
            st.error(
                "Cle Groq requise pour le mode cloud.\n\n"
                "1. Creez un compte gratuit sur https://console.groq.com\n"
                "2. Generez une cle API\n"
                "3. Ajoutez dans Streamlit Cloud → Settings → Secrets :\n"
                "   `GROQ_API_KEY = \"gsk_...\"`"
            )
            st.stop()
        client = ChatGroq(model=GROQ_MODEL, api_key=groq_api_key, temperature=0.1)
        return {"type": "groq", "client": client}, f"Groq ({GROQ_MODEL})"

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
        elif m.get("type") == "annexe":
            label = f"Annexe {m['annexe']} : {m['title']}"
        else:
            label = m.get("title", "Considerant")
        sources.append(label)
    return sources

# =============================================
# Initialisation
# =============================================

db = load_vectorstore()
LLM_INFO, LLM_LABEL = load_llm()
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



def call_llm(question: str, context: str, context_label: str = "AI Act") -> str:
    """
    Appel LLM avec memoire SYSTEMATIQUE :
    L'historique (6 derniers messages) est TOUJOURS envoye au LLM.
    Le LLM est assez intelligent pour ignorer l'historique quand il n'est pas pertinent.

    ChatOllama (local) et ChatGroq (cloud) utilisent tous les deux
    l'interface LangChain standard → meme code, zero branchement.
    """
    messages = [SystemMessage(content=RESPONSE_PROMPT)]

    # Toujours inclure l'historique (le LLM saura quoi en faire)
    history = get_chat_history()
    if history.messages:
        for msg in history.messages[-6:]:
            content = msg.content[:800] + "..." if len(msg.content) > 800 else msg.content
            messages.append(type(msg)(content=content))

    messages.append(HumanMessage(
        content=f"Contexte ({context_label}) :\n{context}\n\nQuestion : {question}"
    ))

    try:
        return LLM_INFO["client"].invoke(messages).content
    except Exception as e:
        error_msg = str(e)
        if "401" in error_msg or "authentication" in error_msg.lower() or "invalid_api_key" in error_msg.lower():
            raise RuntimeError(
                f"Erreur 401 : cle GROQ_API_KEY invalide.\n"
                f"Verifiez le secret dans Streamlit Cloud → Settings → Secrets.\n"
                f"Detail : {error_msg}"
            ) from e
        elif "429" in error_msg or "rate" in error_msg.lower():
            raise RuntimeError(
                f"Erreur 429 : quota Groq atteint. Reessayez dans quelques secondes.\n"
                f"Detail : {error_msg}"
            ) from e
        else:
            raise RuntimeError(f"Erreur LLM : {error_msg}") from e


def _direct_or_llm(docs, question: str, wants_analysis: bool, label: str) -> dict:
    """
    Si l'utilisateur veut juste le texte (ex: "article 5") → retour direct.
    Si l'utilisateur veut un resume/explication → passer le texte au LLM.
    """
    sources = get_sources(docs)
    context = format_docs(docs)
    if wants_analysis:
        if len(context) > 4000:
            context = context[:4000] + "\n\n[... tronque ...]"
        response_text = call_llm(question, context, f"AI Act — {label}")
        return {
            "response": response_text,
            "sources": sources,
            "mode": f"{label} (analyse LLM)",
        }
    return {
        "response": context,
        "sources": sources,
        "mode": f"{label} (texte integral)",
    }


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
    # Detection : l'utilisateur veut-il un RESUME/EXPLICATION ?
    # Si oui, on passe par le LLM meme pour un article/considerant precis.
    # ========================================
    wants_analysis = bool(re.search(
        r"(r[eé]sum|synth[eé]|explique|compar|analys|simplifie|vulgar|tradui|"
        r"d[eé]taille|d[eé]veloppe|pr[eé]cise|reformule|"
        r"en quoi|que signifie|que veut dire|qu.?est.ce que|comment)",
        q
    ))

    # ========================================
    # MODE 1 : DIRECT — regex article/considerant/annexe (0 appel LLM)
    # Supporte : "article 5", "articles 5 et 8", "articles 5 à 8",
    #            "considérant 12", "considérants 1 et 2",
    #            "annexe III", "annexes I et IV"
    # Si wants_analysis=True, on passe le texte au LLM pour synthese.
    # ========================================

    # --- Articles : plage (5 à 8) ---
    range_match = re.search(r"articles?\s+(\d+)\s+[àa]\s+(\d+)", q)
    if range_match:
        start, end = int(range_match.group(1)), int(range_match.group(2))
        all_docs = []
        nums = []
        for n in range(start, min(end + 1, start + 20)):  # max 20 articles
            docs = docstore_lookup(db, {"article": str(n)})
            if docs:
                all_docs.extend(docs)
                nums.append(str(n))
        if all_docs:
            return _direct_or_llm(all_docs, question, wants_analysis,
                                  f"Articles {', '.join(nums)}")

    # --- Articles : un ou plusieurs (5, "5 et 8", "5, 8 et 12") ---
    # Etape 1 : capturer le bloc "article(s) 5 et 8" ou "art. 5, 8 et 12"
    art_block = re.search(
        r"(?:articles?|art\.?)\s+((?:premier|\d+)(?:\s*(?:,|et)\s*(?:premier|\d+))*)",
        q
    )
    art_matches = re.findall(r"(premier|\d+)", art_block.group(1)) if art_block else []
    if art_matches:
        all_docs = []
        nums = []
        for m in art_matches:
            num = "1" if m == "premier" else m
            if num not in nums:  # eviter les doublons
                nums.append(num)
                docs = docstore_lookup(db, {"article": num})
                all_docs.extend(docs)
        if all_docs:
            return _direct_or_llm(all_docs, question, wants_analysis,
                                  f"Article{'s' if len(nums) > 1 else ''} {', '.join(nums)}")

    # --- Considerants : un ou plusieurs ("considérants 1 et 2", "considérants 1, 2 et 3") ---
    cons_block = re.search(
        r"consid[ée]rants?\s+((?:\d+)(?:\s*(?:,|et)\s*\d+)*)",
        q
    )
    cons_matches = re.findall(r"(\d+)", cons_block.group(1)) if cons_block else []
    if cons_matches:
        all_docs = []
        for num in cons_matches:
            docs = docstore_lookup(db, {"type": "considerant", "numero": f"({num})"})
            all_docs.extend(docs)
        if all_docs:
            return _direct_or_llm(all_docs, question, wants_analysis,
                                  f"Considerant{'s' if len(cons_matches) > 1 else ''} {', '.join(cons_matches)}")

    # --- Annexes : une ou plusieurs (annexe III, annexes I et IV) ---
    annexe_matches = re.findall(r"annexe\s+([IVXLC]+|\d+)", q, re.IGNORECASE)
    if annexe_matches:
        all_docs = []
        nums = []
        for num in annexe_matches:
            num_upper = num.upper()
            if num_upper not in nums:
                nums.append(num_upper)
                docs = docstore_lookup(db, {"type": "annexe", "annexe": num_upper})
                all_docs.extend(docs)
        if all_docs:
            return _direct_or_llm(all_docs, question, wants_analysis,
                                  f"Annexe{'s' if len(nums) > 1 else ''} {', '.join(nums)}")

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
    # MODE 2bis : CONVERSATION — question de suivi ou courte reference
    # Si l'historique existe et que la question semble etre un suivi,
    # on repond avec le LLM + historique SANS chercher sur internet.
    # Evite que "etait-il un peintre aussi?" cherche sur DuckDuckGo
    # et retourne un resultat sans rapport.
    # ========================================
    history = get_chat_history()
    if history.messages:
        is_short = len(question.split()) < 15
        has_pronoun = bool(re.search(
            r"\b(il|elle|ils|elles|lui|son|sa|ses|leur|ce|cet|cette|ces|"
            r"le meme|la meme|aussi|[eé]galement|en plus|de plus|"
            r"je m.appelle|mon nom|comment je|qui suis|quel |sur quel|"
            r"a.?t.?il|a.?t.?elle|est.?il|est.?elle|[eé]tait.?il|[eé]tait.?elle|"
            r"le pr[eé]c[eé]dent|ci.?dessus|plus haut|ta r[eé]ponse)\b",
            q
        ))
        # Verbe d'action en debut de phrase courte = reference implicite
        # "resume le", "explique ca", "detaille", "continue"
        action_followup = bool(re.match(
            r"^(r[eé]sum|synth[eé]|explique|d[eé]taille|d[eé]veloppe|pr[eé]cise|"
            r"reformule|tradui|simplifie|continue|poursui|compl[eè]te|approfondi|"
            r"r[eé]p[eè]te|redis|relis)",
            q
        ))
        if is_short and (has_pronoun or action_followup):
            response_text = call_llm(
                question,
                "Pas de contexte supplementaire. Reponds en utilisant l'historique de conversation.",
                "Conversation (historique)",
            )
            return {
                "response": response_text,
                "sources": [],
                "mode": "Conversation (memoire)",
            }

    # ========================================
    # MODE 3 : LLM SEUL — connaissances propres du modele
    # Le LLM repond avec ses connaissances. S'il ne sait pas,
    # il repond [RECHERCHE_WEB] → on declenche DuckDuckGo.
    # ========================================
    response_text = call_llm(
        question,
        "Aucun document pertinent dans la base AI Act. "
        "Reponds avec tes propres connaissances. "
        "Si tu ne connais PAS la reponse ou que tu n'es pas sur, "
        "reponds EXACTEMENT [RECHERCHE_WEB] sur la premiere ligne.",
        "Connaissances du modele",
    )

    # Si le LLM sait repondre → on retourne sa reponse
    if "[RECHERCHE_WEB]" not in response_text:
        return {
            "response": response_text,
            "sources": ["Connaissances du modele"],
            "mode": "LLM (connaissances propres)",
        }

    # ========================================
    # MODE 4 : WEB — DuckDuckGo + LLM (1 appel LLM)
    # Declenche UNIQUEMENT si le LLM a dit [RECHERCHE_WEB]
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
        "response": "Je n'ai trouve aucune information pertinente, ni dans le AI Act, "
                     "ni dans mes connaissances, ni sur internet.",
        "sources": [],
        "mode": "Aucun resultat",
    }

# =============================================
# Interface Streamlit
# =============================================

st.set_page_config(page_title="Expert AI Act", page_icon="EU", layout="wide")
st.title("Expert AI Act (UE 2024/1689)")
st.caption(f"RAG + DuckDuckGo — {LLM_LABEL}")

with st.sidebar:
    st.header("4 modes automatiques")
    st.markdown(
        "**1. Direct** : article, considerant ou annexe mot a mot\n\n"
        "**2. RAG** : recherche semantique dans le AI Act + reponse IA\n\n"
        "**3. LLM** : connaissances propres du modele\n\n"
        "**4. Web** : recherche internet (si le LLM ne sait pas)\n\n"
        "---\n"
        "Exemples :\n"
        "- *Donne-moi l'article 5*\n"
        "- *Articles 5 et 8*\n"
        "- *Resume le considerant 12*\n"
        "- *Annexe III*\n"
        "- *Je recrute par IA, suis-je conforme ?*\n"
        "- *Qui est Emmanuel Macron ?*\n"
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
            try:
                result = process_question(question)
            except RuntimeError as e:
                st.error(str(e))
                st.stop()

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
