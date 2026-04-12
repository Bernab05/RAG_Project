"""
Construit l'index FAISS à partir des chunks du AI Act.
Utilise sentence-transformers (paraphrase-multilingual-mpnet-base-v2).
"""

from pathlib import Path

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

from chunker import parse_ai_act

INDEX_DIR = Path(__file__).parent / "faiss_index"
MODEL_NAME = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"


def build():
    print("1/3 - Parsing du AI Act...")
    chunks = parse_ai_act()
    print(f"      {len(chunks)} chunks créés")

    # Convertir en documents LangChain
    documents = [
        Document(page_content=c["content"], metadata=c["metadata"])
        for c in chunks
    ]

    print(f"2/3 - Chargement du modèle d'embeddings : {MODEL_NAME}")
    embeddings = HuggingFaceEmbeddings(
        model_name=MODEL_NAME,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True, "batch_size": 32},
    )

    print("3/3 - Construction de l'index FAISS...")
    db = FAISS.from_documents(documents, embeddings)
    db.save_local(str(INDEX_DIR))
    print(f"      Index sauvegardé dans {INDEX_DIR}/")


if __name__ == "__main__":
    build()
