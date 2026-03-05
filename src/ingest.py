from pathlib import Path
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

load_dotenv()

DATA_PATH = Path("data")
INDEX_PATH = Path("faiss_index")


def load_documents():
    docs = []

    for file in DATA_PATH.glob("*"):
        if file.suffix == ".pdf":
            loader = PyPDFLoader(str(file))
            docs.extend(loader.load())

        elif file.suffix == ".txt":
            loader = TextLoader(str(file))
            docs.extend(loader.load())

    return docs


def main():

    docs = load_documents()

    print(f"Loaded {len(docs)} documents")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    # chunk_size

    # How much text per chunk (roughly characters).

    # Too small → lacks context, answers become shallow

    # Too big → retrieval becomes noisy, expensive, worse accuracy

    # Good starting points

    # PDFs / dense docs: 800–1200

    # Code / manuals: 1200–1800

    # Short notes: 500–900

    # chunk_overlap

    # Repeats some text between chunks to preserve continuity.

    # Too low → answers miss context across chunk boundaries

    # Too high → duplicates in retrieval

    # Good starting points

    # 100–200 (for chunk size 800–1200)

    # Rule of thumb: overlap ≈ 10–20% of chunk_size.

    chunks = splitter.split_documents(docs)

    print(f"Created {len(chunks)} chunks")

    embeddings = OpenAIEmbeddings()

    vectorstore = FAISS.from_documents(chunks, embeddings)

    vectorstore.save_local("faiss_index")

    print("Vector index created successfully")

##documents
#    ↓
# split into chunks
#    ↓
# create embeddings
#    ↓
# store vectors in FAISS

if __name__ == "__main__":
    main()