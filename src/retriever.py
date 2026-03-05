from dotenv import load_dotenv
load_dotenv()

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS

from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor


INDEX_DIR = "faiss_index"


def load_vectorstore():
    embeddings = OpenAIEmbeddings()
    db = FAISS.load_local(
        INDEX_DIR,
        embeddings,
        allow_dangerous_deserialization=True
    )
    return db


def get_base_retriever(k: int = 5, search_type: str = "mmr"):
    """
    search_type:
      - 'similarity' = basic
      - 'mmr' = less duplicate chunks (recommended)
    """
    db = load_vectorstore()
    return db.as_retriever(
        search_type=search_type,
        search_kwargs={"k": k}
    )


def get_advanced_retriever(k: int = 5):
    """
    Advanced retrieval:
      1) MultiQueryRetriever improves recall by generating multiple query variants
      2) Contextual compression reduces irrelevant text from retrieved chunks
    """
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    base = get_base_retriever(k=k, search_type="mmr")

    multi = MultiQueryRetriever.from_llm(
        retriever=base,
        llm=llm
    )

    compressor = LLMChainExtractor.from_llm(llm)

    compressed = ContextualCompressionRetriever(
        base_retriever=multi,
        base_compressor=compressor
    )

    return compressed