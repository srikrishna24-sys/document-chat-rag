from typing import List
from dotenv import load_dotenv
load_dotenv()

from langchain_openai import ChatOpenAI
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from retriever import get_advanced_retriever


def format_docs(docs: List[Document]) -> str:
    parts = []
    for i, d in enumerate(docs, start=1):
        src = d.metadata.get("source", "unknown")
        page = d.metadata.get("page", None)
        loc = f"{src}" + (f" (page {page})" if page is not None else "")
        parts.append(f"[{i}] {loc}\n{d.page_content}")
    return "\n\n".join(parts)


def build_chain():
    retriever = get_advanced_retriever(k=5)

    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are a careful assistant. Answer ONLY from the SOURCES. "
         "If you cannot find the answer in the sources, say 'I don't know'. "
         "Cite sources like [1], [2]."),
        ("human",
         "QUESTION:\n{question}\n\nSOURCES:\n{context}")
    ])

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain


def main():
    chain = build_chain()

    while True:
        q = input("\nAsk (or type 'exit'): ").strip()
        if q.lower() == "exit":
            break

        ans = chain.invoke(q)
        print("\n" + ans)


if __name__ == "__main__":
    main()