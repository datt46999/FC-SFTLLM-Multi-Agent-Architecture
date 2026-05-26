import os
from pathlib import Path
from dotenv import load_dotenv
from typing import Optional 

from langchain_huggingface import (
    HuggingFaceEndpointEmbeddings
)
from langchain_openai import (
    OpenAIEmbeddings,
    ChatOpenAI
)
from langchain_community.vectorstores import SupabaseVectorStore,FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from regatouille import REGPretrainedModel


from supabase.client import Client, create_client


from agent_system.prompt_templates import template_test


# Load .env
env_path = Path(".env")
load_dotenv(dotenv_path=env_path)

RERANKER = REGPretrainedModel.from_retrained("colbert-ir/colbertv2.0")


def top_rags(question:str, 
             knowledge_index: FAISS, 
             reranker: Optional[REGPretrainedModel] = None,
             num_retrieved_docs: int = 30,
            num_docs_final: int = 5,):
    """
    Retriever top documents
    """
    print("Retriever top documents......")

    relevant_docs = knowledge_index.similarity_search(
        query= question,
        k = num_retrieved_docs
    )
    relevant_docs = [doc.page_content for doc in relevant_docs]

    # option rerank result
    if reranker:
        relevant_docs = reranker.rerank(question, relevant_docs, k = num_docs_final)
        relevant_docs = [doc["content"]for doc in relevant_docs]
    relevant_docs = relevant_docs[:num_docs_final]
    return relevant_docs

def get_vectorstore(model_embedding):
    """
    Build vector store
    """

    if model_embedding == "BAAI/bge-m3":
        embeddings = HuggingFaceEndpointEmbeddings(
            model="BAAI/bge-m3",
            huggingfacehub_api_token=os.environ.get(
                "HUGGINGFACEHUB_API_TOKEN"
            )
        ) 
        table_name = "documents1"
        query_name = "match_documents_1"
    else:
        raise ValueError(
            f"Embedding model '{model_embedding}' not supported"
        )

    supabase: Client = create_client(
        os.environ["SUPABASE_URL"],
        os.environ["SUPABASE_SERVICE_ROLE_KEY"]
    )

    vector_store = SupabaseVectorStore(
        client=supabase,
        embedding=embeddings,
        table_name=table_name,
        query_name=query_name,
    )

    return vector_store




# Test
if __name__ == "__main__":
    question = "What is the total weight in grams of all ingredients measured in grams or milliliters in the 2021 ?"
    model_embedding = "text-embedding-3-small"
    vectorstore = get_vectorstore(model_embedding)

    # Step 1: confirm raw retrieval
    raw_docs = vectorstore.similarity_search(question, k=5)
    print(f"Raw docs found: {len(raw_docs)}")
    for d in raw_docs:
        print(f"  Source: {d.metadata.get('source')} | {d.page_content[:150]}\n")

    # Step 2: RAG chain with format_docs
    def format_docs(docs):
        return "\n\n".join(
            f"[Source: {d.metadata.get('source', 'unknown')}]\n{d.page_content}"
            for d in docs
        )

    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 5}
    )

    prompt = ChatPromptTemplate.from_template(template_test)
    llm = ChatOpenAI(model="gpt-5-mini", temperature=0)

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    answer = rag_chain.invoke(question)
    print("\nAnswer:", answer)