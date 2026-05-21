import json
import os
import time
from supabase.client import Client, create_client

from langchain_community.document_loaders import DirectoryLoader, UnstructuredFileLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from langchain_experimental.text_splitter import SemanticChunker          
from langchain_openai import OpenAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpointEmbeddings              
from langchain_community.vectorstores import SupabaseVectorStore

from pathlib import Path
from dotenv import load_dotenv

env_path = Path("") / ".env"
load_dotenv(dotenv_path=env_path)


def load_data(data_path):

    file_loader = DirectoryLoader(
        path=data_path,
        glob="**/*.pdf",
        loader_cls= UnstructuredFileLoader,       #         PyPDFLoader,                                          
    )
    docs = file_loader.load()

    file_path = os.path.join(data_path, "metadata.jsonl")
    if os.path.exists(file_path):
        print(f" File found: {file_path}")
    else:
        print(f" File not found: {file_path}")
        return docs

    json_docs = []
    with open(file_path, 'r', encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line.strip())
            doc = Document(
                page_content=record.get("Question", ""),
                metadata={
                    "task_id":      record.get("task_id", ""),
                    "level":        record.get("Level", ""),
                    "file_path":    record.get("file_path", ""),
                    "final_answer": record.get("Final answer", ""),
                    "source":       "metadata.jsonl",
                }
            )
            json_docs.append(doc)

    combine_docs = docs + json_docs
    return combine_docs


def embed_and_store(docs, option_embedding):
    model_embeddings = ["BAAI/bge-m3", "meta-llama/Meta-Llama-3-8B", "text-embedding-3-small"]

    supabase_url = os.environ.get("SUPABASE_URL")
    supabase_key = os.environ.get("SUPABASE_SERVICE_ROLE_KEY")
    hf_token = os.environ.get("HUGGINGFACEHUB_API_TOKEN")       
    supabase: Client = create_client(supabase_url, supabase_key)

    if option_embedding == "1":
        print(f"Model_embeddings: {model_embeddings[0]} via HuggingFace API")
        embeddings = HuggingFaceEndpointEmbeddings(     # ← API call, no local download
            model=model_embeddings[0],
            huggingfacehub_api_token=hf_token,
        )
        table_name = "documents1"
        query_name = "match_documents_1"
    elif option_embedding == "3":
        print(f"Model_embeddings: {model_embeddings[2]} via OpenAI API")
        embeddings = OpenAIEmbeddings(model=model_embeddings[2])
        table_name = "documents3"
        query_name = "match_documents_3"
    else:
        print(f"No model suitable for option: {option_embedding}")
        return None

    # semantic chunking------------------------------------
    text_splitter = SemanticChunker(
        embeddings=embeddings,
        breakpoint_threshold_amount=0.85,
    )
    chunks = text_splitter.split_documents(docs)
    print(f"Total chunks after splitting: {len(chunks)}")

    # batch upload------------------------------------
    BATCH_SIZE = 50
    total_batches = (len(chunks) + BATCH_SIZE - 1) // BATCH_SIZE

    vector_store = None
    for i in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[i : i + BATCH_SIZE]
        batch_num = (i // BATCH_SIZE) + 1
        print(f"⏳ Uploading batch {batch_num}/{total_batches} ({len(batch)} chunks)...")

        try:
            if vector_store is None:
                vector_store = SupabaseVectorStore.from_documents(
                    documents=batch,
                    embedding=embeddings,
                    client=supabase,
                    table_name=table_name,
                    query_name=query_name,
                )
            else:
                vector_store.add_documents(batch)

            print(f"✅ Batch {batch_num}/{total_batches} done")
            time.sleep(5)

        except Exception as e:
            print(f"🎯 Batch {batch_num} failed: {e}")
            print(f"   Retrying in 5 seconds...")
            time.sleep(5)
            try:
                vector_store.add_documents(batch)
                print(f" Batch {batch_num} retry succeeded")
            except Exception as e2:
                print(f" Batch {batch_num} retry also failed, skipping: {e2}")
                continue

    print(f"All batches uploaded!")
    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 5},
    )
    return retriever

if __name__ == "__main__":
    data_path = "data/combinedata"

    docs = load_data(data_path)
    print(f"\nTotal docs loaded: {len(docs)}")

    option_embedding = input("Option model for embedding - push 1 or 2 or 3: ")
    retriever = embed_and_store(docs, option_embedding)
    if retriever:
        print("🎯 Retriever ready!")