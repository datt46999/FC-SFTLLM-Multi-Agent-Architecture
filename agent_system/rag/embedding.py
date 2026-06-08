import json
from typing import List, Optional
from tqdm import tqdm
import os
import pickle
import time

from datasets import load_dataset
from transformers import AutoTokenizer
from supabase.client import Client, create_client

from langchain_community.document_loaders import DirectoryLoader, UnstructuredFileLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document as LangchainDocument
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain_community.vectorstores import SupabaseVectorStore, FAISS
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai import OpenAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpointEmbeddings
from pathlib import Path
from dotenv import load_dotenv

env_path = Path("") / ".env"
load_dotenv(dotenv_path=env_path)


MARKDOWN_SEPARATORS = [
    "\n#{1,6} ",
    "```\n",
    "\n\\*\\*\\*+\n",
    "\n---+\n",
    "\n___+\n",
    "\n\n",
    "\n",
    " ",
    "",
]

EMBEDDING_MODEL_NAME =  "BAAI/bge-m3"
VECTORSTORE_PATH = "./data/vectorstore/faiss_index"

def load_docs(file_path)->List[LangchainDocument]:
    """
    Load GAIA questions from metadata.jsonl and HuggingFace docs,
    then combine them into a single knowledge base.
    """
    # ── 1. Load GAIA metadata.jsonl ──────────────────────────────────────────
    gaia_docs: List[LangchainDocument] = []

    file_path = os.path.join(file_path, "metadata.jsonl")
    if os.path.exists(file_path):
        print(f" File found: {file_path}")
    else:
        print(f" File not found: {file_path}")
       
    
    with open(file_path, 'r', encoding="utf-8") as f:
        
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line.strip())
            doc = LangchainDocument(                       
                page_content=record.get("Question", ""),
                metadata={
                    "task_id":      record.get("task_id", ""),
                    "level":        record.get("Level", ""),
                    "file_path":    record.get("file_path", ""),
                    "final_answer": record.get("Final answer", ""),
                    "source":       "metadata.jsonl",
                }
            )
            gaia_docs.append(doc)

    # ── 2. Load HuggingFace doc knowledge base ───────────────────────────────
    hf_dataset =load_dataset("m-ric/huggingface_doc")["train"]  # renamed import alias
    hf_docs: List[LangchainDocument] = [
        LangchainDocument(
            page_content=doc["text"],
            metadata={"source": doc["source"]}
        )
        for doc in tqdm(hf_dataset, desc="Loading HF docs")
    ]

    # ── 3. Combine both sources ───────────────────────────────────────────────
    combined_docs: List[LangchainDocument] = gaia_docs + hf_docs
    print(f"Loaded {len(gaia_docs)} GAIA docs and {len(hf_docs)} HF docs "
          f"→ {len(combined_docs)} total")
    return combined_docs


def split_documents(
    chunk_size: int,
    knowledge_base: List[LangchainDocument],
    tokenizer_name: Optional[str] = EMBEDDING_MODEL_NAME,
) -> List[LangchainDocument]:

    text_splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer( 
        AutoTokenizer.from_pretrained(tokenizer_name),
        chunk_size=chunk_size,
        chunk_overlap=int(chunk_size / 10),
        add_start_index=True,
        strip_whitespace=True,
        separators=MARKDOWN_SEPARATORS,                    
    )

    docs_processed: List[LangchainDocument] = []
    for doc in tqdm(knowledge_base, desc="Splitting documents"):
        docs_processed += text_splitter.split_documents([doc])

    # Deduplicate by page content
    seen_contents: dict[str, bool] = {}                    
    docs_unique_processed: List[LangchainDocument] = []
    for doc in docs_processed:
        if doc.page_content not in seen_contents:
            seen_contents[doc.page_content] = True          
            docs_unique_processed.append(doc)

    print(f"Split into {len(docs_processed)} chunks → "
          f"{len(docs_unique_processed)} unique chunks after dedup")
    
   
    return docs_unique_processed


def create_vectorstore(
    docs_processed: List[LangchainDocument],
    model_name: str = EMBEDDING_MODEL_NAME,     
    save_path: str = VECTORSTORE_PATH,           
) -> FAISS:                                                 

    embedding_model = HuggingFaceEmbeddings(
        model_name=model_name,                             
        multi_process=True,
        model_kwargs={"device": "cuda"},
        encode_kwargs={"normalize_embeddings": True},
    )

    knowledge_vector_store = FAISS.from_documents(
        docs_processed,
        embedding_model,
        distance_strategy=DistanceStrategy.COSINE,
    )
    os.makedirs(save_path, exist_ok=True)
    knowledge_vector_store.save_local(save_path)
    print(f"Vector store saved to: {save_path}")
    return knowledge_vector_store

if __name__ =="__main__":





    Knowledge_base = load_docs("data/combinedata")
    split_doc = split_documents(
        chunk_size= 512,
        knowledge_base= Knowledge_base,
        tokenizer_name= EMBEDDING_MODEL_NAME
    )
    vectorsotor =create_vectorstore(
        docs_processed=split_documents,
        model_name=EMBEDDING_MODEL_NAME,
        save_path= VECTORSTORE_PATH
     )

    # embedding_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    # vectorstore = FAISS.load_local(
    #     "./data/vectorstore/faiss_index",
    #     embedding_model,
    #     allow_dangerous_deserialization=True
    # )
    print(f"Vectorstore contains {vectorstore.index.ntotal} vectors")