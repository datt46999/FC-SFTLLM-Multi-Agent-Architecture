import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, pipeline
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFacePipeline
from langchain_groq import ChatGroq 






def get_llm():
    """
    Return the appropriate LLM based on LLM_BACKEND env variable.
    Options:
        gpt-4o                                  → OPENAI
        qwen/qwen3-32b                          → OPENROUTER
        gugukaka/Meta_Llama_3_8B_Instruct_xLAM → LOCAL 
        gugukaka/Qwen2_5_7B_Instruct_xLAM      → LOCAL 


    """
    backend = os.getenv("LLM_BACKEND", "LOCAL").strip()

    if backend == "LOCAL":
        model_name = os.getenv("LOCAL_MODEL")
        # return _load_local_llm(model_name), backend
    elif backend == "OPENAI":
        model_name = os.getenv("OPENAI_MODEL")
        print(f"[LLM] Using OpenAI model: {model_name}")
        return ChatOpenAI(model="gpt-4o", temperature=0), backend
    elif backend =="OPENROUTER":
        model_name = os.getenv("OPENROUTER_MODEL")
        return ChatGroq(model=model_name, temperature=0), backend
    else:
        raise ValueError(f"Unknown LLM_BACKEND='{backend}'. Choose 'LOCAL' or 'OPENAI'or 'OPENROUTER'.")