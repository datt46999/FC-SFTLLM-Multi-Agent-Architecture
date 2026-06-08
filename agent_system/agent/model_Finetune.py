import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, pipeline
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFacePipeline
from langchain_groq import ChatGroq 




def _load_local_llm(HF_MODEL_ID) -> HuggingFacePipeline:
    """Load fine-tuned Qwen2.5-xLAM from HuggingFace directly in Python."""
    print(f"[LLM] Loading local model: {HF_MODEL_ID}")

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4"
    )

    tokenizer = AutoTokenizer.from_pretrained(HF_MODEL_ID, trust_remote_code=True)

    model = AutoModelForCausalLM.from_pretrained(
        HF_MODEL_ID,
        quantization_config=quantization_config,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )
    model.eval()

    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=1024,
        temperature=0.01,
        do_sample=True,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    return HuggingFacePipeline(pipeline=pipe)


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
        return _load_local_llm(model_name), backend
    elif backend == "OPENAI":
        model_name = os.getenv("OPENAI_MODEL")
        print(f"[LLM] Using OpenAI model: {model_name}")
        return ChatOpenAI(model=model_name, temperature=0), backend
    elif backend =="OPENROUTER":
        model_name = os.getenv("OPENROUTER_MODEL")
        return ChatGroq(model=model_name, temperature=0), backend
    else:
        raise ValueError(f"Unknown LLM_BACKEND='{backend}'. Choose 'LOCAL' or 'OPENAI'or 'OPENROUTER'.")