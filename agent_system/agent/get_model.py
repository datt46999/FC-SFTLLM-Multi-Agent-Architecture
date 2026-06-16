import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, pipeline
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFacePipeline, ChatHuggingFace
from langchain_groq import ChatGroq 

# from agent_system.agent.local_loader import load_local_llm


def load_local_chat_llm(model_name: str):
    """Loads local HF model and wraps it into a LangChain Chat-compatible object."""
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4"
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quantization_config,
        torch_dtype=torch.bfloat16, 
        device_map="auto",
        trust_remote_code=True
    )
    model.eval()
    
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # 1. Create the base text-generation pipeline
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=512,
        temperature=0.0000000001, # Keep it low for agent accuracy
        do_sample=False,
        repetition_penalty=1.3, 
        add_special_tokens=False,
        return_full_text=False,   
        # pad_token_id=tokenizer.pad_token_id,
        # eos_token_id=tokenizer.eos_token_id,
    )
    hf_pipeline = HuggingFacePipeline(pipeline=pipe)


    tokenizer.chat_template = "{% for message in messages %}{{ message['content'] }}{% endfor %}"
    
    chat_model = ChatHuggingFace(llm=hf_pipeline, tokenizer=tokenizer)
    return chat_model

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
        llm = load_local_chat_llm(model_name)
        return llm, backend
    elif backend == "OPENAI":
        model_name = os.getenv("OPENAI_MODEL")
        print(f"[LLM] Using OpenAI model: {model_name}")
        return ChatOpenAI(model="gpt-4o", temperature=0), backend
    elif backend =="OPENROUTER":
        model_name = os.getenv("OPENROUTER_MODEL")
        return ChatGroq(model=model_name, temperature=0), backend
    else:
        raise ValueError(f"Unknown LLM_BACKEND='{backend}'. Choose 'LOCAL' or 'OPENAI'or 'OPENROUTER'.")
