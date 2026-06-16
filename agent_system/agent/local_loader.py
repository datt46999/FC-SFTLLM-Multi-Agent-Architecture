import torch
import os
from dotenv import load_dotenv

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, pipeline
from langchain_huggingface import HuggingFacePipeline
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import PromptTemplate
from agent_system.agent.system_prompts import system_prompt 
from agent_system.rag.retriever import top_rags  # Ensure this function/retriever accepts the query string

load_dotenv()

top_rags = top_rags()

def load_local_llm(model_name):
    """Loads model into GPU memory once."""
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
    
    # Fix potential missing pad tokens in some causal models (like Llama)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
        
    return model, tokenizer

def get_generate(model, tokenizer, temperature):
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=1024,
        temperature=temperature,
        do_sample=True,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    return HuggingFacePipeline(pipeline=pipe)


MODEL_NAME = os.getenv("LOCAL_MODEL")
if not MODEL_NAME:
    raise ValueError("LOCAL_MODEL environment variable is not set in your .env file.")

print("Loading local LLM into memory...")
MODEL_LLM, TOKENIZER = load_local_llm(MODEL_NAME)
GENERATE = get_generate(model=MODEL_LLM, tokenizer=TOKENIZER, temperature=0)

langchain_prompt = PromptTemplate.from_template(
    template=system_prompt + "\n\nContext:\n{context}\n\nQuestion:\n{question}"
)

def get_text_response(inputs: str):
    rag_chain = (
        {
            "context": top_rags,          
            "question": RunnablePassthrough()
        }
        | langchain_prompt  # <-- Use the wrapped LangChain template here instead of the raw string
        | GENERATE
        | StrOutputParser()
    )
    
    formatted_input = f"<user> {inputs}</user>\n\n<tools>"
    response = rag_chain.invoke(formatted_input)
    return response

if __name__ == "__main__":
    while True:
        query = input("\nQuestions (or type 'exit' to quit): ")
        if query.lower() == 'exit':
            break
        if not query.strip():
            continue
            
        try:
            output = get_text_response(query)
            print(f"\nAnswer:\n{output}")
        except Exception as e:
            print(f"An error occurred: {e}")
