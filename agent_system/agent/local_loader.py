
import torch
import os
from dotenv import load_dotenv

from transformers import  AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig,pipeline

from langchain_huggingface import HuggingFacePipeline
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from agent_system.agent.system_prompts import system_prompt 
from agent_system.rag.retriever import top_rag # top 5 document after using rerank for retriever


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


def get_generate(model, tokenizer, temperature):
    
    pipe = pipeline(
        "text-generation",
        model = model,
        tokenizer = tokenizer,
        max_new_tokens = 1024,
        temperature = temperature,
        do_sample = True,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    return HuggingFacePipeline(pipeline = pipe)
def load_local_llm(model_name):
    quantization_config = BitsAndBytesConfig(
        load_in_4bit = True,
        bnb_4bit_compute_dtype = torch.bfloat16,
        bnb_4bit_quant_type ="nf4"
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config = quantization_config,
        torch_dtype = torch.bfloat16,
        device_map = "auto",
        trust_remote_code = True
    )
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code = True)
    return model, tokenizer
    
def get_text_response(inputs: str, model_name, temperature: float= 0.001):
    
    model_llm, tokenizer = load_local_llm(model_name)

    generate = get_generate(
        model = model_llm,
        tokenizer = tokenizer,
        temperature= temperature 
    )
    text_inputs = f"<user> {inputs}</user>\n\n<tools>"

    rag_chain = (
        {"context": top_rag | format_docs, "question":RunnablePassthrough()}
        | system_prompt 
        | generate
        | StrOutputParser()  # string output parser
    )
    response = rag_chain.invoke(text_inputs)
    return response

if __name__ == "__main__":
    query = input("Questions: ")
    model_name = os.getenv("MODEL_LOCAL")
    print(get_text_response(query, model_name))


