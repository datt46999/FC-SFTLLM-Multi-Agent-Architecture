from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace, HuggingFacePipeline
from dotenv import load_dotenv
from transformers  import AutoModelForCausalLM, AutoTokenizer, pipeline
from pathlib import Path
import torch
env_path = Path(".env")
load_dotenv(dotenv_path=env_path)


def select_llm_model(option_model):

    if option_model == "OpenAI":
        return ChatOpenAI(model="gpt-4o", temperature=0,do_sample=False)
    elif option_model == "HuggingFace":
        llm = HuggingFaceEndpoint(
            repo_id="Qwen/Qwen2.5-32B-Instruct",  # base model được HF support
            temperature=0,
            do_sample=False
        )
        return ChatHuggingFace(llm=llm)
    elif option_model  == "qwen_FT":
        model_path = "./finetune/Qwen2.5-7B-Instruct-xLAM"
        
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=1024,
            do_sample=False
        )
        return HuggingFacePipeline(pipeline=pipe)
    elif option_model == "llama_FT":
        model_path = "./finetune/Qwen2_5_7B_Instruct_xLAM"
        
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=1024,
            do_sample=False
        )
        return HuggingFacePipeline(pipeline=pipe)

    else:
        print("Model not defined")
        return None


# test

# if __name__ == "__main__":
#     for option in ["qwen"]:
#         print(f"\n--- {option} ---")
#         llm = select_llm_model(option)
#         if llm:
#             response = llm.invoke("What is AI?")
#             print(response.content)




# from langchain_huggingface import HuggingFaceEndpoint
# from dotenv import load_dotenv

# load_dotenv()

# question = "If you had a time machine, but could only go to the past or the future once and never return, which would you choose and why?"

# llm = HuggingFaceEndpoint(
#     repo_id="gugukaka/Meta_Llama_3_8B_Instruct_xLAM",
#     temperature=0.7,
#     max_new_tokens=128,
# )

# response = llm.invoke(question)

# print(response)