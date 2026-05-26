from peft import PeftModel
import torch
from typing import Tuple

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from configs.FT_config import ModelConfig, TrainingConfig, create_training_config
from finetune.scripts.merge_adapter import setup_tokenizer
from finetune.scripts.setup_ import compute_dtype, attn_implementation      # 
from finetune.scripts.training import auto_configure_model                   #
# from dotenv import load_dotenv
# from pathlib import Path

# env_path = Path(".env")
# load_dotenv(dotenv_path=env_path)
def load_trained_model(
    model_config: ModelConfig,
    adapter_path: str,
    compute_dtype: torch.dtype,
    attn_implementation: str
) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:

    print(f"Load trained model from {adapter_path}")   
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_quant_type="nf4"
    )

    tokenizer = setup_tokenizer(model_config)
    print(f"Tokenizer loaded for {model_config.model_name}")

    base_model = AutoModelForCausalLM.from_pretrained(
        model_config.model_name,
        quantization_config=quantization_config,
        torch_dtype=compute_dtype,
        device_map={"": 0},
        attn_implementation=attn_implementation,
        trust_remote_code=True
    )

    print(f"Load LoRA adapter from {adapter_path}...")
    model = PeftModel.from_pretrained(base_model, adapter_path)


    merged_model = model.merge_and_unload()

    # save merged model
    save_path = "./gugukaka/Meta_Llama_3_8B_Instruct_xLAM"

    merged_model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    merged_model.eval()          # ✅ eval on merged_model
    # repo_id = "gugukaka/Meta_Llama_3_8B_Instruct_xLAM"
    # merged_model.push_to_hub(repo_id)
    # tokenizer.push_to_hub(repo_id)
    return merged_model, tokenizer


def generate_function_call(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompt: str,
    max_new_tokens: int = 1024,
    temperature: float = 0.7,
    do_sample: bool = True
) -> str:

    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")  # ✅ tránh dùng 'input' là keyword

    generation_kwargs = {
        "max_new_tokens": max_new_tokens,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "do_sample": do_sample
    }
    if do_sample:
        generation_kwargs["temperature"] = temperature

    with torch.no_grad():
        output = model.generate(**inputs, **generation_kwargs)

    result = tokenizer.decode(output[0], skip_special_tokens=True)
    return result
def test_function_calling_examples(model: AutoModelForCausalLM, 
                                   tokenizer: AutoTokenizer)-> None:
    

    print("test function calling capabilities")


    test_cases = [
        {
            "name":"Mathematical Function",
            "prompt": "<user> Check if the numbers 8 and 1233 are powersof two.</user>\n\n<tools>"
        },
        {
            "name":"Weather Query",
            "prompt": "<user> What's the wether like in New Youk today?</user>\n\n<tools>"
        },
        {
            "name":"Data Proceccing",
            "prompt": "<user> Calculate the average on these numbers:10, 20, 30, 40, 50</user>\n\n<tools>"
        },

    ]
    for i , test_case in enumerate(test_cases,1):
        print(f"\n {'='*60}")
        print(f"Test case {i}: {test_case['name']}")
        print(f"{'='*60}\n ")

        results = generate_function_call(
            model= model, 
            tokenizer=tokenizer,
            prompt=test_case["prompt"],
            max_new_tokens=1024,
            temperature= 0.7, 
            do_sample= True
        )
        print(f"\n Complete response: ")
        print("-"*60)
        print(results)
        print("-"*60)

    print("All test cases complete")


if __name__ == "__main__":

    model_name = "Qwen/Qwen2.5-7B-Instruct"
    custom_pad_token = "<|im_end|>"


    # model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
    # custom_pad_token = "<|eot_id|>"
    model_config = auto_configure_model(model_name, custom_pad_token=custom_pad_token)
    training_config = create_training_config(model_name)

    # ✅ Tìm checkpoint mới nhất nếu max_steps không chính xác
    adapter_path = "./finetune/Qwen2_5_7B_Instruct_xLAM/checkpoint-750"
    print(f"Loading adapter from: {adapter_path}")

    trained_model, trained_tokenizer = load_trained_model(
        model_config=model_config,
        adapter_path=adapter_path,
        compute_dtype=compute_dtype,
        attn_implementation=attn_implementation
    )

    # test_prompt = "<user>Check if the numbers 8 and 124 are powers of two.</user>\n\n<tools>"  # ✅ fix typo "ans"
    # result = generate_function_call(trained_model, trained_tokenizer, test_prompt)

    # print(f"\nTest result for {model_config.model_name.split('/')[-1]}")
    # print(result)

    # test_function_calling_examples(trained_model,trained_tokenizer)