import torch

from transformers import AutoModelForCausalLM, BitsAndBytesConfig, AutoTokenizer
from peft import LoraConfig, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig

from configs.FT_config import ModelConfig, TrainingConfig

def setup_tokenizer(model_config: ModelConfig) -> AutoTokenizer:
    """
    Initialize and configure the tokenizer using model configuration.
    
    Args:
        model_config: Model configuration with all token information
        
    Returns:
        AutoTokenizer: Configured tokenizer with proper pad token settings
    """
    print(f"🔤 Loading tokenizer for {model_config.model_name}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_config.model_name, use_fast=True)
    
    # Configure padding token using values from model_config
    tokenizer.pad_token = model_config.pad_token
    tokenizer.pad_token_id = model_config.pad_token_id
    tokenizer.padding_side = model_config.padding_side
    
    print(f" Tokenizer configured - pad: '{model_config.pad_token}' (ID: {model_config.pad_token_id})")
    
    return tokenizer  



def create_qlora_model(model_config: ModelConfig,
                       tokenizer: AutoTokenizer,
                       compute_dtype: torch.dtype,
                       attn_implementation:str)-> AutoModelForCausalLM:
    """
    create and config Qlora-enable model for effect fine-tuning

    QLoRA uses 4-bit quantization and low-rank adapters to enable
    fine-tuning large models on consumer GPUs.
    """
    

    print(f"Create Qlora model....{model_config.model_name}")
    
    # configure 4-bit quantization for memoty efficiency
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,                    # Enable 4-bit quantization
        bnb_4bit_quant_type="nf4",           # Use NF4 quantization
        bnb_4bit_compute_dtype=compute_dtype, # Computation data type
        bnb_4bit_use_double_quant=True,      # Double quantization for more memory savings
    )

    print("Loading quantized model....")

    model = AutoModelForCausalLM.from_pretrained(
        model_config.model_name,
        quantization_config = bnb_config,
        device_map={"":0},
        attn_implementation = attn_implementation,
        torch_dtype = compute_dtype,
        trust_remote_code = True
    )

    # prepate model for k-bit training (require for QLoRA)
    model = prepare_model_for_kbit_training(
        model,
        gradient_checkpointing_kwargs = {"use_reentrant": True}
    )

    # configure tokenizer setting in model
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.use_cache = False #Disable cache for training

    return model


def create_lora_config(training_config: TrainingConfig)->LoraConfig:
    """
    Create LoRA configuration for parameter-efficient fine-tuning.
    
    LoRA (Low-Rank Adaptation) adds small trainable matrices to specific
    model layers while keeping the base model frozen.
    
    Args:
        training_config (TrainingConfig): Training configuration with LoRA parameters
        
    Returns:
        LoraConfig: Configured LoRA adapter settings
        
    LoRA Parameters:
        - r (rank): Dimensionality of adaptation matrices (higher = more capacity)
        - alpha: Scaling factor for LoRA weights
        - dropout: Regularization to prevent overfitting
        - target_modules: Which model layers to adapt
    """
    print("Configuring LoRA adapted...")
    # Target modules for both Llama and Qwen architectures
    target_modules = [
        'k_proj', 'q_proj', 'v_proj', 'o_proj',  # Attention projections
        "gate_proj", "down_proj", "up_proj"       # Feed-forward projections
    ]
    
    lora_config = LoraConfig(
        lora_alpha=training_config.lora_alpha,
        lora_dropout=training_config.lora_dropout,
        r=training_config.lora_r,
        bias="none",                             # Don't adapt bias terms
        task_type="CAUSAL_LM",                   # Causal language modeling
        target_modules=target_modules
    )
    
    print(f"🎯 LoRA targeting modules: {target_modules}")
    print(f"📊 LoRA parameters: r={training_config.lora_r}, alpha={training_config.lora_alpha}")
    
    return lora_config
