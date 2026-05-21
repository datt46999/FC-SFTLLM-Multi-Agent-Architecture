import torch


from trl import SFTTrainer, SFTConfig
from datasets import Dataset
from transformers import AutoModelForCausalLM,AutoTokenizer, AutoConfig

from configs.FT_config import TrainingConfig, ModelConfig


from finetune.scripts.merge_adapter import create_lora_config
from finetune.scripts.setup_ import compute_dtype, attn_implementation



def auto_configure_model(model_name:str, custom_pad_token:str = None)-> ModelConfig:
    """
    auto configure any model from extracting infor from its tokenizer
    Args:
        model_name: HuggingFace model identifier
        custom_pad_token: Custom pad token if model doesn't have one
        
    Returns:
        ModelConfig: Complete model configuration
    """
    print(f"Loading model configuration: {model_name}")

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast = True)
    model_config = AutoConfig.from_pretrained(model_name)

    # get basic infor of model
    model_type = getattr(model_config, "model_type", "unknown")
    vocab_size = getattr(model_config, "vocab_size", len(tokenizer.get_vocab()))

    print(f"Model type: {model_type}, vocab_size: {vocab_size:,}")

    # get EOS token(required)
    eos_token = tokenizer.eos_token
    eos_token_id = tokenizer.eos_token_id
    if eos_token is None:
        raise ValueError(f"Model {model_name} miss EOS token")
    
    pad_token = tokenizer.pad_token
    pad_token_id = tokenizer.pad_token_id

    if pad_token is None:
        if custom_pad_token is None:
            raise ValueError(f"Model needs custom_pad_token. Use '<|eot_id|>' for Llama, '<|im_end|>' for Qwen")
        
        pad_token = custom_pad_token
        if pad_token in tokenizer.get_vocab():
            pad_token_id = tokenizer.get_vocab()[pad_token]
        else:
            tokenizer.add_special_tokens({'pad_token': pad_token})
            pad_token_id = tokenizer.pad_token_id
    
    print(f"✅ Configured - pad: '{pad_token}' (ID: {pad_token_id}), eos: '{eos_token}' (ID: {eos_token_id})")
    
    return ModelConfig(
        model_name=model_name,
        pad_token=pad_token,
        pad_token_id=pad_token_id,
        padding_side='left',  # Standard for causal LMs
        eos_token=eos_token,
        eos_token_id=eos_token_id,
        vocab_size=vocab_size,
        model_type=model_type
    )

def train_qLoRA_model(dataset: Dataset,
                      model: AutoModelForCausalLM,
                      training_config: TrainingConfig,
                      compute_dtype: torch.dtype, model_name:str) ->SFTTrainer:
                      
    dataset = dataset.select_columns(["text"])                  
    peft_config = create_lora_config(training_config)

    training_arguments = SFTConfig(
        output_dir=training_config.output_dir,
        optim="adamw_8bit",                      # 8-bit optimizer for memory efficiency
        per_device_train_batch_size=training_config.batch_size,
        gradient_accumulation_steps=training_config.gradient_accumulation_steps,
        log_level="info",                        # Detailed logging
        save_steps=training_config.save_steps,
        logging_steps=training_config.logging_steps,
        learning_rate=training_config.learning_rate,
        fp16=compute_dtype == torch.float16,     # Use FP16 if not using bfloat16
        bf16=compute_dtype == torch.bfloat16,    # Use bfloat16 if supported
        max_steps=training_config.max_steps,
        warmup_ratio=training_config.warmup_ratio,
        lr_scheduler_type="linear",
        dataset_text_field="text",               # Field containing training text
        max_length=training_config.max_seq_length,
        remove_unused_columns=True,             # Keep all dataset columns
        
        # Additional stability and performance settings
        dataloader_drop_last=True,               # Drop incomplete batches
        gradient_checkpointing=True,             # Enable gradient checkpointing
        save_total_limit=3,                      # Keep only 3 most recent checkpoints
        load_best_model_at_end=False,            # Don't load best model (saves memory)
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        peft_config=peft_config,
        args=training_arguments,
    )
    
    print(f"📊 Training configuration:")
    print(f"   • Dataset size: {len(dataset):,} samples")
    print(f"   • Batch size: {training_config.batch_size}")
    print(f"   • Gradient accumulation: {training_config.gradient_accumulation_steps}")
    print(f"   • Effective batch size: {training_config.batch_size * training_config.gradient_accumulation_steps}")
    print(f"   • Max steps: {training_config.max_steps:,}")
    print(f"   • Learning rate: {training_config.learning_rate}")
    print(f"   • Output directory: {training_config.output_dir}")
    
    # Start training
    print("\n🏁 Beginning training...")
    trainer.train()

    model_clean = model_name.split('/')[-1].replace('-', '_').replace('.', '_')
    trainer.push_to_hub(f"gugukaka/{model_clean}_xLAM")
    print("✅ Training completed successfully!")
    
    return trainer