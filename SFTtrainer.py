from finetune.scripts.merge_adapter   import setup_tokenizer, create_qlora_model
from finetune.scripts.training import auto_configure_model, train_qLoRA_model
from finetune.scripts.prepare_dataset import load_and_process_dataset, preview_dataset_sample
from finetune.scripts.setup_ import compute_dtype, attn_implementation
from configs.FT_config import ModelConfig, TrainingConfig, create_training_config

def run_finetuning_pipeline(
    model_name: str,
    custom_pad_token: str = "<|eot_id|>",
    sample_size: int = None,
    dataset_preview_index: int = 0
):
    """
    End-to-end QLoRA fine-tuning pipeline.

    Args:
        model_name: HuggingFace model identifier
        custom_pad_token: Pad token for the model family
            - '<|eot_id|>'  → Llama 3+ 
            - '<|im_end|>'  → Qwen 2+
            - '</s>'        → Mistral
            - '<|end|>'     → Phi 3+
        sample_size: Number of dataset samples to use (None = full dataset)
        dataset_preview_index: Index of the sample to preview
    """


    print(f"🎯 Selected Model: {model_name}")
    print(f"\n🔧 Auto-configuring everything for {model_name}...")

    # Auto-configure model and training
    model_config = auto_configure_model(model_name, custom_pad_token=custom_pad_token)
    training_config = create_training_config(model_name)

    print(f"\n🎉 Ready to fine-tune! Everything configured automatically:")
    print(f"   ✅ Model type:  {model_config.model_type}")
    print(f"   ✅ Vocabulary:  {model_config.vocab_size:,} tokens")
    print(f"   ✅ Pad token:   '{model_config.pad_token}' (ID: {model_config.pad_token_id})")
    print(f"   ✅ Output dir:  {training_config.output_dir}")
    print(f"\n🚀 Configuration complete for {model_name}!")



    # Step 2 – Tokenizer
    print(f"\n📝 Setting up tokenizer...")
    tokenizer = setup_tokenizer(model_config)

    # Step 3 – Dataset
    print(f"\n📊 Loading and processing xLAM dataset...")
    dataset = load_and_process_dataset(tokenizer, sample_size=sample_size)

    print(f"\n👀 Dataset sample preview:")
    preview_dataset_sample(dataset, index=dataset_preview_index)

    # Step 4 – Model
    print(f"\n🏗️  Creating QLoRA model...")
    model = create_qlora_model(
        model_config,
        tokenizer,
        compute_dtype,
        attn_implementation
    )

    # Step 5 – Training
    print(f"\n🎯 Starting training...")
    trainer = train_qLoRA_model(
        dataset=dataset,
        model=model,
        training_config=training_config,
        compute_dtype=compute_dtype,
        model_name = model_name
    )

    print(f"\n🎉 Fine-tuning completed for {model_config.model_name.split('/')[-1]}!")
    print(f"📁 Model saved to: {training_config.output_dir}")
    print(f"🔍 To test the model, run the inference cells below")

    return trainer, model, tokenizer, model_config, training_config



if __name__ == "__main__":
    model_name="Qwen/Qwen2.5-7B-Instruct"
    run_finetuning_pipeline(
        model_name= model_name,
        custom_pad_token='<|im_end|>',
    )
  





