from dataclasses import dataclass

@dataclass
class ModelConfig:
    """Configuration for model-specific settings."""
    model_name: str           # HuggingFace model identifier
    pad_token: str           # Padding token for the tokenizer
    pad_token_id: int        # Numerical ID for the padding token
    padding_side: str        # Side to add padding ('left' or 'right')
    eos_token: str          # End of sequence token
    eos_token_id: int       # End of sequence token ID
    vocab_size: int         # Vocabulary size
    model_type: str         # Model architecture type

@dataclass
class TrainingConfig:
    """Configuration for training hyperparameters."""
    output_dir: str                    # Directory to save model checkpoints
    batch_size: int = 4              # Training batch size per device
    gradient_accumulation_steps: int = 8  # Steps to accumulate gradients
    learning_rate: float = 1e-4       # Learning rate for optimization
    max_steps: int = 750             # Maximum training steps
    max_seq_length: int = 2048       # Maximum sequence length
    lora_r: int = 16                  # LoRA rank parameter
    lora_alpha: int = 16              # LoRA alpha scaling parameter
    lora_dropout: float = 0.05        # LoRA dropout rate
    save_steps: int = 250             # Steps between checkpoint saves
    logging_steps: int = 10           # Steps between log outputs
    warmup_ratio: float = 0.1   



def create_training_config(model_name: str, **kwargs) -> TrainingConfig:
    """Create training configuration with automatic output directory."""
    # Create clean directory name from model name
    model_clean = model_name.split('/')[-1].replace('-', '_').replace('.', '_')
    default_output_dir = f"./{model_clean}_xLAM"
    
    config_dict = {'output_dir': default_output_dir, **kwargs}
    return TrainingConfig(**config_dict)