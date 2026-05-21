
import torch
import os

from typing import Tuple
def setup_hardware_config() -> Tuple[torch.dtype, str]:
    """
    Automatically detect and configure hardware-specific settings.
    
    Returns:
        Tuple[torch.dtype, str]: compute_dtype and attention_implementation
    """
    print("🔍 Detecting hardware capabilities...")
    
    if torch.cuda.is_bf16_supported():
        print("✅ bfloat16 supported - using optimal precision")
        print("📦 Installing FlashAttention for better performance...")
        
        # Install FlashAttention for supported hardware
        os.system('pip install flash_attn --no-build-isolation')
        
        compute_dtype = torch.bfloat16
        attn_implementation = 'flash_attention_2'
        
        print("🚀 Configuration: bfloat16 + FlashAttention 2")
    else:
        print("⚠️  bfloat16 not supported - using float16 fallback")
        compute_dtype = torch.float16
        attn_implementation = 'sdpa'  # Scaled Dot Product Attention
        
        print("🔄 Configuration: float16 + SDPA")
    
    return compute_dtype, attn_implementation


# Configure hardware settings
compute_dtype, attn_implementation = setup_hardware_config()