# Standard imports
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from typing import List, Dict, Tuple
from dataclasses import dataclass
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
import gc

# Configuration
@dataclass
class Config:
    """Configuration parameters for the analysis"""
    BATCH_SIZE: int = 167  # Chosen as it divides evenly into total token count
    TOTAL_TOKEN_COUNT: int = 128256
    MODEL_1B_PATH: str = "meta-llama/Llama-3.2-1B-Instruct"
    MODEL_3B_PATH: str = "meta-llama/Llama-3.2-3B-Instruct"
    DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"

config = Config()

def load_model_and_tokenizer(model_path: str):
    """
    Load model and tokenizer from path with proper error handling
    
    Args:
        model_path: HuggingFace model path
        
    Returns:
        tuple: (tokenizer, model, embedding_layer, lm_head)
    """
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(model_path).to(config.DEVICE)
        embedding = model.model.embed_tokens
        lm_head = model.lm_head
        return tokenizer, model, embedding, lm_head
    except Exception as e:
        raise RuntimeError(f"Error loading model {model_path}: {str(e)}")

def find_embedding_l2_info(model_embedding) -> torch.Tensor:
    """Calculate L2 norms of embedding weights"""
    model_embedding_weights = model_embedding.weight
    return torch.norm(model_embedding_weights, p=2, dim=1)

def find_max_min_var(l2_norms: torch.Tensor) -> Tuple[float, float, float]:
    """Find max, min and variance of L2 norms"""
    return (
        torch.max(l2_norms).item(),
        torch.min(l2_norms).item(),
        torch.var(l2_norms).item()
    )

def find_max_min_indices(l2_norms: torch.Tensor) -> Tuple[int, int]:
    """Find indices of max and min L2 norms"""
    max_val = torch.max(l2_norms)
    min_val = torch.min(l2_norms)
    return (
        torch.where(l2_norms == max_val)[0].item(),
        torch.where(l2_norms == min_val)[0].item()
    )

def find_mismatch_indices(
    tot_token_count: int,
    model_embedding,
    lm_head,
    batch_size: int = config.BATCH_SIZE
) -> Tuple[List[int], List[int]]:
    """
    Find indices where embedding->lm_head mapping doesn't preserve token identity
    
    Args:
        tot_token_count: Total number of tokens to check
        model_embedding: Embedding layer
        lm_head: Language model head
        batch_size: Batch size for processing
        
    Returns:
        Tuple of (original indices, mismatched indices)
    """
    original_different_indices = []
    mismatched_indices = []
    
    try:
        for i in tqdm(range(tot_token_count // batch_size)):
            inputs = torch.tensor([j + i * batch_size for j in range(batch_size)]).to(config.DEVICE)
            
            with torch.no_grad():
                embedding_out = model_embedding(inputs)
                head_out = lm_head(embedding_out)
                returned_items = torch.argmax(head_out, dim=-1)
                
            mask = returned_items != inputs
            if torch.any(mask):
                original_different_indices.extend(inputs[mask].cpu().tolist())
                mismatched_indices.extend(returned_items[mask].cpu().tolist())
                
            # Clear cache periodically
            if i % 10 == 0 and config.DEVICE == "cuda":
                torch.cuda.empty_cache()
                
    except Exception as e:
        raise RuntimeError(f"Error in mismatch detection: {str(e)}")
        
    return original_different_indices, mismatched_indices

def analyze_mismatches(
    original_indices: List[int],
    mismatched_indices: List[int],
    tokenizer
) -> pd.DataFrame:
    """
    Create a DataFrame analyzing token mismatches
    
    Args:
        original_indices: Original token indices
        mismatched_indices: Mismatched token indices
        tokenizer: Tokenizer for decoding indices
        
    Returns:
        DataFrame with mismatch analysis
    """
    data = []
    for orig, mismatch in zip(original_indices, mismatched_indices):
        data.append({
            'Original Index': orig,
            'Mismatched Index': mismatch,
            'Original Token': tokenizer.decode([orig]),
            'Mismatched Token': tokenizer.decode([mismatch])
        })
    return pd.DataFrame(data)

def main():
    """Main analysis workflow"""
    print(f"Using device: {config.DEVICE}")
    
    # Load models
    print("Loading 1B model...")
    tokenizer_1b, model_1b, embed_1b, lm_head_1b = load_model_and_tokenizer(config.MODEL_1B_PATH)
    
    print("Loading 3B model...")
    tokenizer_3b, model_3b, embed_3b, lm_head_3b = load_model_and_tokenizer(config.MODEL_3B_PATH)
    
    # Verify embedding-lm_head weight sharing
    print("\nChecking weight sharing:")
    print(f"1B model: {torch.all(embed_1b.weight == lm_head_1b.weight)}")
    print(f"3B model: {torch.all(embed_3b.weight == lm_head_3b.weight)}")
    
    # L2 norm analysis
    print("\nAnalyzing L2 norms...")
    l2_norms_1b = find_embedding_l2_info(embed_1b)
    l2_norms_3b = find_embedding_l2_info(embed_3b)
    
    max_1b, min_1b, var_1b = find_max_min_var(l2_norms_1b)
    max_3b, min_3b, var_3b = find_max_min_var(l2_norms_3b)
    
    print(f"\n1B Model L2 Stats:")
    print(f"Max: {max_1b:.4f}")
    print(f"Min: {min_1b:.4f}")
    print(f"Variance: {var_1b:.6f}")
    
    print(f"\n3B Model L2 Stats:")
    print(f"Max: {max_3b:.4f}")
    print(f"Min: {min_3b:.4f}")
    print(f"Variance: {var_3b:.6f}")
    
    # Find mismatches
    print("\nFinding token mismatches...")
    orig_1b, mismatch_1b = find_mismatch_indices(
        config.TOTAL_TOKEN_COUNT, embed_1b, lm_head_1b
    )
    orig_3b, mismatch_3b = find_mismatch_indices(
        config.TOTAL_TOKEN_COUNT, embed_3b, lm_head_3b
    )
    
    # Analyze results
    print("\nAnalyzing mismatches...")
    df_1b = analyze_mismatches(orig_1b, mismatch_1b, tokenizer_1b)
    df_3b = analyze_mismatches(orig_3b, mismatch_3b, tokenizer_3b)
    
    print(f"\n1B Model Mismatch Rate: {len(orig_1b)/config.TOTAL_TOKEN_COUNT:.4%}")
    print(f"3B Model Mismatch Rate: {len(orig_3b)/config.TOTAL_TOKEN_COUNT:.4%}")
    
    # Save results
    print("\nSaving results...")
    df_1b.to_csv('1b_mismatches.csv', index=False)
    df_3b.to_csv('3b_mismatches.csv', index=False)
    
    # Cleanup
    if config.DEVICE == "cuda":
        torch.cuda.empty_cache()
    gc.collect()

if __name__ == "__main__":
    main()