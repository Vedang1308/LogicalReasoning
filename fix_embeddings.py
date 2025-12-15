#!/usr/bin/env python3
"""
fix_embeddings.py - Bake connector boost into model weights

This script:
1. Loads the trained model (from ./output/connector_model/final)
2. Identifies connector token IDs using the tokenizer and config
3. Multiplies the embedding weights for these tokens by the boost factor (1.1x)
4. Saves the fixed model to ./output/connector_model/fixed

This ensures the model performs correctly during standard evaluation (which doesn't apply the boost).
"""

import torch
import logging
import os
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
from utils.config import Config

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def get_connector_token_ids(tokenizer, config: Config):
    """Find token IDs for all connector words."""
    logger.info("Identifying connector token IDs...")
    
    connector_words = set()
    for _, words in config.connector_types.items():
        connector_words.update(words)
    
    logger.info(f"Found {len(connector_words)} unique connector phrases in config")
    
    # Simple strategy: tokenize each word and take the first token
    # This aligns with how the single-token boost was likely intended/implemented
    token_ids = set()
    
    for word in connector_words:
        # Note: Add leading space for coherence with Llama tokenizer
        ids = tokenizer.encode(" " + word, add_special_tokens=False)
        if ids:
            token_ids.add(ids[0]) # Boost the primary token
            
            # Also try without space
            ids_nospace = tokenizer.encode(word, add_special_tokens=False)
            if ids_nospace:
                token_ids.add(ids_nospace[0])
                
    logger.info(f" Identified {len(token_ids)} unique token IDs to boost")
    return token_ids

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Bake connector boost into model weights")
    parser.add_argument("--input_path", type=str, default="./output/connector_model/final",
                      help="Path to the trained model (default: ./output/connector_model/final)")
    parser.add_argument("--boost_factor", type=float, default=None,
                      help="Override config boost factor (e.g. 1.05)")
                      
    args = parser.parse_args()

    logger.info("="*70)
    logger.info("FIXING MODEL EMBEDDINGS")
    logger.info("="*70)
    
    input_path = args.input_path
    output_path = args.output_path
    
    if not os.path.exists(input_path):
        if "/" in input_path:
            logger.info(f"Input path '{input_path}' not found locally. Assuming it is a HuggingFace Hub repository.")
        else:
            logger.error(f"❌ Input model not found at {input_path}")
            logger.info(" Please verify the path or provide a valid HuggingFace repo ID.")
            return
        
    config = Config()
    # Use argument if provided, else config
    boost_factor = args.boost_factor if args.boost_factor is not None else config.boost_factor
    logger.info(f"Boost factor: {boost_factor}x")
    
    # 1. Load Tokenizer
    logger.info(f"\n[1/4] Loading tokenizer from {input_path}...")
    try:
        # Added fix_mistral_regex=True to suppress warnings and ensure correctness
        tokenizer = AutoTokenizer.from_pretrained(input_path, fix_mistral_regex=True)
    except Exception as e:
        logger.error(f"Failed to load tokenizer from {input_path}: {e}")
        return
    
    # 2. Load Model
    logger.info(f"\n[2/4] Loading model from {input_path}...")
    try:
        # Load with torch_dtype=torch.float16 or bfloat16 to match training
        model = AutoModelForCausalLM.from_pretrained(
            input_path, 
            torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
            device_map="auto"
        )
    except Exception as e:
        logger.error(f"Failed to load model from {input_path}: {e}")
        return
    
    # 3. Apply Boost
    logger.info(f"\n[3/4] Baking boost into embeddings...")
    connector_ids = get_connector_token_ids(tokenizer, config)
    
    # Access embedding layer
    # For Llama, it's model.model.embed_tokens
    embeddings = model.get_input_embeddings()
    original_weights = embeddings.weight.data.clone()
    
    count = 0
    with torch.no_grad():
        for tid in connector_ids:
            if tid < embeddings.num_embeddings:
                embeddings.weight.data[tid] *= boost_factor
                count += 1
                
    logger.info(f"✓ Boosted {count} tokens by {boost_factor}x")
    
    # Verify a sample
    if connector_ids:
        sample_id = list(connector_ids)[0]
        old_norm = torch.norm(original_weights[sample_id]).item()
        new_norm = torch.norm(embeddings.weight.data[sample_id]).item()
        ratio = new_norm/old_norm if old_norm > 0 else 0
        logger.info(f"  Sample token {sample_id} norm: {old_norm:.4f} -> {new_norm:.4f} (Ratio: {ratio:.2f}x)")

    # 4. Save Model
    logger.info(f"\n[4/4] Saving fixed model to {output_path}...")
    try:
        model.save_pretrained(output_path)
        tokenizer.save_pretrained(output_path)
        logger.info(f"✓ Model and tokenizer saved")
    except Exception as e:
         logger.error(f"Failed to save model to {output_path}: {e}")
         return
    
    logger.info("\n" + "="*70)
    logger.info(f"✅ DONE! Fixed model saved to: {output_path}")
    logger.info("You can now run evaluation on this path.")
    logger.info("="*70)

if __name__ == "__main__":
    main()
