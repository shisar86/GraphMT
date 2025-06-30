import argparse
import yaml
import os
import torch
import numpy as np
from types import SimpleNamespace
import sys
sys.path.append('.')

from src.models.lsoformer import LSOformer
from src.utils.data_utils import create_graph_from_bench, tokenize_heuristics

def convert_config_types(config):
    """Convert configuration values to appropriate types"""
    numeric_fields = [
        'learning_rate', 'weight_decay', 'grad_clip', 'val_split', 
        'hidden_dim', 'embedding_dim', 'num_encoder_layers',
        'num_decoder_layers', 'nhead', 'dropout', 'batch_size', 'num_epochs',
        'lr_patience', 'early_stopping_patience', 'checkpoint_interval'
    ]
    
    for field in numeric_fields:
        if hasattr(config, field):
            value = getattr(config, field)
            if isinstance(value, str):
                try:
                    if '.' in value or 'e' in value.lower():
                        setattr(config, field, float(value))
                    else:
                        setattr(config, field, int(value))
                except ValueError:
                    pass
    
    return config

def predict_level(model, bench_file, recipe, vocab, device='cpu'):
    """
    Predict the level for a given bench file and recipe
    
    Args:
        model: Trained LSOformer model
        bench_file: Path to the .bench file
        recipe: List of heuristics
        vocab: Dictionary mapping heuristics to token IDs
        device: Device to use for prediction
        
    Returns:
        Predicted level
    """
    model.to(device)
    model.eval()
    
    # Create graph from bench file
    graph_data, _, _, _ = create_graph_from_bench(bench_file)
    graph_data = graph_data.to(device)
    
    # Tokenize recipe
    recipe_tokens = [vocab.get(h, 0) for h in recipe]  # Use 0 for unknown heuristics
    recipe_tokens = torch.tensor([recipe_tokens], dtype=torch.long).to(device)
    
    # Predict level
    with torch.no_grad():
        predicted_level = model(graph_data, recipe_tokens)
    
    return predicted_level.item()

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Predict levels using LSOformer model')
    parser.add_argument('--config', type=str, default='configs/default.yaml', help='Path to config file')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--bench', type=str, required=True, help='Path to bench file')
    parser.add_argument('--recipe', type=str, required=True, help='Recipe (semicolon-separated heuristics)')
    parser.add_argument('--vocab', type=str, required=True, help='Path to vocabulary file')
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Convert config to namespace for easier access
    config = SimpleNamespace(**config)
    
    # Convert config values to appropriate types
    config = convert_config_types(config)
    
    # Load vocabulary
    with open(args.vocab, 'r') as f:
        vocab = yaml.safe_load(f)
    
    # Create model
    model = LSOformer(
        node_feature_dim=config.node_feature_dim,
        heuristic_vocab_size=len(vocab),
        hidden_dim=config.hidden_dim,
        embedding_dim=config.embedding_dim,
        num_encoder_layers=config.num_encoder_layers,
        num_decoder_layers=config.num_decoder_layers,
        nhead=config.nhead,
        dropout=config.dropout
    )
    
    # Load checkpoint
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Parse recipe
    recipe = [h.strip() for h in args.recipe.split(';')]
    
    # Predict level
    predicted_level = predict_level(model, args.bench, recipe, vocab)
    
    print(f"Predicted level: {predicted_level:.2f}")

if __name__ == '__main__':
    main()
