import argparse
import yaml
import os
import torch
import numpy as np
import random
from types import SimpleNamespace
import sys
sys.path.append('.')

from src.models.lsoformer import LSOformer
from src.training.trainer import LSOformerTrainer
from src.datasets.aig_dataset import MultiAIGDataset

def set_seed(seed):
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

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
    
    bool_fields = ['early_stopping']
    for field in bool_fields:
        if hasattr(config, field):
            value = getattr(config, field)
            if isinstance(value, str):
                setattr(config, field, value.lower() == 'true')
    
    return config

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train LSOformer model')
    parser.add_argument('--config', type=str, default='configs/default.yaml', help='Path to config file')
    parser.add_argument('--bench', type=str, help='Path to bench file')
    parser.add_argument('--recipe', type=str, help='Path to recipe CSV file')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    args = parser.parse_args()
    
    # Set random seed
    set_seed(args.seed)
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Convert config to namespace for easier access
    config = SimpleNamespace(**config)
    
    # Convert config values to appropriate types
    config = convert_config_types(config)
    
    # Update config with command line arguments
    if args.bench:
        config.bench_file = args.bench
    if args.recipe:
        config.recipe_file = args.recipe
    
    # Create dataset
    print(f"Loading dataset from {config.bench_file} and {config.recipe_file}...")
    dataset = MultiAIGDataset(
        bench_dir='data/bench',
        recipe_file=config.recipe_file
    )
    
    # Update heuristic vocabulary size in config
    config.heuristic_vocab_size = dataset.get_vocab_size()
    
    # Create model
    print("Creating LSOformer model...")
    model = LSOformer(
        node_feature_dim=config.node_feature_dim,
        heuristic_vocab_size=config.heuristic_vocab_size,
        hidden_dim=config.hidden_dim,
        embedding_dim=config.embedding_dim,
        num_encoder_layers=config.num_encoder_layers,
        num_decoder_layers=config.num_decoder_layers,
        nhead=config.nhead,
        dropout=config.dropout
    )
    
    # Print model summary
    print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")
    print(f"Training on CPU")
    
    # Create trainer
    trainer = LSOformerTrainer(
        model=model,
        dataset=dataset,
        config=config
    )
    
    # Train model
    print("Starting training...")
    best_val_loss = trainer.train()
    print(f"Training completed with best validation loss: {best_val_loss:.4f}")

if __name__ == '__main__':
    main()
