import argparse
import yaml
import os
import torch
import torch.nn as nn
import numpy as np
import random
from types import SimpleNamespace

from src.models.lsoformer import LSOformer
from src.datasets.aig_dataset import AIGDataset
from src.training.trainer import LSOformerTrainer
from src.training.ssl_trainer import SSLTrainer

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

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train LSOformer model')
    parser.add_argument('--config', type=str, default='configs/default.yaml', help='Path to config file')
    parser.add_argument('--dataset', type=str, choices=['EPFL', 'OABCD', 'PD'], default='EPFL', help='Dataset to use')
    parser.add_argument('--ssl', action='store_true', help='Use SSL auxiliary task')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    args = parser.parse_args()
    
    # Set random seed
    set_seed(args.seed)
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Load dataset-specific config if available
    dataset_config_path = f'configs/{args.dataset.lower()}.yaml'
    if os.path.exists(dataset_config_path):
        with open(dataset_config_path, 'r') as f:
            dataset_config = yaml.safe_load(f)
            # Update config with dataset-specific values
            for key, value in dataset_config.items():
                config[key] = value
    
    # Convert config to namespace for easier access
    config = SimpleNamespace(**config)
    
    # Create datasets
    train_dataset = AIGDataset(
        data_dir=config.data_dir,
        dataset_name=args.dataset,
        split='train'
    )
    
    val_dataset = AIGDataset(
        data_dir=config.data_dir,
        dataset_name=args.dataset,
        split='val'
    )
    
    # Create model
    model = LSOformer(
        node_feature_dim=config.node_feature_dim,
        heuristic_vocab_size=config.heuristic_vocab_size,
        hidden_dim=config.hidden_dim,
        embedding_dim=config.embedding_dim,
        num_encoder_layers=config.num_encoder_layers,
        num_decoder_layers=config.num_decoder_layers,
        nhead=config.nhead,
        dropout=config.dropout,
        encoder_type=config.encoder_type
    )
    
    # Print model summary
    print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Create trainer
    if args.ssl:
        trainer = SSLTrainer(
            model=model,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            config=config
        )
    else:
        trainer = LSOformerTrainer(
            model=model,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            config=config
        )
    
    # Train model
    best_val_loss = trainer.train()
    print(f"Training completed with best validation loss: {best_val_loss:.4f}")

if __name__ == '__main__':
    main()
