import argparse
import yaml
import os
import torch
import numpy as np
from types import SimpleNamespace
from tqdm import tqdm

from src.models.lsoformer import LSOformer
from src.datasets.aig_dataset import AIGDataset
from src.utils.metrics import mean_absolute_percentage_error, root_mean_squared_error

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Evaluate LSOformer model')
    parser.add_argument('--config', type=str, default='configs/default.yaml', help='Path to config file')
    parser.add_argument('--dataset', type=str, choices=['EPFL', 'OABCD', 'PD'], default='EPFL', help='Dataset to use')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--output', type=str, default='results.json', help='Path to output file')
    args = parser.parse_args()
    
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
    
    # Create test dataset
    test_dataset = AIGDataset(
        data_dir=config.data_dir,
        dataset_name=args.dataset,
        split='test'
    )
    
    # Create data loader
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True
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
    
    # Load checkpoint
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    # Evaluate model
    model.eval()
    delay_mape_list = []
    area_mape_list = []
    delay_rmse_list = []
    area_rmse_list = []
    
    results = {
        'circuit_results': [],
        'overall_metrics': {}
    }
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            # Move data to device
            graph_data = batch['graph_data'].to(device)
            recipe_tokens = batch['recipe_tokens'].to(device)
            node_depths = batch['node_depths'].to(device)
            
            # Get target QoR
            delay_trajectory = batch['delay_trajectory'].to(device)
            area_trajectory = batch['area_trajectory'].to(device)
            
            # Forward pass
            predicted_qor = model(graph_data, recipe_tokens, node_depths)
            
            # Calculate metrics for delay
            delay_mape = mean_absolute_percentage_error(predicted_qor, delay_trajectory)
            delay_rmse = root_mean_squared_error(predicted_qor, delay_trajectory)
            
            # Calculate metrics for area (assuming model can predict both)
            # In practice, you might need a separate model or output head for area
            area_mape = mean_absolute_percentage_error(predicted_qor, area_trajectory)
            area_rmse = root_mean_squared_error(predicted_qor, area_trajectory)
            
            # Store metrics
            delay_mape_list.append(delay_mape.item())
            area_mape_list.append(area_mape.item())
            delay_rmse_list.append(delay_rmse.item())
            area_rmse_list.append(area_rmse.item())
            
            # Store individual circuit results
            for i in range(len(batch['circuit_id'])):
                circuit_id = batch['circuit_id'][i]
                recipe_id = batch['recipe_id'][i]
                
                # Get predictions and targets for current sample
                pred_delay = predicted_qor[i].cpu().numpy().tolist()
                true_delay = delay_trajectory[i].cpu().numpy().tolist()
                pred_area = predicted_qor[i].cpu().numpy().tolist()  # Same as delay for now
                true_area = area_trajectory[i].cpu().numpy().tolist()
                
                # Calculate final QoR metrics
                final_delay_mape = mean_absolute_percentage_error(
                    predicted_qor[i, -1:], delay_trajectory[i, -1:]
                ).item()
                
                final_area_mape = mean_absolute_percentage_error(
                    predicted_qor[i, -1:], area_trajectory[i, -1:]
                ).item()
                
                results['circuit_results'].append({
                    'circuit_id': circuit_id,
                    'recipe_id': recipe_id,
                    'predicted_delay': pred_delay,
                    'true_delay': true_delay,
                    'predicted_area': pred_area,
                    'true_area': true_area,
                    'final_delay_mape': final_delay_mape,
                    'final_area_mape': final_area_mape
                })
    
    # Calculate overall metrics
    avg_delay_mape = np.mean(delay_mape_list)
    avg_area_mape = np.mean(area_mape_list)
    avg_delay_rmse = np.mean(delay_rmse_list)
    avg_area_rmse = np.mean(area_rmse_list)
    
    std_delay_mape = np.std(delay_mape_list)
    std_area_mape = np.std(area_mape_list)
    std_delay_rmse = np.std(delay_rmse_list)
    std_area_rmse = np.std(area_rmse_list)
    
    results['overall_metrics'] = {
        'delay_mape': {
            'mean': avg_delay_mape,
            'std': std_delay_mape
        },
        'area_mape': {
            'mean': avg_area_mape,
            'std': std_area_mape
        },
        'delay_rmse': {
            'mean': avg_delay_rmse,
            'std': std_delay_rmse
        },
        'area_rmse': {
            'mean': avg_area_rmse,
            'std': std_area_rmse
        }
    }
    
    # Print results
    print(f"Evaluation results on {args.dataset} dataset:")
    print(f"Delay MAPE: {avg_delay_mape:.4f} ± {std_delay_mape:.4f}")
    print(f"Area MAPE: {avg_area_mape:.4f} ± {std_area_mape:.4f}")
    print(f"Delay RMSE: {avg_delay_rmse:.4f} ± {std_delay_rmse:.4f}")
    print(f"Area RMSE: {avg_area_rmse:.4f} ± {std_area_rmse:.4f}")
    
    # Save results
    import json
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Results saved to {args.output}")

if __name__ == '__main__':
    main()
