import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class GraphEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=2):
        super(GraphEncoder, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        
        # GCN layers
        self.layers = nn.ModuleList()
        self.layers.append(GCNConv(input_dim, hidden_dim))
        for _ in range(num_layers - 2):
            self.layers.append(GCNConv(hidden_dim, hidden_dim))
        self.layers.append(GCNConv(hidden_dim, output_dim))
    
    def forward(self, x, edge_index, batch=None):
        """Forward pass through the graph encoder"""
        for i, layer in enumerate(self.layers):
            x = layer(x, edge_index)
            if i < len(self.layers) - 1:
                x = F.relu(x)
        
        return x

class LevelWiseNodePooling(nn.Module):
    def __init__(self, node_dim):
        super(LevelWiseNodePooling, self).__init__()
        self.node_dim = node_dim
        
    def forward(self, node_embeddings, node_depths, max_depth):
        """Performs level-wise pooling of node embeddings based on their depths in the DAG"""
        level_embeddings = []
        
        for level in range(max_depth + 1):
            # Get nodes at current level
            mask = (node_depths == level)
            if mask.sum() > 0:
                level_nodes = node_embeddings[mask]
                
                # Apply mean and max pooling
                mean_pool = torch.mean(level_nodes, dim=0)
                max_pool, _ = torch.max(level_nodes, dim=0)
                
                # Concatenate the pooled embeddings
                level_emb = torch.cat([mean_pool, max_pool], dim=0)
            else:
                # If no nodes at this level, create a zero embedding
                level_emb = torch.zeros(2 * self.node_dim, device=node_embeddings.device)
            
            level_embeddings.append(level_emb)
        
        # Stack to create sequence
        level_embeddings = torch.stack(level_embeddings, dim=0)
        
        return level_embeddings
