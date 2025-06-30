import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
sys.path.append('.')
from src.models.graph_encoder import GraphEncoder, LevelWiseNodePooling
from src.models.recipe_encoder import RecipeEncoder, PositionalEncoding
from src.models.transformer_decoder import CausalTransformerDecoder

class MLPRegressor(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim=1):
        super(MLPRegressor, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc3 = nn.Linear(hidden_dim // 2, output_dim)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class LSOformer(nn.Module):
    def __init__(self, 
                 node_feature_dim, 
                 heuristic_vocab_size,
                 hidden_dim=32, 
                 embedding_dim=64,
                 num_encoder_layers=2,
                 num_decoder_layers=6,
                 nhead=8,
                 dropout=0.1):
        super(LSOformer, self).__init__()
        
        # Graph encoder for AIG
        self.graph_encoder = GraphEncoder(
            input_dim=node_feature_dim,
            hidden_dim=hidden_dim,
            output_dim=hidden_dim,
            num_layers=num_encoder_layers
        )
        
        # Level-wise pooling for AIG
        self.level_pooling = LevelWiseNodePooling(node_dim=hidden_dim)
        
        # Recipe encoder
        self.recipe_encoder = RecipeEncoder(
            vocab_size=heuristic_vocab_size,
            embedding_dim=2 * hidden_dim  # 2x for concatenated mean and max pooling
        )
        
        # Positional encoding
        self.positional_encoding = PositionalEncoding(d_model=2 * hidden_dim)
        
        # Transformer decoder
        self.transformer_decoder = CausalTransformerDecoder(
            d_model=2 * hidden_dim,
            nhead=nhead,
            num_layers=num_decoder_layers,
            dim_feedforward=4 * hidden_dim,
            dropout=dropout
        )
        
        # MLP regressor for level prediction
        self.level_regressor = MLPRegressor(
            input_dim=2 * hidden_dim,
            hidden_dim=2 * hidden_dim,
            output_dim=1
        )
        
    def forward(self, graph_data, recipe_tokens, node_depths=None):
        """Forward pass of the LSOformer model"""
        batch_size = recipe_tokens.size(0)
        seq_len = recipe_tokens.size(1)
        
        # Encode graph
        node_embeddings = self.graph_encoder(graph_data.x, graph_data.edge_index, 
                                             graph_data.batch if hasattr(graph_data, 'batch') else None)
        
        # Get max depth for each graph in the batch
        if node_depths is None:
            node_depths = graph_data.node_depths
        max_depth = node_depths.max().item()
        
        # Apply level-wise pooling
        aig_embeddings = []
        for i in range(batch_size):
            # For single graph case, use all nodes
            mask = torch.ones(node_embeddings.size(0), dtype=torch.bool, device=node_embeddings.device)
            graph_nodes = node_embeddings[mask]
            graph_depths = node_depths[mask]
            
            # Apply level-wise pooling
            graph_level_emb = self.level_pooling(graph_nodes, graph_depths, max_depth)
            aig_embeddings.append(graph_level_emb)
        
        # Stack to get [batch_size, max_depth+1, 2*hidden_dim]
        aig_embeddings = torch.stack(aig_embeddings, dim=0)
        
        # Encode recipes
        recipe_embeddings = self.recipe_encoder(recipe_tokens)  # [batch_size, seq_len, 2*hidden_dim]
        
        # Add positional encoding
        recipe_embeddings = self.positional_encoding(recipe_embeddings)
        
        # Prepare for transformer decoder
        # Transformer expects [seq_len, batch_size, dim]
        recipe_embeddings = recipe_embeddings.transpose(0, 1)
        aig_embeddings = aig_embeddings.transpose(0, 1)
        
        # Apply transformer decoder
        decoder_output = self.transformer_decoder(
            tgt=recipe_embeddings,
            memory=aig_embeddings
        )  # [seq_len, batch_size, 2*hidden_dim]
        
        # Convert back to [batch_size, seq_len, 2*hidden_dim]
        decoder_output = decoder_output.transpose(0, 1)
        
        # Use the final token's representation to predict the level
        final_representation = decoder_output[:, -1, :]  # [batch_size, 2*hidden_dim]
        
        # Predict level
        level_prediction = self.level_regressor(final_representation)  # [batch_size, 1]
        
        return level_prediction
