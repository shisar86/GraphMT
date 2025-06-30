import torch
import torch.nn as nn
import math

class RecipeEncoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(RecipeEncoder, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        
        # Embedding layer for heuristics
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
    def forward(self, recipe_tokens):
        """Encode recipes (sequences of heuristics)"""
        return self.embedding(recipe_tokens)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        
        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        """Add positional encoding to input embeddings"""
        return x + self.pe[:, :x.size(1)]
