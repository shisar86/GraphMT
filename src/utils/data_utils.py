import os
import torch
import pandas as pd
import numpy as np
from torch_geometric.data import Data

def parse_bench_file(bench_file_path):
    """Parse a .bench file and create a graph representation"""
    nodes = []
    edges = []
    node_features = {}
    inputs = []
    outputs = []
    
    with open(bench_file_path, 'r') as f:
        lines = f.readlines()
    
    # Process each line
    for line in lines:
        line = line.strip()
        
        # Skip comments and empty lines
        if not line or line.startswith('#'):
            continue
        
        # Process INPUT
        if line.startswith('INPUT('):
            node = line[6:-1].strip()  # Extract node name
            if node not in nodes:
                nodes.append(node)
            node_features[node] = {'type': 0, 'inverted': 0}  # Input node
            inputs.append(node)
            
        # Process OUTPUT
        elif line.startswith('OUTPUT('):
            node = line[7:-1].strip()  # Extract node name
            if node not in nodes:
                nodes.append(node)
            node_features[node] = {'type': 1, 'inverted': 0}  # Output node
            outputs.append(node)
            
        # Process gates
        elif '=' in line:
            parts = line.split('=')
            output_node = parts[0].strip()
            gate_expr = parts[1].strip()
            
            if output_node not in nodes:
                nodes.append(output_node)
                node_features[output_node] = {'type': 2, 'inverted': 0}  # Intermediate node
            
            # Process AND gates
            if 'AND(' in gate_expr:
                input_nodes = gate_expr[gate_expr.find('(')+1:gate_expr.find(')')].split(',')
                input_nodes = [n.strip() for n in input_nodes]
                
                for input_node in input_nodes:
                    # Check if the input is inverted
                    if input_node.startswith('NOT('):
                        actual_node = input_node[4:-1].strip()
                        if actual_node not in nodes:
                            nodes.append(actual_node)
                            node_features[actual_node] = {'type': 2, 'inverted': 0}
                        edges.append((actual_node, output_node))
                        node_features[output_node]['inverted'] += 1
                    else:
                        if input_node not in nodes:
                            nodes.append(input_node)
                            node_features[input_node] = {'type': 2, 'inverted': 0}
                        edges.append((input_node, output_node))
            
            # Process NOT gates
            elif 'NOT(' in gate_expr:
                input_node = gate_expr[4:-1].strip()
                if input_node not in nodes:
                    nodes.append(input_node)
                    node_features[input_node] = {'type': 2, 'inverted': 0}
                edges.append((input_node, output_node))
                node_features[output_node]['inverted'] += 1
    
    return nodes, edges, node_features, inputs, outputs

def create_graph_from_bench(bench_file_path):
    """Create a PyTorch Geometric graph from a .bench file"""
    nodes, edges, node_features, inputs, outputs = parse_bench_file(bench_file_path)
    
    # Create node mapping
    node_to_idx = {node: i for i, node in enumerate(nodes)}
    
    # Create edge index
    edge_index = []
    for src, dst in edges:
        edge_index.append([node_to_idx[src], node_to_idx[dst]])
    
    if not edge_index:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
    else:
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    
    # Create node features
    x = []
    for node in nodes:
        features = node_features[node]
        x.append([features['type'], features['inverted']])
    
    x = torch.tensor(x, dtype=torch.float)
    
    # Calculate node depths
    node_depths = calculate_node_depths(edge_index, len(nodes))
    
    # Create graph data
    graph_data = Data(x=x, edge_index=edge_index, node_depths=node_depths)
    
    return graph_data, nodes, inputs, outputs

def calculate_node_depths(edge_index, num_nodes):
    """Calculate the depth of each node in the DAG"""
    # Create adjacency list
    adj_list = [[] for _ in range(num_nodes)]
    for i in range(edge_index.size(1)):
        src, dst = edge_index[0, i].item(), edge_index[1, i].item()
        adj_list[dst].append(src)
    
    # Initialize depths
    depths = torch.zeros(num_nodes, dtype=torch.long)
    visited = [False] * num_nodes
    
    # DFS to calculate depths
    def dfs(node):
        if visited[node]:
            return depths[node]
        
        visited[node] = True
        
        if not adj_list[node]:  # Input node
            depths[node] = 0
            return 0
        
        max_depth = 0
        for parent in adj_list[node]:
            max_depth = max(max_depth, dfs(parent) + 1)
        
        depths[node] = max_depth
        return max_depth
    
    # Calculate depths for all nodes
    for node in range(num_nodes):
        if not visited[node]:
            dfs(node)
    
    return depths

def parse_recipe_csv(csv_file_path):
    """Parse recipe data from CSV file"""
    recipes = []
    stats = {}
    
    df = pd.read_csv(csv_file_path)
    
    for _, row in df.iterrows():
        design = row.get('design', None)
        recipe_str = row.get('recipe', None)
        nodes = row.get('nodes', None)
        levels = row.get('levels', None)
        iterations = row.get('iterations', None)
        
        if recipe_str:
            # Clean and parse recipe
            recipe_str = recipe_str.strip('"')
            heuristics = [h.strip() for h in recipe_str.split(';')]
            
            recipes.append(heuristics)
            stats[tuple(heuristics)] = {
                'design': design,
                'nodes': nodes,
                'levels': levels,
                'iterations': iterations
            }
    
    return recipes, stats

def tokenize_heuristics(recipes):
    """Tokenize heuristics in recipes"""
    # Extract unique heuristics
    unique_heuristics = set()
    for recipe in recipes:
        for heuristic in recipe:
            unique_heuristics.add(heuristic)
    
    # Create vocabulary
    vocab = {heuristic: i for i, heuristic in enumerate(sorted(unique_heuristics))}
    
    # Tokenize recipes
    tokenized_recipes = []
    for recipe in recipes:
        tokenized_recipe = [vocab[heuristic] for heuristic in recipe]
        tokenized_recipes.append(tokenized_recipe)
    
    return vocab, tokenized_recipes

def pad_sequences(sequences, max_len=None, padding_value=0):
    """
    Pad sequences to the same length

    Args:
        sequences: List of sequences (can be tensors or lists)
        max_len: Maximum length (if None, use the length of the longest sequence)
        padding_value: Value to use for padding

    Returns:
        padded_sequences: Tensor of padded sequences
    """
    if max_len is None:
        max_len = max(len(seq) for seq in sequences)
    padded_sequences = []
    for seq in sequences:
        if isinstance(seq, torch.Tensor):
            seq = seq.tolist()
        padded_seq = seq + [padding_value] * (max_len - len(seq))
        padded_sequences.append(padded_seq)
    return torch.tensor(padded_sequences)
