import torch
from torch.utils.data import Dataset
import os
import pandas as pd
import sys
sys.path.append('.')
from src.utils.data_utils import create_graph_from_bench, tokenize_heuristics, pad_sequences

class MultiAIGDataset(Dataset):
    def __init__(self, bench_dir, recipe_file, transform=None):
        """
        Dataset for multiple AIG graphs and recipes.

        Args:
            bench_dir: Directory containing all .bench files
            recipe_file: Path to the recipe CSV file (with 'design' column)
            transform: Optional transform to be applied to samples
        """
        self.bench_dir = bench_dir
        self.recipe_file = recipe_file
        self.transform = transform

        # Load all .bench files
        self.design_to_path = {fname: os.path.join(bench_dir, fname)
                               for fname in os.listdir(bench_dir) if fname.endswith('.bench')}
        # Preload all graphs
        self.graphs = {design: create_graph_from_bench(path)[0]
                       for design, path in self.design_to_path.items()}

        # Load recipes
        df = pd.read_csv(recipe_file)
        self.samples = []
        all_recipes = []
        for _, row in df.iterrows():
            design = row['design']
            recipe_str = row['recipe'].strip('"')
            heuristics = [h.strip() for h in recipe_str.split(';')]
            levels = row['levels']
            self.samples.append({'design': design, 'heuristics': heuristics, 'levels': levels})
            all_recipes.append(heuristics)

        # Tokenize all heuristics
        self.vocab, self.tokenized_recipes = tokenize_heuristics(all_recipes)
        # Assign tokenized recipes to samples
        for i, sample in enumerate(self.samples):
            sample['tokenized'] = self.tokenized_recipes[i]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        design = sample['design']
        graph_data = self.graphs[design]
        recipe_tokens = torch.tensor(sample['tokenized'], dtype=torch.long)
        levels = float(sample['levels'])
        out = {
            'graph_data': graph_data,
            'recipe_tokens': recipe_tokens,
            'node_depths': graph_data.node_depths,
            'levels': torch.tensor([levels], dtype=torch.float),
            'design': design
        }
        if self.transform:
            out = self.transform(out)
        return out

    def get_vocab_size(self):
        return len(self.vocab)

    def get_batch(self, indices):
        samples = [self.__getitem__(idx) for idx in indices]
        batch = {
            'graph_data': [s['graph_data'] for s in samples],  # List of graphs!
            'recipe_tokens': pad_sequences([s['recipe_tokens'] for s in samples]),
            'node_depths': [s['node_depths'] for s in samples],
            'levels': torch.cat([s['levels'] for s in samples], dim=0),
            'design': [s['design'] for s in samples]
        }
        return batch
