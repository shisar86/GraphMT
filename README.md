# LSOformer: Learning Logic Synthesis Optimization Recipes with Transformers and GNNs

LSOformer is a novel deep learning framework that integrates graph neural networks and transformers to predict logic synthesis outcomes from circuit structures and optimization recipes. It enables automated evaluation of heuristic sequences on And-Inverter Graphs (AIGs), pioneering an end-to-end approach for learning the impact of optimization recipes in electronic design automation.

## Features
- Graph neural network encoding for AIG circuit structures
- Transformer-based sequence modeling for optimization recipes
- End-to-end training and evaluation pipeline

## Getting Started
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Prepare your data in the `data/` directory (see example files).
3. Train the model:
   ```bash
   python main.py --config configs/default.yaml
   ```

## Tools & Technologies
Python, PyTorch, PyTorch Geometric, Transformers, YAML, Pandas

## Project Structure
- `main.py`: Entry point for training
- `src/`: Source code (models, datasets, training)
- `data/`: Circuit and recipe data
- `configs/`: YAML configuration files
- `scripts/`: Utility scripts

## License
MIT License
