import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import time
from tqdm import tqdm
import logging
import sys
sys.path.append('.')
from src.utils.metrics import mean_absolute_percentage_error
from torch_geometric.data import Batch


class LSOformerTrainer:
    def __init__(self, model, dataset, config, device=None):
        """Trainer for LSOformer model"""
        self.model = model
        self.dataset = dataset
        self.config = config
        
        # Set up device
        self.device = device if device is not None else torch.device('cpu')
        self.model.to(self.device)
        
        # Set up optimizer
        self.optimizer = optim.Adam(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        
        # Set up learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=config.lr_patience,
            verbose=True
        )
        
        # Set up loss function
        self.criterion = nn.MSELoss()
        
        # Initialize best validation loss
        self.best_val_loss = float('inf')
        
        # Set up logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger('LSOformerTrainer')
        
        # Create checkpoint directory
        os.makedirs(config.checkpoint_dir, exist_ok=True)
    
    def train_epoch(self, train_indices):

        self.model.train()
        epoch_loss = 0.0
        epoch_mape = 0.0
        
        # Create batches
        num_batches = max(len(train_indices) // self.config.batch_size, 1)
        
        for i in range(num_batches):
            end_idx = min((i + 1) * self.config.batch_size, len(train_indices))
            batch_indices = train_indices[i * self.config.batch_size:end_idx]
            batch = self.dataset.get_batch(batch_indices)
            
            # Create a batched graph from the list of graphs
            graph_batch = Batch.from_data_list(batch['graph_data'])
            graph_batch = graph_batch.to(self.device)
            
            # Move other data to device
            recipe_tokens = batch['recipe_tokens'].to(self.device)
            
            # For node_depths, we need to handle it differently since it's now a list
            # We'll use the node_depths from the batched graph
            node_depths = graph_batch.node_depths
            
            # Get target QoR (using levels as a proxy for delay)
            target_levels = batch['levels'].to(self.device).unsqueeze(-1)
            
            # Forward pass
            self.optimizer.zero_grad()
            predicted_levels = self.model(graph_batch, recipe_tokens, node_depths)
            
            # Calculate loss
            loss = self.criterion(predicted_levels, target_levels)
            
            # Backward pass
            loss.backward()
            
            # Clip gradients
            nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
            
            # Update weights
            self.optimizer.step()
            
            # Calculate metrics
            mape = mean_absolute_percentage_error(predicted_levels.detach(), target_levels)
            
            # Update epoch statistics
            epoch_loss += loss.item()
            epoch_mape += mape.item()
        
        # Calculate average metrics
        avg_loss = epoch_loss / num_batches
        avg_mape = epoch_mape / num_batches
        
        return avg_loss, avg_mape

    
    def validate(self, val_indices):

        self.model.eval()
        val_loss = 0.0
        val_mape = 0.0
        
        # Create batches
        num_batches = max(len(val_indices) // self.config.batch_size, 1)
        
        with torch.no_grad():
            for i in range(num_batches):
                end_idx = min((i + 1) * self.config.batch_size, len(val_indices))
                batch_indices = val_indices[i * self.config.batch_size:end_idx]
                batch = self.dataset.get_batch(batch_indices)
                
                # Create a batched graph from the list of graphs
                from torch_geometric.data import Batch
                graph_batch = Batch.from_data_list(batch['graph_data'])
                graph_batch = graph_batch.to(self.device)
                
                # Move other data to device
                recipe_tokens = batch['recipe_tokens'].to(self.device)
                target_levels = batch['levels'].to(self.device).unsqueeze(-1)
                
                # Forward pass
                predicted_levels = self.model(graph_batch, recipe_tokens)
                
                # Calculate loss
                loss = self.criterion(predicted_levels, target_levels)
                
                # Calculate metrics
                mape = mean_absolute_percentage_error(predicted_levels, target_levels)
                
                # Update validation statistics
                val_loss += loss.item()
                val_mape += mape.item()
        
        # Calculate average metrics
        avg_loss = val_loss / num_batches
        avg_mape = val_mape / num_batches
        
        return avg_loss, avg_mape

    
    def train(self):
        """Train the model for the specified number of epochs"""
        self.logger.info(f"Starting training on {self.device}")
        
        # Split dataset into training and validation sets
        dataset_size = len(self.dataset)
        indices = list(range(dataset_size))
        np.random.shuffle(indices)
        split = int(np.floor(self.config.val_split * dataset_size))
        train_indices, val_indices = indices[split:], indices[:split]
        
        self.logger.info(f"Training set size: {len(train_indices)}")
        self.logger.info(f"Validation set size: {len(val_indices)}")
        
        for epoch in range(self.config.num_epochs):
            start_time = time.time()
            
            # Train for one epoch
            train_loss, train_mape = self.train_epoch(train_indices)
            
            # Validate
            val_loss, val_mape = self.validate(val_indices)
            
            # Update learning rate
            self.scheduler.step(val_loss)
            
            # Log metrics
            epoch_time = time.time() - start_time
            self.logger.info(f"Epoch {epoch+1}/{self.config.num_epochs} - "
                            f"Time: {epoch_time:.2f}s - "
                            f"Train Loss: {train_loss:.4f} - "
                            f"Train MAPE: {train_mape:.4f} - "
                            f"Val Loss: {val_loss:.4f} - "
                            f"Val MAPE: {val_mape:.4f}")
            
            # Save checkpoint if validation loss improved
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                checkpoint_path = os.path.join(self.config.checkpoint_dir, f"best_model.pth")
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_loss': val_loss,
                    'val_mape': val_mape,
                }, checkpoint_path)
                self.logger.info(f"Saved best model checkpoint to {checkpoint_path}")
            
            # Save regular checkpoint
            if (epoch + 1) % self.config.checkpoint_interval == 0:
                checkpoint_path = os.path.join(self.config.checkpoint_dir, f"checkpoint_epoch_{epoch+1}.pth")
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_loss': val_loss,
                    'val_mape': val_mape,
                }, checkpoint_path)
                self.logger.info(f"Saved checkpoint to {checkpoint_path}")
            
            # Early stopping
            if self.config.early_stopping and epoch > self.config.early_stopping_patience:
                # Check if validation loss hasn't improved for patience epochs
                if val_loss > self.best_val_loss:
                    self.logger.info(f"Early stopping triggered after {epoch+1} epochs")
                    break
        
        self.logger.info("Training completed")
        return self.best_val_loss
