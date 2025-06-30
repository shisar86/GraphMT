import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import os
import time
from tqdm import tqdm
import logging

from ..utils.metrics import mean_absolute_percentage_error

class SSLTrainer:
    def __init__(self, model, train_dataset, val_dataset, config):
        """
        Self-Supervised Learning Trainer for LSOformer model
        
        Args:
            model: LSOformer model
            train_dataset: Training dataset
            val_dataset: Validation dataset
            config: Training configuration
        """
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.config = config
        
        # Create data loaders
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=config.num_workers,
            pin_memory=True
        )
        
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.num_workers,
            pin_memory=True
        )
        
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
        
        # Set up device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        # Set up logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger('SSLTrainer')
        
        # Create checkpoint directory
        os.makedirs(config.checkpoint_dir, exist_ok=True)
    
    def train_epoch(self):
        """Train for one epoch with SSL auxiliary task"""
        self.model.train()
        epoch_loss = 0.0
        epoch_final_mape = 0.0
        epoch_trajectory_mape = 0.0
        
        for batch in tqdm(self.train_loader, desc="Training"):
            # Move data to device
            graph_data = batch['graph_data'].to(self.device)
            recipe_tokens = batch['recipe_tokens'].to(self.device)
            node_depths = batch['node_depths'].to(self.device)
            
            # Get target QoR (using delay as default)
            target_qor_trajectory = batch['delay_trajectory'].to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            predicted_qor = self.model(graph_data, recipe_tokens, node_depths)
            
            # Calculate loss for the entire trajectory (SSL auxiliary task)
            trajectory_loss = self.criterion(predicted_qor, target_qor_trajectory)
            
            # Calculate loss for the final QoR (main task)
            final_qor_pred = predicted_qor[:, -1]
            final_qor_target = target_qor_trajectory[:, -1]
            final_loss = self.criterion(final_qor_pred, final_qor_target)
            
            # Combined loss
            loss = self.config.ssl_weight * trajectory_loss + (1 - self.config.ssl_weight) * final_loss
            
            # Backward pass
            loss.backward()
            
            # Clip gradients
            nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
            
            # Update weights
            self.optimizer.step()
            
            # Calculate metrics
            final_mape = mean_absolute_percentage_error(final_qor_pred.detach(), final_qor_target)
            trajectory_mape = mean_absolute_percentage_error(predicted_qor.detach(), target_qor_trajectory)
            
            # Update epoch statistics
            epoch_loss += loss.item()
            epoch_final_mape += final_mape.item()
            epoch_trajectory_mape += trajectory_mape.item()
        
        # Calculate average metrics
        avg_loss = epoch_loss / len(self.train_loader)
        avg_final_mape = epoch_final_mape / len(self.train_loader)
        avg_trajectory_mape = epoch_trajectory_mape / len(self.train_loader)
        
        return avg_loss, avg_final_mape, avg_trajectory_mape
    
    def validate(self):
        """Validate the model with SSL auxiliary task"""
        self.model.eval()
        val_loss = 0.0
        val_final_mape = 0.0
        val_trajectory_mape = 0.0
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validation"):
                # Move data to device
                graph_data = batch['graph_data'].to(self.device)
                recipe_tokens = batch['recipe_tokens'].to(self.device)
                node_depths = batch['node_depths'].to(self.device)
                
                # Get target QoR (using delay as default)
                target_qor_trajectory = batch['delay_trajectory'].to(self.device)
                
                # Forward pass
                predicted_qor = self.model(graph_data, recipe_tokens, node_depths)
                
                # Calculate loss for the entire trajectory (SSL auxiliary task)
                trajectory_loss = self.criterion(predicted_qor, target_qor_trajectory)
                
                # Calculate loss for the final QoR (main task)
                final_qor_pred = predicted_qor[:, -1]
                final_qor_target = target_qor_trajectory[:, -1]
                final_loss = self.criterion(final_qor_pred, final_qor_target)
                
                # Combined loss
                loss = self.config.ssl_weight * trajectory_loss + (1 - self.config.ssl_weight) * final_loss
                
                # Calculate metrics
                final_mape = mean_absolute_percentage_error(final_qor_pred, final_qor_target)
                trajectory_mape = mean_absolute_percentage_error(predicted_qor, target_qor_trajectory)
                
                # Update validation statistics
                val_loss += loss.item()
                val_final_mape += final_mape.item()
                val_trajectory_mape += trajectory_mape.item()
        
        # Calculate average metrics
        avg_loss = val_loss / len(self.val_loader)
        avg_final_mape = val_final_mape / len(self.val_loader)
        avg_trajectory_mape = val_trajectory_mape / len(self.val_loader)
        
        return avg_loss, avg_final_mape, avg_trajectory_mape
    
    def train(self):
        """Train the model with SSL for the specified number of epochs"""
        self.logger.info(f"Starting SSL training on {self.device}")
        self.logger.info(f"Training set size: {len(self.train_dataset)}")
        self.logger.info(f"Validation set size: {len(self.val_dataset)}")
        
        for epoch in range(self.config.num_epochs):
            start_time = time.time()
            
            # Train for one epoch
            train_loss, train_final_mape, train_trajectory_mape = self.train_epoch()
            
            # Validate
            val_loss, val_final_mape, val_trajectory_mape = self.validate()
            
            # Update learning rate
            self.scheduler.step(val_loss)
            
            # Log metrics
            epoch_time = time.time() - start_time
            self.logger.info(f"Epoch {epoch+1}/{self.config.num_epochs} - "
                            f"Time: {epoch_time:.2f}s - "
                            f"Train Loss: {train_loss:.4f} - "
                            f"Train Final MAPE: {train_final_mape:.4f} - "
                            f"Train Trajectory MAPE: {train_trajectory_mape:.4f} - "
                            f"Val Loss: {val_loss:.4f} - "
                            f"Val Final MAPE: {val_final_mape:.4f} - "
                            f"Val Trajectory MAPE: {val_trajectory_mape:.4f}")
            
            # Save checkpoint if validation loss improved
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                checkpoint_path = os.path.join(self.config.checkpoint_dir, f"best_ssl_model.pth")
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_loss': val_loss,
                    'val_final_mape': val_final_mape,
                    'val_trajectory_mape': val_trajectory_mape,
                }, checkpoint_path)
                self.logger.info(f"Saved best SSL model checkpoint to {checkpoint_path}")
            
            # Save regular checkpoint
            if (epoch + 1) % self.config.checkpoint_interval == 0:
                checkpoint_path = os.path.join(self.config.checkpoint_dir, f"ssl_checkpoint_epoch_{epoch+1}.pth")
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_loss': val_loss,
                    'val_final_mape': val_final_mape,
                    'val_trajectory_mape': val_trajectory_mape,
                }, checkpoint_path)
                self.logger.info(f"Saved SSL checkpoint to {checkpoint_path}")
            
            # Early stopping
            if self.config.early_stopping and epoch > self.config.early_stopping_patience:
                # Check if validation loss hasn't improved for patience epochs
                if val_loss > self.best_val_loss:
                    self.logger.info(f"Early stopping triggered after {epoch+1} epochs")
                    break
        
        self.logger.info("SSL training completed")
        return self.best_val_loss
