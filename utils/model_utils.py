#!/usr/bin/env python3
"""
Model Utilities for Drone Data Leakage Classification
Authors: Anas AlSobeh (Southern Illinois University Carbondale)
         Omar Darwish (Eastern Michigan University)

This module provides utilities for GPT model handling, training, and evaluation.
"""

import torch
import torch.nn as nn
from torch.optim import AdamW
from transformers import (
    GPT2LMHeadModel, GPT2Tokenizer, GPT2Config,
    AutoTokenizer, AutoModelForSequenceClassification,
    TrainingArguments, Trainer, EarlyStoppingCallback
)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
import numpy as np
import logging
from typing import Dict, Any, Tuple, Optional
import yaml
import os

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GPTClassifier(nn.Module):
    """
    GPT-based classifier for drone data leakage classification.
    """
    
    def __init__(self, model_name: str = "gpt2", num_labels: int = 2, dropout_rate: float = 0.1):
        """
        Initialize the GPT classifier.
        
        Args:
            model_name (str): Pre-trained model name
            num_labels (int): Number of classification labels
            dropout_rate (float): Dropout rate
        """
        super(GPTClassifier, self).__init__()
        
        self.model_name = model_name
        self.num_labels = num_labels
        
        # Load pre-trained GPT model and tokenizer
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load model configuration
        config = GPT2Config.from_pretrained(model_name)
        config.num_labels = num_labels
        config.pad_token_id = self.tokenizer.eos_token_id
        
        # Load the base model
        self.gpt_model = GPT2LMHeadModel.from_pretrained(model_name, config=config)
        
        # Add classification head
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(config.hidden_size, num_labels)
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        """Initialize the weights of the classification head."""
        nn.init.normal_(self.classifier.weight, std=0.02)
        nn.init.zeros_(self.classifier.bias)
    
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, 
                labels: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Forward pass of the model.
        
        Args:
            input_ids (torch.Tensor): Input token IDs
            attention_mask (torch.Tensor): Attention mask
            labels (torch.Tensor, optional): Target labels
            
        Returns:
            Dict[str, torch.Tensor]: Model outputs
        """
        # Get GPT outputs
        outputs = self.gpt_model.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        # Get the last hidden state
        last_hidden_state = outputs.last_hidden_state
        
        # Pool the sequence (use the last token's representation)
        batch_size = last_hidden_state.size(0)
        sequence_lengths = attention_mask.sum(dim=1) - 1
        pooled_output = last_hidden_state[range(batch_size), sequence_lengths]
        
        # Apply dropout and classification
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        
        outputs = {"logits": logits}
        
        # Calculate loss if labels are provided
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits, labels)
            outputs["loss"] = loss
        
        return outputs

class ModelTrainer:
    """
    Trainer class for GPT-based classification model.
    """
    
    def __init__(self, config_path: str = "config.yaml"):
        """
        Initialize the model trainer.
        
        Args:
            config_path (str): Path to configuration file
        """
        self.config = self._load_config(config_path)
        self.device = self._get_device()
        self.model = None
        self.tokenizer = None
        
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            logger.warning(f"Config file {config_path} not found. Using default configuration.")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration."""
        return {
            'model': {
                'pretrained_model': 'gpt2',
                'num_labels': 2,
                'learning_rate': 2e-5,
                'batch_size': 16,
                'num_epochs': 10
            }
        }
    
    def _get_device(self) -> torch.device:
        """Get the appropriate device for training."""
        if torch.cuda.is_available():
            device = torch.device("cuda")
            logger.info(f"Using GPU: {torch.cuda.get_device_name()}")
        else:
            device = torch.device("cpu")
            logger.info("Using CPU")
        return device
    
    def initialize_model(self) -> GPTClassifier:
        """
        Initialize the GPT classifier model.
        
        Returns:
            GPTClassifier: Initialized model
        """
        model_config = self.config['model']
        
        self.model = GPTClassifier(
            model_name=model_config['pretrained_model'],
            num_labels=model_config['num_labels']
        )
        
        self.tokenizer = self.model.tokenizer
        self.model.to(self.device)
        
        logger.info(f"Model initialized: {model_config['pretrained_model']}")
        logger.info(f"Number of parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        return self.model
    
    def evaluate_zero_shot(self, test_loader) -> Dict[str, float]:
        """
        Evaluate the model in zero-shot setting (before fine-tuning).
        
        Args:
            test_loader: Test data loader
            
        Returns:
            Dict[str, float]: Evaluation metrics
        """
        if self.model is None:
            self.initialize_model()
        
        self.model.eval()
        all_predictions = []
        all_labels = []
        all_probabilities = []
        
        logger.info("Evaluating zero-shot performance...")
        
        with torch.no_grad():
            for batch in test_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs['logits']
                
                # Get predictions and probabilities
                probabilities = torch.softmax(logits, dim=-1)
                predictions = torch.argmax(logits, dim=-1)
                
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())
        
        # Calculate metrics
        metrics = self._calculate_metrics(all_labels, all_predictions, all_probabilities)
        
        logger.info("Zero-shot evaluation completed:")
        for metric, value in metrics.items():
            logger.info(f"  {metric}: {value:.4f}")
        
        return metrics
    
    def fine_tune_model(self, train_loader, val_loader, output_dir: str = "models/fine_tuned") -> None:
        """
        Fine-tune the GPT model on the drone classification task.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            output_dir (str): Output directory for saving the model
        """
        if self.model is None:
            self.initialize_model()
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=self.config['model']['num_epochs'],
            per_device_train_batch_size=self.config['model']['batch_size'],
            per_device_eval_batch_size=self.config['model']['batch_size'],
            learning_rate=self.config['model']['learning_rate'],
            weight_decay=0.01,
            logging_dir=f"{output_dir}/logs",
            logging_steps=100,
            eval_steps=500,
            save_steps=1000,
            evaluation_strategy="steps",
            save_strategy="steps",
            load_best_model_at_end=True,
            metric_for_best_model="eval_accuracy",
            greater_is_better=True,
            warmup_steps=500,
            gradient_accumulation_steps=1,
            dataloader_num_workers=2,
            remove_unused_columns=False,
        )
        
        # Initialize trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_loader.dataset,
            eval_dataset=val_loader.dataset,
            compute_metrics=self._compute_metrics_for_trainer,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
        )
        
        logger.info("Starting fine-tuning...")
        
        # Train the model
        trainer.train()
        
        # Save the final model
        trainer.save_model(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        
        logger.info(f"Fine-tuning completed. Model saved to {output_dir}")
    
    def evaluate_fine_tuned(self, test_loader, model_path: str = None) -> Dict[str, float]:
        """
        Evaluate the fine-tuned model.
        
        Args:
            test_loader: Test data loader
            model_path (str, optional): Path to fine-tuned model
            
        Returns:
            Dict[str, float]: Evaluation metrics
        """
        if model_path:
            # Load fine-tuned model
            self.model = GPTClassifier.from_pretrained(model_path)
            self.tokenizer = GPT2Tokenizer.from_pretrained(model_path)
            self.model.to(self.device)
        
        if self.model is None:
            raise ValueError("No model available for evaluation")
        
        self.model.eval()
        all_predictions = []
        all_labels = []
        all_probabilities = []
        
        logger.info("Evaluating fine-tuned model...")
        
        with torch.no_grad():
            for batch in test_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs['logits']
                
                # Get predictions and probabilities
                probabilities = torch.softmax(logits, dim=-1)
                predictions = torch.argmax(logits, dim=-1)
                
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())
        
        # Calculate metrics
        metrics = self._calculate_metrics(all_labels, all_predictions, all_probabilities)
        
        logger.info("Fine-tuned model evaluation completed:")
        for metric, value in metrics.items():
            logger.info(f"  {metric}: {value:.4f}")
        
        return metrics
    
    def _calculate_metrics(self, labels: list, predictions: list, probabilities: list) -> Dict[str, float]:
        """
        Calculate evaluation metrics.
        
        Args:
            labels (list): True labels
            predictions (list): Predicted labels
            probabilities (list): Prediction probabilities
            
        Returns:
            Dict[str, float]: Calculated metrics
        """
        accuracy = accuracy_score(labels, predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='weighted')
        
        # Calculate ROC AUC
        probabilities = np.array(probabilities)
        if probabilities.shape[1] == 2:  # Binary classification
            roc_auc = roc_auc_score(labels, probabilities[:, 1])
        else:
            roc_auc = roc_auc_score(labels, probabilities, multi_class='ovr')
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'roc_auc': roc_auc
        }
    
    def _compute_metrics_for_trainer(self, eval_pred) -> Dict[str, float]:
        """
        Compute metrics for the Hugging Face trainer.
        
        Args:
            eval_pred: Evaluation predictions
            
        Returns:
            Dict[str, float]: Computed metrics
        """
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        
        accuracy = accuracy_score(labels, predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='weighted')
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
    
    def save_model(self, model_path: str) -> None:
        """
        Save the current model.
        
        Args:
            model_path (str): Path to save the model
        """
        if self.model is None:
            raise ValueError("No model to save")
        
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        torch.save(self.model.state_dict(), model_path)
        logger.info(f"Model saved to {model_path}")
    
    def load_model(self, model_path: str) -> None:
        """
        Load a saved model.
        
        Args:
            model_path (str): Path to the saved model
        """
        if self.model is None:
            self.initialize_model()
        
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        logger.info(f"Model loaded from {model_path}")

