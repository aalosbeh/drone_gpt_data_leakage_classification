#!/usr/bin/env python3

import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from transformers import (
    GPT2Tokenizer, GPT2LMHeadModel, GPT2Config,
    TrainingArguments, Trainer, EarlyStoppingCallback
)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
import logging
import json
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

# Add utils to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from utils.data_utils import DroneDataset
from utils.eval_utils import ModelEvaluator

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GPTFineTuner:
    """
    GPT Fine-tuning implementation for drone data classification.
    """
    
    def __init__(self, model_name: str = "gpt2", num_labels: int = 2):
        """
        Initialize the fine-tuner.
        
        Args:
            model_name (str): Pre-trained model name
            num_labels (int): Number of classification labels
        """
        self.model_name = model_name
        self.num_labels = num_labels
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = None
        self.model = None
        
        logger.info(f"Initializing GPT Fine-tuner with {model_name}")
        logger.info(f"Using device: {self.device}")
    
    def load_model(self):
        """Load and prepare the model for fine-tuning."""
        logger.info(f"Loading model: {self.model_name}")
        
        # Load tokenizer
        self.tokenizer = GPT2Tokenizer.from_pretrained(self.model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load model configuration
        config = GPT2Config.from_pretrained(self.model_name)
        config.num_labels = self.num_labels
        config.pad_token_id = self.tokenizer.eos_token_id
        
        # Load base model
        self.model = GPT2LMHeadModel.from_pretrained(self.model_name, config=config)
        
        # Add classification head
        self.model.score = nn.Linear(config.hidden_size, self.num_labels)
        
        # Initialize classification head weights
        nn.init.normal_(self.model.score.weight, std=0.02)
        nn.init.zeros_(self.model.score.bias)
        
        self.model.to(self.device)
        
        logger.info(f"Model loaded successfully")
        logger.info(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        return self.model
    
    def prepare_datasets(self, X_train: np.ndarray, X_val: np.ndarray, X_test: np.ndarray,
                        y_train: np.ndarray, y_val: np.ndarray, y_test: np.ndarray,
                        max_length: int = 256):
        """
        Prepare datasets for training.
        
        Args:
            X_train, X_val, X_test: Feature arrays
            y_train, y_val, y_test: Label arrays
            max_length (int): Maximum sequence length
            
        Returns:
            Tuple: (train_dataset, val_dataset, test_dataset)
        """
        logger.info("Preparing datasets for fine-tuning...")
        
        # Create datasets
        train_dataset = DroneDataset(X_train, y_train, self.tokenizer, max_length)
        val_dataset = DroneDataset(X_val, y_val, self.tokenizer, max_length)
        test_dataset = DroneDataset(X_test, y_test, self.tokenizer, max_length)
        
        logger.info(f"Datasets prepared:")
        logger.info(f"  Training: {len(train_dataset)} samples")
        logger.info(f"  Validation: {len(val_dataset)} samples")
        logger.info(f"  Test: {len(test_dataset)} samples")
        
        return train_dataset, val_dataset, test_dataset
    
    def compute_metrics(self, eval_pred):
        """
        Compute metrics for evaluation.
        
        Args:
            eval_pred: Evaluation predictions from trainer
            
        Returns:
            Dict: Computed metrics
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
    
    def fine_tune(self, train_dataset, val_dataset, output_dir: str = "models/fine_tuned",
                  num_epochs: int = 3, batch_size: int = 8, learning_rate: float = 2e-5):
        """
        Fine-tune the GPT model.
        
        Args:
            train_dataset: Training dataset
            val_dataset: Validation dataset
            output_dir (str): Output directory
            num_epochs (int): Number of training epochs
            batch_size (int): Batch size
            learning_rate (float): Learning rate
        """
        if self.model is None:
            self.load_model()
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            learning_rate=learning_rate,
            weight_decay=0.01,
            logging_dir=f"{output_dir}/logs",
            logging_steps=50,
            eval_steps=100,
            save_steps=200,
            eval_strategy="steps",
            save_strategy="steps",
            load_best_model_at_end=True,
            metric_for_best_model="eval_accuracy",
            greater_is_better=True,
            warmup_steps=100,
            gradient_accumulation_steps=2,
            dataloader_num_workers=2,
            remove_unused_columns=False,
            report_to=None,  # Disable wandb
        )
        
        # Custom trainer class for classification
        class ClassificationTrainer(Trainer):
            def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
                labels = inputs.pop("labels")
                outputs = model(**inputs)
                
                # Get last hidden state and pool
                last_hidden_state = outputs.last_hidden_state
                batch_size = last_hidden_state.size(0)
                sequence_lengths = inputs['attention_mask'].sum(dim=1) - 1
                pooled_output = last_hidden_state[range(batch_size), sequence_lengths]
                
                # Classification
                logits = model.score(pooled_output)
                
                # Compute loss
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(logits, labels)
                
                return (loss, {"logits": logits}) if return_outputs else loss
        
        # Initialize trainer
        trainer = ClassificationTrainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=self.compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
        )
        
        logger.info("Starting fine-tuning...")
        
        # Train the model
        train_result = trainer.train()
        
        # Save the model
        trainer.save_model(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        
        # Save training metrics
        with open(f"{output_dir}/training_results.json", 'w') as f:
            json.dump(train_result.metrics, f, indent=2)
        
        logger.info(f"Fine-tuning completed. Model saved to {output_dir}")
        
        return train_result
    
    def evaluate_model(self, test_dataset, model_path: str = None):
        """
        Evaluate the fine-tuned model.
        
        Args:
            test_dataset: Test dataset
            model_path (str, optional): Path to fine-tuned model
            
        Returns:
            Dict: Evaluation results
        """
        if model_path and os.path.exists(model_path):
            # Load fine-tuned model
            logger.info(f"Loading fine-tuned model from {model_path}")
            self.model = GPT2LMHeadModel.from_pretrained(model_path)
            self.tokenizer = GPT2Tokenizer.from_pretrained(model_path)
            self.model.to(self.device)
        
        if self.model is None:
            raise ValueError("No model available for evaluation")
        
        self.model.eval()
        
        # Create data loader
        test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)
        
        all_predictions = []
        all_labels = []
        all_probabilities = []
        
        logger.info("Evaluating fine-tuned model...")
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(test_loader):
                if batch_idx % 10 == 0:
                    logger.info(f"Processing batch {batch_idx + 1}/{len(test_loader)}")
                
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                # Forward pass
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                
                # Get pooled representation
                last_hidden_state = outputs.last_hidden_state
                batch_size = last_hidden_state.size(0)
                sequence_lengths = attention_mask.sum(dim=1) - 1
                pooled_output = last_hidden_state[range(batch_size), sequence_lengths]
                
                # Classification
                logits = self.model.score(pooled_output)
                
                # Get predictions and probabilities
                probabilities = torch.softmax(logits, dim=-1)
                predictions = torch.argmax(logits, dim=-1)
                
                all_predictions.extend(predictions.detach().cpu().numpy())
                all_labels.extend(labels.detach().cpu().numpy())
                all_probabilities.extend(probabilities.detach().cpu().numpy())
        
        # Calculate metrics
        accuracy = accuracy_score(all_labels, all_predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_predictions, average='weighted')
        
        # ROC AUC
        probabilities_array = np.array(all_probabilities)
        roc_auc = roc_auc_score(all_labels, probabilities_array[:, 1])
        
        results = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'roc_auc': roc_auc,
            'predictions': all_predictions,
            'labels': all_labels,
            'probabilities': all_probabilities
        }
        
        logger.info("Fine-tuned model evaluation completed:")
        logger.info(f"  Accuracy: {accuracy:.4f}")
        logger.info(f"  Precision: {precision:.4f}")
        logger.info(f"  Recall: {recall:.4f}")
        logger.info(f"  F1-score: {f1:.4f}")
        logger.info(f"  ROC AUC: {roc_auc:.4f}")
        
        return results

def run_complete_pipeline():
    """Run the complete fine-tuning and evaluation pipeline."""
    logger.info("=" * 80)
    logger.info("GPT FINE-TUNING AND POST-TUNING EVALUATION")
    logger.info("=" * 80)
    
    # Load preprocessed data
    logger.info("Loading preprocessed data...")
    X_train = np.load("data/X_train.npy")
    X_val = np.load("data/X_val.npy")
    X_test = np.load("data/X_test.npy")
    y_train = np.load("data/y_train.npy")
    y_val = np.load("data/y_val.npy")
    y_test = np.load("data/y_test.npy")
    
    logger.info(f"Data loaded:")
    logger.info(f"  Training: {X_train.shape}")
    logger.info(f"  Validation: {X_val.shape}")
    logger.info(f"  Test: {X_test.shape}")
    
    # Initialize fine-tuner
    fine_tuner = GPTFineTuner("gpt2", num_labels=2)
    
    # Prepare datasets
    train_dataset, val_dataset, test_dataset = fine_tuner.prepare_datasets(
        X_train, X_val, X_test, y_train, y_val, y_test
    )
    
    # Fine-tune the model
    logger.info("\n" + "=" * 50)
    logger.info("FINE-TUNING PHASE")
    logger.info("=" * 50)
    
    train_result = fine_tuner.fine_tune(
        train_dataset, val_dataset,
        output_dir="models/fine_tuned",
        num_epochs=3,
        batch_size=8,
        learning_rate=2e-5
    )
    
    # Evaluate fine-tuned model
    logger.info("\n" + "=" * 50)
    logger.info("POST-TUNING EVALUATION")
    logger.info("=" * 50)
    
    post_tuning_results = fine_tuner.evaluate_model(test_dataset, "models/fine_tuned")
    
    # Save results
    os.makedirs("../outputs", exist_ok=True)
    with open("outputs/post_tuning_results.json", 'w') as f:
        # Convert numpy arrays to lists for JSON serialization
        results_to_save = {k: v.tolist() if isinstance(v, np.ndarray) else v 
                          for k, v in post_tuning_results.items()}
        json.dump(results_to_save, f, indent=2)
    
    # Create evaluation visualizations
    evaluator = ModelEvaluator(output_dir="../outputs")
    evaluator.plot_confusion_matrix(
        post_tuning_results['labels'], 
        post_tuning_results['predictions'],
        "GPT Fine-tuned",
        "figures/gpt_finetuned_confusion_matrix.png"
    )
    
    evaluator.plot_roc_curve(
        np.array(post_tuning_results['labels']),
        np.array(post_tuning_results['probabilities']),
        "GPT Fine-tuned",
        "figures/gpt_finetuned_roc_curve.png"
    )
    
    logger.info("=" * 80)
    logger.info("FINE-TUNING PIPELINE COMPLETED SUCCESSFULLY!")
    logger.info("=" * 80)
    
    return post_tuning_results

def main():
    """Main execution function."""
    try:
        results = run_complete_pipeline()
        
        logger.info(f"\n🎉 FINAL RESULTS:")
        logger.info(f"Post-tuning Accuracy: {results['accuracy']:.4f}")
        logger.info(f"Post-tuning F1-score: {results['f1']:.4f}")
        logger.info(f"Post-tuning ROC AUC: {results['roc_auc']:.4f}")
        
    except Exception as e:
        logger.error(f"Pipeline failed with error: {str(e)}")
        raise

if __name__ == "__main__":
    main()


