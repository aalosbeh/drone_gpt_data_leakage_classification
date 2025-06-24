#!/usr/bin/env python3
"""
Simplified GPT Fine-tuning for Drone Data Classification
Authors: Anas AlSobeh (Southern Illinois University Carbondale)
         Omar Darwish (Eastern Michigan University)

This script implements a simplified but effective GPT fine-tuning approach.
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score, confusion_matrix
import logging
import json
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# Add utils to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from utils.data_utils import DroneDataset

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SimpleGPTClassifier(nn.Module):
    """
    Simplified GPT classifier for drone data classification.
    """
    
    def __init__(self, model_name: str = "gpt2", num_labels: int = 2):
        super(SimpleGPTClassifier, self).__init__()
        
        self.model_name = model_name
        self.num_labels = num_labels
        
        # Load tokenizer
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load GPT model
        self.gpt = GPT2LMHeadModel.from_pretrained(model_name)
        
        # Classification head
        self.classifier = nn.Linear(self.gpt.config.hidden_size, num_labels)
        self.dropout = nn.Dropout(0.1)
        
        # Initialize classifier weights
        nn.init.normal_(self.classifier.weight, std=0.02)
        nn.init.zeros_(self.classifier.bias)
    
    def forward(self, input_ids, attention_mask, labels=None):
        # Get GPT outputs
        outputs = self.gpt.transformer(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = outputs.last_hidden_state
        
        # Pool the sequence (use the last non-padding token)
        batch_size = last_hidden_state.size(0)
        sequence_lengths = attention_mask.sum(dim=1) - 1
        pooled_output = last_hidden_state[range(batch_size), sequence_lengths]
        
        # Apply dropout and classification
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        
        outputs = {"logits": logits}
        
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits, labels)
            outputs["loss"] = loss
        
        return outputs

class SimpleGPTTrainer:
    """
    Simple trainer for GPT classification.
    """
    
    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.model.to(device)
    
    def train_epoch(self, train_loader, optimizer, scheduler=None):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        correct_predictions = 0
        total_predictions = 0
        
        progress_bar = tqdm(train_loader, desc="Training")
        
        for batch in progress_bar:
            optimizer.zero_grad()
            
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs['loss']
            logits = outputs['logits']
            
            loss.backward()
            optimizer.step()
            
            if scheduler:
                scheduler.step()
            
            # Calculate accuracy
            predictions = torch.argmax(logits, dim=-1)
            correct_predictions += (predictions == labels).sum().item()
            total_predictions += labels.size(0)
            
            total_loss += loss.item()
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{correct_predictions/total_predictions:.4f}'
            })
        
        avg_loss = total_loss / len(train_loader)
        accuracy = correct_predictions / total_predictions
        
        return avg_loss, accuracy
    
    def evaluate(self, eval_loader):
        """Evaluate the model."""
        self.model.eval()
        total_loss = 0
        all_predictions = []
        all_labels = []
        all_probabilities = []
        
        with torch.no_grad():
            for batch in tqdm(eval_loader, desc="Evaluating"):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs['loss']
                logits = outputs['logits']
                
                total_loss += loss.item()
                
                # Get predictions and probabilities
                probabilities = torch.softmax(logits, dim=-1)
                predictions = torch.argmax(logits, dim=-1)
                
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())
        
        avg_loss = total_loss / len(eval_loader)
        
        # Calculate metrics
        accuracy = accuracy_score(all_labels, all_predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_predictions, average='weighted')
        
        # ROC AUC
        probabilities_array = np.array(all_probabilities)
        roc_auc = roc_auc_score(all_labels, probabilities_array[:, 1])
        
        return {
            'loss': avg_loss,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'roc_auc': roc_auc,
            'predictions': all_predictions,
            'labels': all_labels,
            'probabilities': all_probabilities
        }

def create_data_loaders(X_train, X_val, X_test, y_train, y_val, y_test, tokenizer, batch_size=8):
    """Create data loaders for training."""
    train_dataset = DroneDataset(X_train, y_train, tokenizer, max_length=256)
    val_dataset = DroneDataset(X_val, y_val, tokenizer, max_length=256)
    test_dataset = DroneDataset(X_test, y_test, tokenizer, max_length=256)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    return train_loader, val_loader, test_loader

def plot_training_history(train_losses, train_accuracies, val_accuracies, save_path="figures/training_history.png"):
    """Plot training history."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Loss plot
    ax1.plot(train_losses, label='Training Loss', color='blue')
    ax1.set_title('Training Loss', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Accuracy plot
    ax2.plot(train_accuracies, label='Training Accuracy', color='blue')
    ax2.plot(val_accuracies, label='Validation Accuracy', color='red')
    ax2.set_title('Model Accuracy', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def plot_confusion_matrix(labels, predictions, save_path="figures/confusion_matrix_finetuned.png"):
    """Plot confusion matrix."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    cm = confusion_matrix(labels, predictions)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
               xticklabels=['Low Risk (Group 1)', 'High Risk (Group 2)'],
               yticklabels=['Low Risk (Group 1)', 'High Risk (Group 2)'])
    plt.title('GPT Fine-tuned Model - Confusion Matrix', fontsize=16, fontweight='bold')
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def main():
    """Main execution function."""
    logger.info("=" * 80)
    logger.info("SIMPLIFIED GPT FINE-TUNING FOR DRONE CLASSIFICATION")
    logger.info("=" * 80)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Load preprocessed data
    logger.info("Loading preprocessed data...")
    X_train = np.load("data/X_train.npy")
    X_val = np.load("data/X_val.npy")
    X_test = np.load("data/X_test.npy")
    y_train = np.load("data/y_train.npy")
    y_val = np.load("data/y_val.npy")
    y_test = np.load("data/y_test.npy")
    
    logger.info(f"Data shapes: Train {X_train.shape}, Val {X_val.shape}, Test {X_test.shape}")
    
    # Initialize model
    logger.info("Initializing GPT classifier...")
    model = SimpleGPTClassifier("gpt2", num_labels=2)
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Create data loaders
    logger.info("Creating data loaders...")
    train_loader, val_loader, test_loader = create_data_loaders(
        X_train, X_val, X_test, y_train, y_val, y_test, 
        model.tokenizer, batch_size=8
    )
    
    # Initialize trainer
    trainer = SimpleGPTTrainer(model, device)
    
    # Setup optimizer
    optimizer = optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
    
    # Training loop
    logger.info("Starting training...")
    num_epochs = 3
    train_losses = []
    train_accuracies = []
    val_accuracies = []
    
    best_val_accuracy = 0
    
    for epoch in range(num_epochs):
        logger.info(f"\nEpoch {epoch + 1}/{num_epochs}")
        
        # Train
        train_loss, train_acc = trainer.train_epoch(train_loader, optimizer)
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)
        
        # Validate
        val_results = trainer.evaluate(val_loader)
        val_accuracies.append(val_results['accuracy'])
        
        logger.info(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        logger.info(f"Val Acc: {val_results['accuracy']:.4f}, Val F1: {val_results['f1']:.4f}")
        
        # Save best model
        if val_results['accuracy'] > best_val_accuracy:
            best_val_accuracy = val_results['accuracy']
            os.makedirs("models/simple_finetuned", exist_ok=True)
            torch.save(model.state_dict(), "models/simple_finetuned/best_model.pth")
            logger.info("New best model saved!")
    
    # Final evaluation on test set
    logger.info("\nEvaluating on test set...")
    test_results = trainer.evaluate(test_loader)
    
    logger.info("=" * 80)
    logger.info("FINAL RESULTS")
    logger.info("=" * 80)
    logger.info(f"Test Accuracy: {test_results['accuracy']:.4f}")
    logger.info(f"Test Precision: {test_results['precision']:.4f}")
    logger.info(f"Test Recall: {test_results['recall']:.4f}")
    logger.info(f"Test F1-score: {test_results['f1']:.4f}")
    logger.info(f"Test ROC AUC: {test_results['roc_auc']:.4f}")
    
    # Save results
    os.makedirs("outputs", exist_ok=True)
    results_to_save = {k: v.tolist() if isinstance(v, np.ndarray) else v 
                      for k, v in test_results.items()}
    
    with open("outputs/simple_finetuning_results.json", 'w') as f:
        json.dump(results_to_save, f, indent=2)
    
    # Create visualizations
    plot_training_history(train_losses, train_accuracies, val_accuracies)
    plot_confusion_matrix(test_results['labels'], test_results['predictions'])
    
    logger.info("\n🎉 Fine-tuning completed successfully!")
    
    return test_results

if __name__ == "__main__":
    main()

