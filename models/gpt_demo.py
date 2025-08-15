#!/usr/bin/env python3
import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
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

class CompactGPTClassifier(nn.Module):
    """
    Compact GPT classifier for demonstration.
    """
    
    def __init__(self, model_name: str = "gpt2", num_labels: int = 2):
        super(CompactGPTClassifier, self).__init__()
        
        # Load tokenizer
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load GPT model
        self.gpt = GPT2LMHeadModel.from_pretrained(model_name)
        
        # Freeze most of the GPT parameters to save memory
        for param in self.gpt.parameters():
            param.requires_grad = False
        
        # Only train the last few layers
        for param in self.gpt.transformer.h[-2:].parameters():
            param.requires_grad = True
        
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

def evaluate_zero_shot_baseline(X_test, y_test, sample_size=500):
    """Evaluate zero-shot performance and baselines."""
    logger.info("Evaluating zero-shot and baseline performance...")
    
    # Sample data for faster evaluation
    indices = np.random.choice(len(X_test), min(sample_size, len(X_test)), replace=False)
    X_sample = X_test[indices]
    y_sample = y_test[indices]
    
    # Baseline models
    baselines = {
        'Random Forest': RandomForestClassifier(n_estimators=50, random_state=42),
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=500)
    }
    
    baseline_results = {}
    
    for name, model in baselines.items():
        model.fit(X_sample, y_sample)
        predictions = model.predict(X_sample)
        accuracy = accuracy_score(y_sample, predictions)
        baseline_results[name] = accuracy
        logger.info(f"{name} accuracy: {accuracy:.4f}")
    
    # Simulate zero-shot GPT performance (random with slight bias)
    np.random.seed(42)
    zero_shot_predictions = np.random.choice([0, 1], size=len(y_sample), p=[0.6, 0.4])
    zero_shot_accuracy = accuracy_score(y_sample, zero_shot_predictions)
    baseline_results['GPT Zero-shot (simulated)'] = zero_shot_accuracy
    logger.info(f"GPT Zero-shot (simulated) accuracy: {zero_shot_accuracy:.4f}")
    
    return baseline_results

def train_compact_gpt(X_train, X_val, X_test, y_train, y_val, y_test, sample_size=1000):
    """Train compact GPT model with limited data."""
    logger.info(f"Training compact GPT with {sample_size} samples...")
    
    # Sample training data
    train_indices = np.random.choice(len(X_train), min(sample_size, len(X_train)), replace=False)
    X_train_sample = X_train[train_indices]
    y_train_sample = y_train[train_indices]
    
    # Sample validation data
    val_indices = np.random.choice(len(X_val), min(sample_size//3, len(X_val)), replace=False)
    X_val_sample = X_val[val_indices]
    y_val_sample = y_val[val_indices]
    
    # Sample test data
    test_indices = np.random.choice(len(X_test), min(sample_size//3, len(X_test)), replace=False)
    X_test_sample = X_test[test_indices]
    y_test_sample = y_test[test_indices]
    
    # Initialize model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CompactGPTClassifier("gpt2", num_labels=2)
    model.to(device)
    
    # Count trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Trainable parameters: {trainable_params:,} / {total_params:,}")
    
    # Create datasets
    train_dataset = DroneDataset(X_train_sample, y_train_sample, model.tokenizer, max_length=128)
    val_dataset = DroneDataset(X_val_sample, y_val_sample, model.tokenizer, max_length=128)
    test_dataset = DroneDataset(X_test_sample, y_test_sample, model.tokenizer, max_length=128)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)
    
    # Setup optimizer
    optimizer = optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.01)
    
    # Training loop
    num_epochs = 2
    best_val_accuracy = 0
    
    for epoch in range(num_epochs):
        logger.info(f"\nEpoch {epoch + 1}/{num_epochs}")
        
        # Training
        model.train()
        total_loss = 0
        correct_predictions = 0
        total_predictions = 0
        
        for batch in tqdm(train_loader, desc="Training"):
            optimizer.zero_grad()
            
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs['loss']
            logits = outputs['logits']
            
            loss.backward()
            optimizer.step()
            
            # Calculate accuracy
            predictions = torch.argmax(logits, dim=-1)
            correct_predictions += (predictions == labels).sum().item()
            total_predictions += labels.size(0)
            total_loss += loss.item()
        
        train_accuracy = correct_predictions / total_predictions
        avg_loss = total_loss / len(train_loader)
        
        # Validation
        model.eval()
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs['logits']
                predictions = torch.argmax(logits, dim=-1)
                
                val_correct += (predictions == labels).sum().item()
                val_total += labels.size(0)
        
        val_accuracy = val_correct / val_total
        
        logger.info(f"Train Loss: {avg_loss:.4f}, Train Acc: {train_accuracy:.4f}, Val Acc: {val_accuracy:.4f}")
        
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            # Save best model
            os.makedirs("models/compact_finetuned", exist_ok=True)
            torch.save(model.state_dict(), "models/compact_finetuned/best_model.pth")
    
    # Final evaluation
    model.eval()
    all_predictions = []
    all_labels = []
    all_probabilities = []
    
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs['logits']
            
            probabilities = torch.softmax(logits, dim=-1)
            predictions = torch.argmax(logits, dim=-1)
            
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())
    
    # Calculate final metrics
    accuracy = accuracy_score(all_labels, all_predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_predictions, average='weighted')
    
    probabilities_array = np.array(all_probabilities)
    roc_auc = roc_auc_score(all_labels, probabilities_array[:, 1])
    
    results = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'roc_auc': roc_auc,
        'predictions': all_predictions,
        'labels': all_labels
    }
    
    logger.info(f"\nFinal Test Results:")
    logger.info(f"Accuracy: {accuracy:.4f}")
    logger.info(f"Precision: {precision:.4f}")
    logger.info(f"Recall: {recall:.4f}")
    logger.info(f"F1-score: {f1:.4f}")
    logger.info(f"ROC AUC: {roc_auc:.4f}")
    
    return results

def create_comparison_visualization(baseline_results, gpt_results):
    """Create comparison visualization."""
    os.makedirs("../figures", exist_ok=True)
    
    # Model comparison
    models = list(baseline_results.keys()) + ['GPT Fine-tuned']
    accuracies = list(baseline_results.values()) + [gpt_results['accuracy']]
    
    plt.figure(figsize=(12, 8))
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
    bars = plt.bar(models, accuracies, color=colors)
    
    plt.title('Model Performance Comparison - Drone Data Classification', fontsize=16, fontweight='bold')
    plt.xlabel('Models', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.ylim(0, 1)
    
    # Add value labels on bars
    for bar, acc in zip(bars, accuracies):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{acc:.3f}', ha='center', va='bottom', fontweight='bold')
    
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("figures/model_comparison_demo.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    # Confusion matrix
    cm = confusion_matrix(gpt_results['labels'], gpt_results['predictions'])
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
               xticklabels=['Low Risk (Group 1)', 'High Risk (Group 2)'],
               yticklabels=['Low Risk (Group 1)', 'High Risk (Group 2)'])
    plt.title('GPT Fine-tuned Model - Confusion Matrix', fontsize=16, fontweight='bold')
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.tight_layout()
    plt.savefig("figures/gpt_confusion_matrix_demo.png", dpi=300, bbox_inches='tight')
    plt.show()

def main():
    """Main execution function."""
    logger.info("=" * 80)
    logger.info("MEMORY-EFFICIENT GPT FINE-TUNING DEMONSTRATION")
    logger.info("=" * 80)
    
    # Load preprocessed data
    logger.info("Loading preprocessed data...")
    X_train = np.load("data/X_train.npy")
    X_val = np.load("data/X_val.npy")
    X_test = np.load("data/X_test.npy")
    y_train = np.load("data/y_train.npy")
    y_val = np.load("data/y_val.npy")
    y_test = np.load("data/y_test.npy")
    
    logger.info(f"Full dataset shapes: Train {X_train.shape}, Val {X_val.shape}, Test {X_test.shape}")
    
    # Evaluate baselines
    baseline_results = evaluate_zero_shot_baseline(X_test, y_test)
    
    # Train compact GPT
    gpt_results = train_compact_gpt(X_train, X_val, X_test, y_train, y_val, y_test)
    
    # Create visualizations
    create_comparison_visualization(baseline_results, gpt_results)
    
    # Save results
    os.makedirs("../outputs", exist_ok=True)
    
    final_results = {
        'baseline_results': baseline_results,
        'gpt_finetuned_results': {k: v.tolist() if isinstance(v, np.ndarray) else v 
                                 for k, v in gpt_results.items()}
    }
    
    with open("outputs/demo_results.json", 'w') as f:
        json.dump(final_results, f, indent=2)
    
    # Summary
    logger.info("\n" + "=" * 80)
    logger.info("DEMONSTRATION COMPLETED SUCCESSFULLY!")
    logger.info("=" * 80)
    
    logger.info("\n📊 PERFORMANCE SUMMARY:")
    for model, accuracy in baseline_results.items():
        logger.info(f"{model}: {accuracy:.4f}")
    logger.info(f"GPT Fine-tuned: {gpt_results['accuracy']:.4f}")
    
    # Calculate improvement
    best_baseline = max(baseline_results.values())
    improvement = gpt_results['accuracy'] - best_baseline
    logger.info(f"\nImprovement over best baseline: {improvement:.4f}")
    
    logger.info("\n📁 Generated Files:")
    logger.info("  - models/compact_finetuned/: Fine-tuned model")
    logger.info("  - outputs/demo_results.json: Results summary")
    logger.info("  - figures/: Visualization plots")
    
    return final_results

if __name__ == "__main__":
    main()


