#!/usr/bin/env python3
import os
import sys
import torch
import numpy as np
import pandas as pd
from transformers import GPT2Tokenizer, GPT2LMHeadModel, GPT2Config
import logging
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Add utils to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from utils.data_utils import DroneDataset
from utils.model_utils import GPTClassifier, ModelTrainer
from utils.eval_utils import ModelEvaluator

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GPTPreTuningEvaluator:
    """
    Evaluator for GPT model pre-tuning (zero-shot) performance.
    """
    
    def __init__(self, model_name: str = "gpt2"):
        """
        Initialize the pre-tuning evaluator.
        
        Args:
            model_name (str): Pre-trained model name
        """
        self.model_name = model_name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = None
        self.model = None
        
        logger.info(f"Initializing GPT Pre-tuning Evaluator with {model_name}")
        logger.info(f"Using device: {self.device}")
    
    def load_model(self):
        """Load the pre-trained GPT model."""
        logger.info(f"Loading pre-trained model: {self.model_name}")
        
        # Load tokenizer
        self.tokenizer = GPT2Tokenizer.from_pretrained(self.model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load model configuration
        config = GPT2Config.from_pretrained(self.model_name)
        config.num_labels = 2
        config.pad_token_id = self.tokenizer.eos_token_id
        
        # Initialize our custom GPT classifier
        self.model = GPTClassifier(self.model_name, num_labels=2)
        self.model.to(self.device)
        self.model.eval()
        
        logger.info(f"Model loaded successfully")
        logger.info(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
    
    def prepare_data_for_gpt(self, X: np.ndarray, y: np.ndarray, max_samples: int = 1000):
        """
        Prepare data for GPT evaluation.
        
        Args:
            X (np.ndarray): Feature array
            y (np.ndarray): Label array
            max_samples (int): Maximum number of samples to evaluate
            
        Returns:
            torch.utils.data.DataLoader: Prepared data loader
        """
        logger.info(f"Preparing data for GPT evaluation...")
        
        # Limit samples for faster evaluation
        if len(X) > max_samples:
            indices = np.random.choice(len(X), max_samples, replace=False)
            X = X[indices]
            y = y[indices]
            logger.info(f"Sampled {max_samples} examples for evaluation")
        
        # Create dataset
        dataset = DroneDataset(X, y, self.tokenizer, max_length=512)
        
        # Create data loader
        data_loader = torch.utils.data.DataLoader(
            dataset, batch_size=8, shuffle=False, num_workers=2
        )
        
        logger.info(f"Data prepared: {len(dataset)} samples")
        return data_loader
    
    def evaluate_zero_shot(self, data_loader):
        """
        Evaluate the model in zero-shot setting.
        
        Args:
            data_loader: Data loader for evaluation
            
        Returns:
            Dict: Evaluation results
        """
        logger.info("Starting zero-shot evaluation...")
        
        if self.model is None:
            self.load_model()
        
        self.model.eval()
        all_predictions = []
        all_labels = []
        all_probabilities = []
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(data_loader):
                if batch_idx % 10 == 0:
                    logger.info(f"Processing batch {batch_idx + 1}/{len(data_loader)}")
                
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                # Forward pass
                try:
                    outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                    logits = outputs['logits']
                    
                    # Get predictions and probabilities
                    probabilities = torch.softmax(logits, dim=-1)
                    predictions = torch.argmax(logits, dim=-1)
                    
                    all_predictions.extend(predictions.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())
                    all_probabilities.extend(probabilities.cpu().numpy())
                    
                except Exception as e:
                    logger.warning(f"Error processing batch {batch_idx}: {e}")
                    continue
        
        # Calculate metrics
        accuracy = accuracy_score(all_labels, all_predictions)
        
        logger.info("Zero-shot evaluation completed!")
        logger.info(f"Accuracy: {accuracy:.4f}")
        
        return {
            'predictions': all_predictions,
            'labels': all_labels,
            'probabilities': all_probabilities,
            'accuracy': accuracy
        }
    
    def create_baseline_comparison(self, X_test: np.ndarray, y_test: np.ndarray):
        """
        Create baseline comparisons for the GPT model.
        
        Args:
            X_test (np.ndarray): Test features
            y_test (np.ndarray): Test labels
            
        Returns:
            Dict: Comparison results
        """
        logger.info("Creating baseline comparisons...")
        
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.linear_model import LogisticRegression
        from sklearn.svm import SVC
        
        # Traditional ML baselines
        baselines = {
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
            'SVM': SVC(random_state=42, probability=True)
        }
        
        baseline_results = {}
        
        for name, model in baselines.items():
            logger.info(f"Training {name}...")
            
            # Use a subset for faster training
            train_size = min(1000, len(X_test))
            indices = np.random.choice(len(X_test), train_size, replace=False)
            X_train_subset = X_test[indices]
            y_train_subset = y_test[indices]
            
            model.fit(X_train_subset, y_train_subset)
            predictions = model.predict(X_test)
            probabilities = model.predict_proba(X_test) if hasattr(model, 'predict_proba') else None
            
            accuracy = accuracy_score(y_test, predictions)
            
            baseline_results[name] = {
                'accuracy': accuracy,
                'predictions': predictions,
                'probabilities': probabilities
            }
            
            logger.info(f"{name} accuracy: {accuracy:.4f}")
        
        return baseline_results
    
    def visualize_results(self, gpt_results: dict, baseline_results: dict = None, 
                         save_dir: str = "figures"):
        """
        Visualize evaluation results.
        
        Args:
            gpt_results (dict): GPT evaluation results
            baseline_results (dict, optional): Baseline model results
            save_dir (str): Directory to save figures
        """
        os.makedirs(save_dir, exist_ok=True)
        
        # GPT Confusion Matrix
        plt.figure(figsize=(10, 8))
        cm = confusion_matrix(gpt_results['labels'], gpt_results['predictions'])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=['Low Risk (Group 1)', 'High Risk (Group 2)'],
                   yticklabels=['Low Risk (Group 1)', 'High Risk (Group 2)'])
        plt.title('GPT Zero-shot Performance - Confusion Matrix', fontsize=16, fontweight='bold')
        plt.xlabel('Predicted Label', fontsize=12)
        plt.ylabel('True Label', fontsize=12)
        plt.tight_layout()
        plt.savefig(f"{save_dir}/gpt_zero_shot_confusion_matrix.png", dpi=300, bbox_inches='tight')
        plt.show()
        
        # Model Comparison
        if baseline_results:
            models = ['GPT Zero-shot'] + list(baseline_results.keys())
            accuracies = [gpt_results['accuracy']] + [baseline_results[name]['accuracy'] for name in baseline_results.keys()]
            
            plt.figure(figsize=(12, 8))
            bars = plt.bar(models, accuracies, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'])
            plt.title('Model Performance Comparison - Pre-tuning', fontsize=16, fontweight='bold')
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
            plt.savefig(f"{save_dir}/pre_tuning_model_comparison.png", dpi=300, bbox_inches='tight')
            plt.show()
    
    def generate_report(self, gpt_results: dict, baseline_results: dict = None,
                       save_path: str = "outputs/pre_tuning_report.md"):
        """
        Generate a comprehensive pre-tuning evaluation report.
        
        Args:
            gpt_results (dict): GPT evaluation results
            baseline_results (dict, optional): Baseline model results
            save_path (str): Path to save the report
        """
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        report_lines = []
        report_lines.append("# GPT Pre-tuning (Zero-shot) Evaluation Report")
        report_lines.append("=" * 60)
        report_lines.append("")
        
        # GPT Results
        report_lines.append("## GPT Zero-shot Performance")
        report_lines.append(f"- **Model**: {self.model_name}")
        report_lines.append(f"- **Accuracy**: {gpt_results['accuracy']:.4f}")
        report_lines.append(f"- **Samples Evaluated**: {len(gpt_results['labels'])}")
        report_lines.append("")
        
        # Classification Report
        report_lines.append("### Detailed Classification Report")
        class_names = ['Low Risk (Group 1)', 'High Risk (Group 2)']
        report = classification_report(gpt_results['labels'], gpt_results['predictions'], 
                                     target_names=class_names)
        report_lines.append("```")
        report_lines.append(report)
        report_lines.append("```")
        report_lines.append("")
        
        # Baseline Comparison
        if baseline_results:
            report_lines.append("## Baseline Model Comparison")
            report_lines.append("")
            report_lines.append("| Model | Accuracy |")
            report_lines.append("|-------|----------|")
            report_lines.append(f"| GPT Zero-shot | {gpt_results['accuracy']:.4f} |")
            
            for name, results in baseline_results.items():
                report_lines.append(f"| {name} | {results['accuracy']:.4f} |")
            
            report_lines.append("")
        
        # Analysis
        report_lines.append("## Analysis")
        report_lines.append("")
        if gpt_results['accuracy'] > 0.5:
            report_lines.append("✅ The GPT model shows better than random performance in zero-shot setting.")
        else:
            report_lines.append("⚠️ The GPT model performance is at or below random chance in zero-shot setting.")
        
        if baseline_results:
            best_baseline = max(baseline_results.items(), key=lambda x: x[1]['accuracy'])
            if gpt_results['accuracy'] > best_baseline[1]['accuracy']:
                report_lines.append(f"✅ GPT zero-shot outperforms the best baseline ({best_baseline[0]}).")
            else:
                report_lines.append(f"⚠️ GPT zero-shot underperforms compared to {best_baseline[0]}.")
        
        report_lines.append("")
        report_lines.append("## Recommendations")
        report_lines.append("- Fine-tuning is recommended to improve performance")
        report_lines.append("- Consider prompt engineering for better zero-shot results")
        report_lines.append("- Evaluate with larger sample sizes for more robust results")
        
        # Save report
        with open(save_path, 'w') as f:
            f.write('\n'.join(report_lines))
        
        logger.info(f"Pre-tuning evaluation report saved to {save_path}")

def main():
    """Main execution function."""
    logger.info("=" * 80)
    logger.info("GPT MODEL IMPLEMENTATION AND PRE-TUNING EVALUATION")
    logger.info("=" * 80)
    
    # Load preprocessed data
    logger.info("Loading preprocessed data...")
    X_test = np.load("../data/X_test.npy")
    y_test = np.load("../data/y_test.npy")
    
    logger.info(f"Test data loaded: {X_test.shape[0]} samples, {X_test.shape[1]} features")
    
    # Initialize evaluator
    evaluator = GPTPreTuningEvaluator("gpt2")
    
    # Prepare data
    data_loader = evaluator.prepare_data_for_gpt(X_test, y_test, max_samples=500)
    
    # Evaluate zero-shot performance
    gpt_results = evaluator.evaluate_zero_shot(data_loader)
    
    # Create baseline comparisons
    baseline_results = evaluator.create_baseline_comparison(X_test, y_test)
    
    # Visualize results
    evaluator.visualize_results(gpt_results, baseline_results)
    
    # Generate report
    evaluator.generate_report(gpt_results, baseline_results)
    
    logger.info("=" * 80)
    logger.info("PRE-TUNING EVALUATION COMPLETED SUCCESSFULLY!")
    logger.info("=" * 80)
    
    logger.info(f"\n📊 RESULTS SUMMARY:")
    logger.info(f"GPT Zero-shot Accuracy: {gpt_results['accuracy']:.4f}")
    
    if baseline_results:
        logger.info(f"\n📊 BASELINE COMPARISON:")
        for name, results in baseline_results.items():
            logger.info(f"{name}: {results['accuracy']:.4f}")

if __name__ == "__main__":
    main()


