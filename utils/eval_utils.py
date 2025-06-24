#!/usr/bin/env python3
"""
Evaluation Utilities for Drone Data Leakage Classification
Authors: Anas AlSobeh (Southern Illinois University Carbondale)
         Omar Darwish (Eastern Michigan University)

This module provides utilities for model evaluation and performance analysis.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, roc_auc_score,
    confusion_matrix, classification_report, roc_curve, auc
)
from sklearn.preprocessing import label_binarize
import pandas as pd
import json
import logging
from typing import Dict, List, Tuple, Any, Optional
import os

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelEvaluator:
    """
    Comprehensive model evaluation utilities.
    """
    
    def __init__(self, class_names: List[str] = None, output_dir: str = "outputs"):
        """
        Initialize the evaluator.
        
        Args:
            class_names (List[str], optional): Names of the classes
            output_dir (str): Output directory for saving results
        """
        self.class_names = class_names or ["Low Risk (Group 1)", "High Risk (Group 2)"]
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Set plotting style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
    
    def evaluate_model(self, y_true: np.ndarray, y_pred: np.ndarray, 
                      y_prob: np.ndarray = None, model_name: str = "Model") -> Dict[str, Any]:
        """
        Comprehensive model evaluation.
        
        Args:
            y_true (np.ndarray): True labels
            y_pred (np.ndarray): Predicted labels
            y_prob (np.ndarray, optional): Prediction probabilities
            model_name (str): Name of the model being evaluated
            
        Returns:
            Dict[str, Any]: Evaluation results
        """
        logger.info(f"Evaluating {model_name}...")
        
        # Basic metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision, recall, f1, support = precision_recall_fscore_support(y_true, y_pred, average=None)
        precision_avg, recall_avg, f1_avg, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted')
        
        # ROC AUC
        roc_auc = None
        if y_prob is not None:
            if len(self.class_names) == 2:  # Binary classification
                roc_auc = roc_auc_score(y_true, y_prob[:, 1] if y_prob.ndim > 1 else y_prob)
            else:  # Multi-class
                roc_auc = roc_auc_score(y_true, y_prob, multi_class='ovr')
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        # Classification report
        report = classification_report(y_true, y_pred, target_names=self.class_names, output_dict=True)
        
        results = {
            'model_name': model_name,
            'accuracy': accuracy,
            'precision_per_class': precision.tolist(),
            'recall_per_class': recall.tolist(),
            'f1_per_class': f1.tolist(),
            'support_per_class': support.tolist(),
            'precision_avg': precision_avg,
            'recall_avg': recall_avg,
            'f1_avg': f1_avg,
            'roc_auc': roc_auc,
            'confusion_matrix': cm.tolist(),
            'classification_report': report
        }
        
        logger.info(f"{model_name} evaluation completed:")
        logger.info(f"  Accuracy: {accuracy:.4f}")
        logger.info(f"  Precision (avg): {precision_avg:.4f}")
        logger.info(f"  Recall (avg): {recall_avg:.4f}")
        logger.info(f"  F1-score (avg): {f1_avg:.4f}")
        if roc_auc:
            logger.info(f"  ROC AUC: {roc_auc:.4f}")
        
        return results
    
    def plot_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray, 
                             model_name: str = "Model", save_path: str = None) -> None:
        """
        Plot confusion matrix.
        
        Args:
            y_true (np.ndarray): True labels
            y_pred (np.ndarray): Predicted labels
            model_name (str): Name of the model
            save_path (str, optional): Path to save the plot
        """
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=self.class_names, yticklabels=self.class_names)
        plt.title(f'Confusion Matrix - {model_name}', fontsize=16, fontweight='bold')
        plt.xlabel('Predicted Label', fontsize=12)
        plt.ylabel('True Label', fontsize=12)
        
        if save_path is None:
            save_path = f"{self.output_dir}/confusion_matrix_{model_name.lower().replace(' ', '_')}.png"
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        logger.info(f"Confusion matrix saved to {save_path}")
    
    def plot_roc_curve(self, y_true: np.ndarray, y_prob: np.ndarray, 
                      model_name: str = "Model", save_path: str = None) -> None:
        """
        Plot ROC curve.
        
        Args:
            y_true (np.ndarray): True labels
            y_prob (np.ndarray): Prediction probabilities
            model_name (str): Name of the model
            save_path (str, optional): Path to save the plot
        """
        plt.figure(figsize=(10, 8))
        
        if len(self.class_names) == 2:  # Binary classification
            fpr, tpr, _ = roc_curve(y_true, y_prob[:, 1] if y_prob.ndim > 1 else y_prob)
            roc_auc = auc(fpr, tpr)
            
            plt.plot(fpr, tpr, color='darkorange', lw=2, 
                    label=f'ROC curve (AUC = {roc_auc:.4f})')
            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
            
        else:  # Multi-class
            # Binarize the output
            y_true_bin = label_binarize(y_true, classes=range(len(self.class_names)))
            
            for i in range(len(self.class_names)):
                fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_prob[:, i])
                roc_auc = auc(fpr, tpr)
                plt.plot(fpr, tpr, lw=2, 
                        label=f'{self.class_names[i]} (AUC = {roc_auc:.4f})')
            
            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title(f'ROC Curve - {model_name}', fontsize=16, fontweight='bold')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        
        if save_path is None:
            save_path = f"{self.output_dir}/roc_curve_{model_name.lower().replace(' ', '_')}.png"
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        logger.info(f"ROC curve saved to {save_path}")
    
    def plot_precision_recall_curve(self, y_true: np.ndarray, y_prob: np.ndarray,
                                   model_name: str = "Model", save_path: str = None) -> None:
        """
        Plot precision-recall curve.
        
        Args:
            y_true (np.ndarray): True labels
            y_prob (np.ndarray): Prediction probabilities
            model_name (str): Name of the model
            save_path (str, optional): Path to save the plot
        """
        from sklearn.metrics import precision_recall_curve, average_precision_score
        
        plt.figure(figsize=(10, 8))
        
        if len(self.class_names) == 2:  # Binary classification
            precision, recall, _ = precision_recall_curve(y_true, y_prob[:, 1] if y_prob.ndim > 1 else y_prob)
            avg_precision = average_precision_score(y_true, y_prob[:, 1] if y_prob.ndim > 1 else y_prob)
            
            plt.plot(recall, precision, color='darkorange', lw=2,
                    label=f'PR curve (AP = {avg_precision:.4f})')
            
        else:  # Multi-class
            # Binarize the output
            y_true_bin = label_binarize(y_true, classes=range(len(self.class_names)))
            
            for i in range(len(self.class_names)):
                precision, recall, _ = precision_recall_curve(y_true_bin[:, i], y_prob[:, i])
                avg_precision = average_precision_score(y_true_bin[:, i], y_prob[:, i])
                plt.plot(recall, precision, lw=2,
                        label=f'{self.class_names[i]} (AP = {avg_precision:.4f})')
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall', fontsize=12)
        plt.ylabel('Precision', fontsize=12)
        plt.title(f'Precision-Recall Curve - {model_name}', fontsize=16, fontweight='bold')
        plt.legend(loc="lower left")
        plt.grid(True, alpha=0.3)
        
        if save_path is None:
            save_path = f"{self.output_dir}/pr_curve_{model_name.lower().replace(' ', '_')}.png"
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        logger.info(f"Precision-Recall curve saved to {save_path}")
    
    def compare_models(self, results_list: List[Dict[str, Any]], 
                      save_path: str = None) -> None:
        """
        Compare multiple models' performance.
        
        Args:
            results_list (List[Dict[str, Any]]): List of evaluation results
            save_path (str, optional): Path to save the comparison plot
        """
        metrics = ['accuracy', 'precision_avg', 'recall_avg', 'f1_avg']
        if any(r.get('roc_auc') is not None for r in results_list):
            metrics.append('roc_auc')
        
        model_names = [r['model_name'] for r in results_list]
        
        # Create comparison DataFrame
        comparison_data = {}
        for metric in metrics:
            comparison_data[metric] = [r.get(metric, 0) for r in results_list]
        
        df = pd.DataFrame(comparison_data, index=model_names)
        
        # Plot comparison
        fig, ax = plt.subplots(figsize=(12, 8))
        df.plot(kind='bar', ax=ax, width=0.8)
        
        plt.title('Model Performance Comparison', fontsize=16, fontweight='bold')
        plt.xlabel('Models', fontsize=12)
        plt.ylabel('Score', fontsize=12)
        plt.legend(title='Metrics', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for container in ax.containers:
            ax.bar_label(container, fmt='%.3f', rotation=90, fontsize=8)
        
        if save_path is None:
            save_path = f"{self.output_dir}/model_comparison.png"
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        logger.info(f"Model comparison saved to {save_path}")
        
        # Print comparison table
        print("\nModel Performance Comparison:")
        print("=" * 80)
        print(df.round(4).to_string())
    
    def save_results(self, results: Dict[str, Any], filename: str = None) -> None:
        """
        Save evaluation results to JSON file.
        
        Args:
            results (Dict[str, Any]): Evaluation results
            filename (str, optional): Output filename
        """
        if filename is None:
            model_name = results.get('model_name', 'model').lower().replace(' ', '_')
            filename = f"{self.output_dir}/evaluation_results_{model_name}.json"
        
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Evaluation results saved to {filename}")
    
    def generate_evaluation_report(self, results_list: List[Dict[str, Any]], 
                                 report_path: str = None) -> None:
        """
        Generate a comprehensive evaluation report.
        
        Args:
            results_list (List[Dict[str, Any]]): List of evaluation results
            report_path (str, optional): Path to save the report
        """
        if report_path is None:
            report_path = f"{self.output_dir}/evaluation_report.md"
        
        report_lines = []
        report_lines.append("# Drone Data Leakage Classification - Evaluation Report")
        report_lines.append("=" * 70)
        report_lines.append("")
        
        # Summary table
        report_lines.append("## Performance Summary")
        report_lines.append("")
        report_lines.append("| Model | Accuracy | Precision | Recall | F1-Score | ROC AUC |")
        report_lines.append("|-------|----------|-----------|--------|----------|---------|")
        
        for results in results_list:
            model_name = results['model_name']
            accuracy = results['accuracy']
            precision = results['precision_avg']
            recall = results['recall_avg']
            f1 = results['f1_avg']
            roc_auc = results.get('roc_auc', 'N/A')
            
            if roc_auc != 'N/A':
                roc_auc = f"{roc_auc:.4f}"
            
            report_lines.append(f"| {model_name} | {accuracy:.4f} | {precision:.4f} | {recall:.4f} | {f1:.4f} | {roc_auc} |")
        
        report_lines.append("")
        
        # Detailed results for each model
        for results in results_list:
            model_name = results['model_name']
            report_lines.append(f"## {model_name} - Detailed Results")
            report_lines.append("")
            
            # Overall metrics
            report_lines.append("### Overall Performance")
            report_lines.append(f"- **Accuracy**: {results['accuracy']:.4f}")
            report_lines.append(f"- **Precision (weighted avg)**: {results['precision_avg']:.4f}")
            report_lines.append(f"- **Recall (weighted avg)**: {results['recall_avg']:.4f}")
            report_lines.append(f"- **F1-score (weighted avg)**: {results['f1_avg']:.4f}")
            if results.get('roc_auc'):
                report_lines.append(f"- **ROC AUC**: {results['roc_auc']:.4f}")
            report_lines.append("")
            
            # Per-class metrics
            report_lines.append("### Per-Class Performance")
            for i, class_name in enumerate(self.class_names):
                if i < len(results['precision_per_class']):
                    precision = results['precision_per_class'][i]
                    recall = results['recall_per_class'][i]
                    f1 = results['f1_per_class'][i]
                    support = results['support_per_class'][i]
                    
                    report_lines.append(f"#### {class_name}")
                    report_lines.append(f"- **Precision**: {precision:.4f}")
                    report_lines.append(f"- **Recall**: {recall:.4f}")
                    report_lines.append(f"- **F1-score**: {f1:.4f}")
                    report_lines.append(f"- **Support**: {support}")
                    report_lines.append("")
            
            # Confusion matrix
            report_lines.append("### Confusion Matrix")
            cm = np.array(results['confusion_matrix'])
            report_lines.append("```")
            report_lines.append("Predicted →")
            header = "True ↓    " + "  ".join([f"{name:>10}" for name in self.class_names])
            report_lines.append(header)
            for i, class_name in enumerate(self.class_names):
                row = f"{class_name:>10}" + "  ".join([f"{cm[i][j]:>10}" for j in range(len(self.class_names))])
                report_lines.append(row)
            report_lines.append("```")
            report_lines.append("")
        
        # Save report
        with open(report_path, 'w') as f:
            f.write('\n'.join(report_lines))
        
        logger.info(f"Evaluation report saved to {report_path}")
        
        return report_lines

