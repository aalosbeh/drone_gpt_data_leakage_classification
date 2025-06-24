#!/usr/bin/env python3
"""
Main Entry Point for Drone Data Leakage Classification
Authors: Anas AlSobeh (Southern Illinois University Carbondale)
         Omar Darwish (Eastern Michigan University)

This script provides the main entry point for the complete research pipeline.
"""

import os
import sys
import argparse
import logging
import yaml
import torch
import numpy as np
from datetime import datetime

# Add utils to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'utils'))

from utils.data_utils import DataLoader as DroneDataLoader
from utils.model_utils import ModelTrainer
from utils.eval_utils import ModelEvaluator

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/experiment.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def setup_directories():
    """Create necessary directories for the project."""
    directories = [
        'data', 'models', 'outputs', 'figures', 'logs', 
        'models/checkpoints', 'models/fine_tuned'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
    
    logger.info("Project directories created successfully")

def load_config(config_path: str = "config.yaml"):
    """Load configuration from YAML file."""
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        logger.info(f"Configuration loaded from {config_path}")
        return config
    except FileNotFoundError:
        logger.error(f"Configuration file {config_path} not found")
        sys.exit(1)

def set_random_seeds(seed: int = 42):
    """Set random seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
    # Make CUDA operations deterministic
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    logger.info(f"Random seeds set to {seed} for reproducibility")

def run_complete_pipeline(config_path: str = "config.yaml"):
    """
    Run the complete research pipeline.
    
    Args:
        config_path (str): Path to configuration file
    """
    logger.info("=" * 80)
    logger.info("DRONE DATA LEAKAGE CLASSIFICATION - COMPLETE PIPELINE")
    logger.info("=" * 80)
    
    # Setup
    setup_directories()
    config = load_config(config_path)
    set_random_seeds(config.get('experiment', {}).get('seed', 42))
    
    # Initialize components
    data_loader = DroneDataLoader(config_path)
    model_trainer = ModelTrainer(config_path)
    evaluator = ModelEvaluator(output_dir="outputs")
    
    # Step 1: Load and preprocess data
    logger.info("\n" + "=" * 50)
    logger.info("STEP 1: DATA LOADING AND PREPROCESSING")
    logger.info("=" * 50)
    
    # Check if preprocessed data exists
    if os.path.exists("data/X_train.npy"):
        logger.info("Loading existing preprocessed data...")
        X_train, X_val, X_test, y_train, y_val, y_test = data_loader.load_preprocessed_data()
    else:
        logger.info("Loading and preprocessing raw data...")
        df = data_loader.load_raw_data()
        X_train, X_val, X_test, y_train, y_val, y_test = data_loader.preprocess_data(df)
        data_loader.save_preprocessed_data(X_train, X_val, X_test, y_train, y_val, y_test)
    
    # Step 2: Initialize model and create data loaders
    logger.info("\n" + "=" * 50)
    logger.info("STEP 2: MODEL INITIALIZATION")
    logger.info("=" * 50)
    
    model = model_trainer.initialize_model()
    train_loader, val_loader, test_loader = data_loader.create_dataloaders(
        X_train, X_val, X_test, y_train, y_val, y_test,
        tokenizer=model_trainer.tokenizer,
        batch_size=config['model']['batch_size']
    )
    
    # Step 3: Zero-shot evaluation (pre-tuning)
    logger.info("\n" + "=" * 50)
    logger.info("STEP 3: ZERO-SHOT EVALUATION (PRE-TUNING)")
    logger.info("=" * 50)
    
    pre_tuning_results = model_trainer.evaluate_zero_shot(test_loader)
    evaluator.save_results(pre_tuning_results, "outputs/pre_tuning_results.json")
    
    # Step 4: Fine-tuning
    logger.info("\n" + "=" * 50)
    logger.info("STEP 4: MODEL FINE-TUNING")
    logger.info("=" * 50)
    
    model_trainer.fine_tune_model(train_loader, val_loader, "models/fine_tuned")
    
    # Step 5: Post-tuning evaluation
    logger.info("\n" + "=" * 50)
    logger.info("STEP 5: POST-TUNING EVALUATION")
    logger.info("=" * 50)
    
    post_tuning_results = model_trainer.evaluate_fine_tuned(test_loader, "models/fine_tuned")
    evaluator.save_results(post_tuning_results, "outputs/post_tuning_results.json")
    
    # Step 6: Comprehensive evaluation and visualization
    logger.info("\n" + "=" * 50)
    logger.info("STEP 6: COMPREHENSIVE EVALUATION")
    logger.info("=" * 50)
    
    # Compare pre and post-tuning results
    results_list = [
        {**pre_tuning_results, 'model_name': 'GPT Pre-tuning (Zero-shot)'},
        {**post_tuning_results, 'model_name': 'GPT Post-tuning (Fine-tuned)'}
    ]
    
    # Generate comparison plots
    evaluator.compare_models(results_list)
    
    # Generate comprehensive report
    evaluator.generate_evaluation_report(results_list)
    
    # Step 7: Final summary
    logger.info("\n" + "=" * 80)
    logger.info("PIPELINE COMPLETED SUCCESSFULLY!")
    logger.info("=" * 80)
    
    logger.info("\n RESULTS SUMMARY:")
    logger.info(f"Pre-tuning Accuracy:  {pre_tuning_results['accuracy']:.4f}")
    logger.info(f"Post-tuning Accuracy: {post_tuning_results['accuracy']:.4f}")
    logger.info(f"Improvement:          {post_tuning_results['accuracy'] - pre_tuning_results['accuracy']:.4f}")
    
    logger.info("\n Generated Files:")
    logger.info("  - models/fine_tuned/: Fine-tuned model")
    logger.info("  - outputs/: Evaluation results and reports")
    logger.info("  - figures/: Visualization plots")
    logger.info("  - logs/: Experiment logs")
    
    return results_list

def run_data_preprocessing_only(config_path: str = "config.yaml"):
    """Run only data preprocessing."""
    logger.info("Running data preprocessing only...")
    
    setup_directories()
    config = load_config(config_path)
    
    data_loader = DroneDataLoader(config_path)
    df = data_loader.load_raw_data()
    X_train, X_val, X_test, y_train, y_val, y_test = data_loader.preprocess_data(df)
    data_loader.save_preprocessed_data(X_train, X_val, X_test, y_train, y_val, y_test)
    
    logger.info("Data preprocessing completed successfully!")

def run_training_only(config_path: str = "config.yaml"):
    """Run only model training."""
    logger.info("Running model training only...")
    
    setup_directories()
    config = load_config(config_path)
    set_random_seeds(config.get('experiment', {}).get('seed', 42))
    
    # Load preprocessed data
    data_loader = DroneDataLoader(config_path)
    X_train, X_val, X_test, y_train, y_val, y_test = data_loader.load_preprocessed_data()
    
    # Initialize model and train
    model_trainer = ModelTrainer(config_path)
    model = model_trainer.initialize_model()
    
    train_loader, val_loader, test_loader = data_loader.create_dataloaders(
        X_train, X_val, X_test, y_train, y_val, y_test,
        tokenizer=model_trainer.tokenizer,
        batch_size=config['model']['batch_size']
    )
    
    model_trainer.fine_tune_model(train_loader, val_loader, "models/fine_tuned")
    
    logger.info("Model training completed successfully!")

def run_evaluation_only(config_path: str = "config.yaml"):
    """Run only model evaluation."""
    logger.info("Running model evaluation only...")
    
    setup_directories()
    config = load_config(config_path)
    
    # Load preprocessed data
    data_loader = DroneDataLoader(config_path)
    X_train, X_val, X_test, y_train, y_val, y_test = data_loader.load_preprocessed_data()
    
    # Initialize components
    model_trainer = ModelTrainer(config_path)
    evaluator = ModelEvaluator(output_dir="outputs")
    
    # Create test loader
    model = model_trainer.initialize_model()
    _, _, test_loader = data_loader.create_dataloaders(
        X_train, X_val, X_test, y_train, y_val, y_test,
        tokenizer=model_trainer.tokenizer,
        batch_size=config['model']['batch_size']
    )
    
    # Evaluate models
    pre_tuning_results = model_trainer.evaluate_zero_shot(test_loader)
    post_tuning_results = model_trainer.evaluate_fine_tuned(test_loader, "models/fine_tuned")
    
    # Generate reports
    results_list = [
        {**pre_tuning_results, 'model_name': 'GPT Pre-tuning (Zero-shot)'},
        {**post_tuning_results, 'model_name': 'GPT Post-tuning (Fine-tuned)'}
    ]
    
    evaluator.compare_models(results_list)
    evaluator.generate_evaluation_report(results_list)
    
    logger.info("Model evaluation completed successfully!")

def main():
    """Main function with command line interface."""
    parser = argparse.ArgumentParser(description="Drone Data Leakage Classification Research Pipeline")
    parser.add_argument("--mode", choices=["complete", "preprocess", "train", "evaluate"], 
                       default="complete", help="Pipeline mode to run")
    parser.add_argument("--config", default="config.yaml", help="Path to configuration file")
    
    args = parser.parse_args()
    
    # Create logs directory
    os.makedirs("logs", exist_ok=True)
    
    try:
        if args.mode == "complete":
            run_complete_pipeline(args.config)
        elif args.mode == "preprocess":
            run_data_preprocessing_only(args.config)
        elif args.mode == "train":
            run_training_only(args.config)
        elif args.mode == "evaluate":
            run_evaluation_only(args.config)
        
        logger.info(f"\n🎉 {args.mode.upper()} mode completed successfully!")
        
    except Exception as e:
        logger.error(f"Pipeline failed with error: {str(e)}")
        raise

if __name__ == "__main__":
    main()

