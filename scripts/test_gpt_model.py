#!/usr/bin/env python3
"""
Simplified GPT Model Test
Authors: Anas AlSobeh (Southern Illinois University Carbondale)
         Omar Darwish (Eastern Michigan University)

This script tests the GPT model implementation with a small sample.
"""

import os
import sys
import torch
import numpy as np
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_gpt_model():
    """Test GPT model implementation with a small sample."""
    logger.info("Testing GPT model implementation...")
    
    # Check device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Load a smaller model for testing
    model_name = "gpt2"
    logger.info(f"Loading {model_name}...")
    
    try:
        # Load tokenizer and model
        tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        tokenizer.pad_token = tokenizer.eos_token
        
        model = GPT2LMHeadModel.from_pretrained(model_name)
        model.to(device)
        model.eval()
        
        logger.info("Model loaded successfully!")
        logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        # Test with sample data
        sample_text = "Drone network traffic analysis: Mean_IAT: 0.001234 Median_IAT: 0.000987 Entropy_IAT: 2.345 Classification task: Determine data leakage risk level."
        
        # Tokenize
        inputs = tokenizer(sample_text, return_tensors="pt", padding=True, truncation=True, max_length=512)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Forward pass
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            
        logger.info(f"Forward pass successful!")
        logger.info(f"Output shape: {logits.shape}")
        
        # Test classification head simulation
        # Get the last token's representation
        last_hidden_state = model.transformer(**inputs).last_hidden_state
        pooled_output = last_hidden_state[:, -1, :]  # Use last token
        
        # Simulate classification head
        classifier = torch.nn.Linear(model.config.hidden_size, 2).to(device)
        class_logits = classifier(pooled_output)
        
        # Get predictions
        probabilities = torch.softmax(class_logits, dim=-1)
        prediction = torch.argmax(class_logits, dim=-1)
        
        logger.info(f"Classification test successful!")
        logger.info(f"Class probabilities: {probabilities.detach().cpu().numpy()}")
        logger.info(f"Prediction: {prediction.detach().cpu().numpy()}")
        
        return True
        
    except Exception as e:
        logger.error(f"Error testing GPT model: {e}")
        return False

def test_data_loading():
    """Test data loading functionality."""
    logger.info("Testing data loading...")
    
    try:
        # Load preprocessed data
        X_test = np.load("data/X_test.npy")
        y_test = np.load("data/y_test.npy")
        
        logger.info(f"Data loaded successfully!")
        logger.info(f"X_test shape: {X_test.shape}")
        logger.info(f"y_test shape: {y_test.shape}")
        logger.info(f"Unique labels: {np.unique(y_test)}")
        
        # Test feature to text conversion
        sample_features = X_test[0]
        feature_names = [
            "Mean_IAT", "Median_IAT", "Min_IAT", "Max_IAT", "STD_IAT",
            "Variance_IAT", "Entropy_IAT", "Packet_Count", "IAT_Range",
            "IAT_CV", "IAT_Ratio", "Log_Mean_IAT", "Log_Entropy_IAT", "Log_Packet_Count"
        ]
        
        # Create text representation
        text_parts = ["Drone network traffic analysis:"]
        for name, value in zip(feature_names, sample_features):
            text_parts.append(f"{name}: {value:.6f}")
        text_parts.append("Classification task: Determine data leakage risk level.")
        
        sample_text = " ".join(text_parts)
        logger.info(f"Sample text representation: {sample_text[:200]}...")
        
        return True
        
    except Exception as e:
        logger.error(f"Error testing data loading: {e}")
        return False

def test_baseline_models():
    """Test baseline model implementations."""
    logger.info("Testing baseline models...")
    
    try:
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import accuracy_score
        
        # Load data
        X_test = np.load("data/X_test.npy")
        y_test = np.load("data/y_test.npy")
        
        # Use small subset for quick test
        n_samples = min(100, len(X_test))
        X_subset = X_test[:n_samples]
        y_subset = y_test[:n_samples]
        
        # Test Random Forest
        rf = RandomForestClassifier(n_estimators=10, random_state=42)
        rf.fit(X_subset, y_subset)
        rf_pred = rf.predict(X_subset)
        rf_acc = accuracy_score(y_subset, rf_pred)
        
        logger.info(f"Random Forest accuracy (training): {rf_acc:.4f}")
        
        # Test Logistic Regression
        lr = LogisticRegression(random_state=42, max_iter=100)
        lr.fit(X_subset, y_subset)
        lr_pred = lr.predict(X_subset)
        lr_acc = accuracy_score(y_subset, lr_pred)
        
        logger.info(f"Logistic Regression accuracy (training): {lr_acc:.4f}")
        
        return True
        
    except Exception as e:
        logger.error(f"Error testing baseline models: {e}")
        return False

def main():
    """Main test function."""
    logger.info("=" * 60)
    logger.info("GPT MODEL IMPLEMENTATION TEST")
    logger.info("=" * 60)
    
    tests = [
        ("Data Loading", test_data_loading),
        ("Baseline Models", test_baseline_models),
        ("GPT Model", test_gpt_model)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        logger.info(f"\n--- Testing {test_name} ---")
        try:
            results[test_name] = test_func()
        except Exception as e:
            logger.error(f"Test {test_name} failed with exception: {e}")
            results[test_name] = False
    
    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("TEST RESULTS SUMMARY")
    logger.info("=" * 60)
    
    for test_name, result in results.items():
        status = " PASSED" if result else " FAILED"
        logger.info(f"{test_name}: {status}")
    
    all_passed = all(results.values())
    if all_passed:
        logger.info("\n All tests passed! GPT model implementation is ready.")
    else:
        logger.info("\n️ Some tests failed. Please check the implementation.")
    
    return all_passed

if __name__ == "__main__":
    main()

