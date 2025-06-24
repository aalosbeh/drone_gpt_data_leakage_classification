#!/usr/bin/env python3
"""
Data Utilities for Drone Data Leakage Classification
Authors: Anas AlSobeh (Southern Illinois University Carbondale)
         Omar Darwish (Eastern Michigan University)

This module provides utilities for data loading, preprocessing, and handling.
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import json
import yaml
import logging
from typing import Tuple, Dict, Any, Optional

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DroneDataset(Dataset):
    """
    PyTorch Dataset class for drone data leakage classification.
    """
    
    def __init__(self, features: np.ndarray, labels: np.ndarray, tokenizer=None, max_length: int = 512):
        """
        Initialize the dataset.
        
        Args:
            features (np.ndarray): Feature array
            labels (np.ndarray): Label array
            tokenizer: Tokenizer for text processing (optional)
            max_length (int): Maximum sequence length
        """
        self.features = torch.FloatTensor(features)
        self.labels = torch.LongTensor(labels)
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self) -> int:
        """Return the length of the dataset."""
        return len(self.features)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single item from the dataset.
        
        Args:
            idx (int): Index of the item
            
        Returns:
            Dict[str, torch.Tensor]: Dictionary containing features and labels
        """
        feature = self.features[idx]
        label = self.labels[idx]
        
        # Convert features to text representation for GPT
        if self.tokenizer is not None:
            # Create a text representation of the features
            feature_text = self._features_to_text(feature)
            
            # Tokenize the text
            encoding = self.tokenizer(
                feature_text,
                truncation=True,
                padding='max_length',
                max_length=self.max_length,
                return_tensors='pt'
            )
            
            return {
                'input_ids': encoding['input_ids'].flatten(),
                'attention_mask': encoding['attention_mask'].flatten(),
                'labels': label,
                'features': feature
            }
        else:
            return {
                'features': feature,
                'labels': label
            }
    
    def _features_to_text(self, features: torch.Tensor) -> str:
        """
        Convert feature vector to text representation for GPT processing.
        
        Args:
            features (torch.Tensor): Feature tensor
            
        Returns:
            str: Text representation of features
        """
        feature_names = [
            "Mean_IAT", "Median_IAT", "Min_IAT", "Max_IAT", "STD_IAT",
            "Variance_IAT", "Entropy_IAT", "Packet_Count", "IAT_Range",
            "IAT_CV", "IAT_Ratio", "Log_Mean_IAT", "Log_Entropy_IAT", "Log_Packet_Count"
        ]
        
        # Create a descriptive text representation
        text_parts = ["Drone network traffic analysis:"]
        
        for i, (name, value) in enumerate(zip(feature_names, features)):
            if i < len(feature_names):
                text_parts.append(f"{name}: {value:.6f}")
        
        text_parts.append("Classification task: Determine data leakage risk level.")
        
        return " ".join(text_parts)

class DataLoader:
    """
    Data loading and preprocessing utilities.
    """
    
    def __init__(self, config_path: str = "config.yaml"):
        """
        Initialize the data loader.
        
        Args:
            config_path (str): Path to configuration file
        """
        self.config = self._load_config(config_path)
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        
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
            'dataset': {
                'path': 'data/dataset_64.csv',
                'target_column': 'label',
                'train_size': 0.6,
                'val_size': 0.2,
                'test_size': 0.2,
                'random_state': 42
            }
        }
    
    def load_raw_data(self, data_path: Optional[str] = None) -> pd.DataFrame:
        """
        Load raw data from CSV file.
        
        Args:
            data_path (str, optional): Path to data file
            
        Returns:
            pd.DataFrame: Raw data
        """
        if data_path is None:
            data_path = self.config['dataset']['path']
        
        logger.info(f"Loading data from {data_path}")
        df = pd.read_csv(data_path)
        logger.info(f"Loaded {len(df)} samples with {df.shape[1]} columns")
        
        return df
    
    def preprocess_data(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Preprocess the data for training.
        
        Args:
            df (pd.DataFrame): Raw data
            
        Returns:
            Tuple: (X_train, X_val, X_test, y_train, y_val, y_test)
        """
        # Extract features and target
        feature_cols = [col for col in df.columns if col not in ['Group', 'StartIndex', 'EndIndex', 'label']]
        X = df[feature_cols].values
        y = df[self.config['dataset']['target_column']].values
        
        # Encode labels
        y_encoded = self.label_encoder.fit_transform(y)
        
        # Split data
        train_size = self.config['dataset']['train_size']
        val_size = self.config['dataset']['val_size']
        random_state = self.config['dataset']['random_state']
        
        # First split: train + val, test
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y_encoded, 
            test_size=self.config['dataset']['test_size'],
            random_state=random_state,
            stratify=y_encoded
        )
        
        # Second split: train, val
        val_ratio = val_size / (train_size + val_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp,
            test_size=val_ratio,
            random_state=random_state,
            stratify=y_temp
        )
        
        # Normalize features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        X_test_scaled = self.scaler.transform(X_test)
        
        logger.info(f"Data preprocessing completed:")
        logger.info(f"  Training set: {X_train_scaled.shape}")
        logger.info(f"  Validation set: {X_val_scaled.shape}")
        logger.info(f"  Test set: {X_test_scaled.shape}")
        
        return X_train_scaled, X_val_scaled, X_test_scaled, y_train, y_val, y_test
    
    def create_dataloaders(self, X_train: np.ndarray, X_val: np.ndarray, X_test: np.ndarray,
                          y_train: np.ndarray, y_val: np.ndarray, y_test: np.ndarray,
                          tokenizer=None, batch_size: int = 16) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """
        Create PyTorch DataLoaders.
        
        Args:
            X_train, X_val, X_test: Feature arrays
            y_train, y_val, y_test: Label arrays
            tokenizer: Tokenizer for text processing
            batch_size (int): Batch size
            
        Returns:
            Tuple[DataLoader, DataLoader, DataLoader]: Train, validation, and test dataloaders
        """
        # Create datasets
        train_dataset = DroneDataset(X_train, y_train, tokenizer)
        val_dataset = DroneDataset(X_val, y_val, tokenizer)
        test_dataset = DroneDataset(X_test, y_test, tokenizer)
        
        # Create dataloaders
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, num_workers=2
        )
        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False, num_workers=2
        )
        test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False, num_workers=2
        )
        
        return train_loader, val_loader, test_loader
    
    def save_preprocessed_data(self, X_train: np.ndarray, X_val: np.ndarray, X_test: np.ndarray,
                              y_train: np.ndarray, y_val: np.ndarray, y_test: np.ndarray,
                              output_dir: str = "data") -> None:
        """
        Save preprocessed data to files.
        
        Args:
            X_train, X_val, X_test: Feature arrays
            y_train, y_val, y_test: Label arrays
            output_dir (str): Output directory
        """
        # Save arrays
        np.save(f"{output_dir}/X_train.npy", X_train)
        np.save(f"{output_dir}/X_val.npy", X_val)
        np.save(f"{output_dir}/X_test.npy", X_test)
        np.save(f"{output_dir}/y_train.npy", y_train)
        np.save(f"{output_dir}/y_val.npy", y_val)
        np.save(f"{output_dir}/y_test.npy", y_test)
        
        # Save metadata
        metadata = {
            'label_classes': self.label_encoder.classes_.tolist(),
            'scaler_mean': self.scaler.mean_.tolist(),
            'scaler_scale': self.scaler.scale_.tolist(),
            'data_shapes': {
                'train': X_train.shape,
                'val': X_val.shape,
                'test': X_test.shape
            }
        }
        
        with open(f"{output_dir}/preprocessing_metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Preprocessed data saved to {output_dir}")
    
    def load_preprocessed_data(self, data_dir: str = "data") -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Load preprocessed data from files.
        
        Args:
            data_dir (str): Data directory
            
        Returns:
            Tuple: (X_train, X_val, X_test, y_train, y_val, y_test)
        """
        X_train = np.load(f"{data_dir}/X_train.npy")
        X_val = np.load(f"{data_dir}/X_val.npy")
        X_test = np.load(f"{data_dir}/X_test.npy")
        y_train = np.load(f"{data_dir}/y_train.npy")
        y_val = np.load(f"{data_dir}/y_val.npy")
        y_test = np.load(f"{data_dir}/y_test.npy")
        
        # Load metadata
        with open(f"{data_dir}/preprocessing_metadata.json", 'r') as f:
            metadata = json.load(f)
        
        # Restore scaler
        self.scaler.mean_ = np.array(metadata['scaler_mean'])
        self.scaler.scale_ = np.array(metadata['scaler_scale'])
        
        # Restore label encoder
        self.label_encoder.classes_ = np.array(metadata['label_classes'])
        
        logger.info(f"Preprocessed data loaded from {data_dir}")
        
        return X_train, X_val, X_test, y_train, y_val, y_test

