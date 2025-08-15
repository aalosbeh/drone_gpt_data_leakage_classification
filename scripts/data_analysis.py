#!/usr/bin/env python3

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import warnings
warnings.filterwarnings('ignore')

# Set style for publication-quality plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class DroneDataAnalyzer:
    """
    Comprehensive analyzer for drone data leakage classification dataset.
    """
    
    def __init__(self, data_path):
        """
        Initialize the analyzer with dataset path.
        
        Args:
            data_path (str): Path to the CSV dataset file
        """
        self.data_path = data_path
        self.df = None
        self.X = None
        self.y = None
        self.X_train = None
        self.X_val = None
        self.X_test = None
        self.y_train = None
        self.y_val = None
        self.y_test = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        
    def load_data(self):
        """Load and perform initial examination of the dataset."""
        print("=" * 80)
        print("DRONE DATA LEAKAGE CLASSIFICATION - DATASET ANALYSIS")
        print("=" * 80)
        
        self.df = pd.read_csv(self.data_path)
        
        print(f"\nDataset Shape: {self.df.shape}")
        print(f" Features: {self.df.shape[1] - 1}")  # Excluding target variable
        print(f" Samples: {self.df.shape[0]}")
        
        print("\nDataset Info:")
        print(self.df.info())
        
        print("\n First 5 rows:")
        print(self.df.head())
        
        print("\n Statistical Summary:")
        print(self.df.describe())
        
        return self.df
    
    def analyze_target_distribution(self):
        """Analyze the distribution of target labels."""
        print("\n" + "=" * 50)
        print("TARGET VARIABLE ANALYSIS")
        print("=" * 50)
        
        # Label distribution
        label_counts = self.df['label'].value_counts()
        print(f"\n Label Distribution:")
        for label, count in label_counts.items():
            percentage = (count / len(self.df)) * 100
            print(f"   {label}: {count} samples ({percentage:.2f}%)")
        
        # Create visualization
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Bar plot
        label_counts.plot(kind='bar', ax=ax1, color=['#FF6B6B', '#4ECDC4'])
        ax1.set_title('Distribution of Risk Groups', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Risk Group', fontsize=12)
        ax1.set_ylabel('Number of Samples', fontsize=12)
        ax1.tick_params(axis='x', rotation=45)
        
        # Pie chart
        ax2.pie(label_counts.values, labels=label_counts.index, autopct='%1.1f%%',
                colors=['#FF6B6B', '#4ECDC4'], startangle=90)
        ax2.set_title('Risk Group Distribution', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig('/home/ubuntu/drone_classification_research/figures/target_distribution.png', 
                   dpi=300, bbox_inches='tight')
        plt.show()
        
        return label_counts
    
    def analyze_features(self):
        """Perform comprehensive feature analysis."""
        print("\n" + "=" * 50)
        print("FEATURE ANALYSIS")
        print("=" * 50)
        
        # Identify feature columns (exclude non-feature columns)
        feature_cols = [col for col in self.df.columns if col not in ['Group', 'StartIndex', 'EndIndex', 'label']]
        
        print(f"\n Feature Columns ({len(feature_cols)}):")
        for i, col in enumerate(feature_cols, 1):
            print(f"   {i:2d}. {col}")
        
        # Missing values analysis
        print(f"\n Missing Values Analysis:")
        missing_values = self.df[feature_cols].isnull().sum()
        if missing_values.sum() == 0:
            print("    No missing values found!")
        else:
            print(missing_values[missing_values > 0])
        
        # Feature statistics by group
        print(f"\n Feature Statistics by Risk Group:")
        for label in self.df['label'].unique():
            print(f"\n   {label.upper()}:")
            subset = self.df[self.df['label'] == label][feature_cols]
            print(f"      Samples: {len(subset)}")
            print(f"      Mean values (top 5 features):")
            means = subset.mean().sort_values(ascending=False)
            for feat, val in means.head().items():
                print(f"         {feat}: {val:.6f}")
        
        return feature_cols
    
    def create_feature_visualizations(self, feature_cols):
        """Create comprehensive feature visualizations."""
        print("\n" + "=" * 50)
        print("CREATING FEATURE VISUALIZATIONS")
        print("=" * 50)
        
        # Select key IAT features for detailed analysis
        key_features = ['Mean_IAT', 'Median_IAT', 'STD_IAT', 'Entropy_IAT', 'Packet_Count']
        available_features = [f for f in key_features if f in feature_cols]
        
        if len(available_features) < len(key_features):
            print(f"⚠️  Some key features not found. Using available: {available_features}")
        
        # Feature distribution plots
        n_features = len(available_features)
        n_cols = 3
        n_rows = (n_features + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 6*n_rows))
        if n_rows == 1:
            axes = axes.reshape(1, -1)
        
        for i, feature in enumerate(available_features):
            row = i // n_cols
            col = i % n_cols
            ax = axes[row, col]
            
            # Create histogram for each group
            for label in self.df['label'].unique():
                data = self.df[self.df['label'] == label][feature]
                ax.hist(data, alpha=0.7, label=label, bins=30)
            
            ax.set_title(f'Distribution of {feature}', fontsize=12, fontweight='bold')
            ax.set_xlabel(feature, fontsize=10)
            ax.set_ylabel('Frequency', fontsize=10)
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # Hide empty subplots
        for i in range(len(available_features), n_rows * n_cols):
            row = i // n_cols
            col = i % n_cols
            axes[row, col].set_visible(False)
        
        plt.tight_layout()
        plt.savefig('/home/ubuntu/drone_classification_research/figures/feature_distributions.png', 
                   dpi=300, bbox_inches='tight')
        plt.show()
        
        # Correlation heatmap
        plt.figure(figsize=(14, 10))
        correlation_matrix = self.df[feature_cols].corr()
        
        # Create mask for upper triangle
        mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
        
        sns.heatmap(correlation_matrix, mask=mask, annot=True, cmap='coolwarm', 
                   center=0, square=True, linewidths=0.5, cbar_kws={"shrink": .8})
        plt.title('Feature Correlation Matrix', fontsize=16, fontweight='bold', pad=20)
        plt.tight_layout()
        plt.savefig('/home/ubuntu/drone_classification_research/figures/correlation_matrix.png', 
                   dpi=300, bbox_inches='tight')
        plt.show()
        
        # Box plots for key features
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        for i, feature in enumerate(available_features[:6]):
            if i < len(axes):
                sns.boxplot(data=self.df, x='label', y=feature, ax=axes[i])
                axes[i].set_title(f'{feature} by Risk Group', fontsize=12, fontweight='bold')
                axes[i].tick_params(axis='x', rotation=45)
        
        # Hide unused subplots
        for i in range(len(available_features), len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        plt.savefig('/home/ubuntu/drone_classification_research/figures/feature_boxplots.png', 
                   dpi=300, bbox_inches='tight')
        plt.show()
    
    def preprocess_data(self, feature_cols):
        """Preprocess the data for machine learning."""
        print("\n" + "=" * 50)
        print("DATA PREPROCESSING")
        print("=" * 50)
        
        # Prepare features and target
        self.X = self.df[feature_cols].copy()
        self.y = self.df['label'].copy()
        
        print(f" Features prepared: {self.X.shape}")
        print(f" Target prepared: {self.y.shape}")
        
        # Encode labels
        self.y_encoded = self.label_encoder.fit_transform(self.y)
        print(f"Label encoding: {dict(zip(self.label_encoder.classes_, range(len(self.label_encoder.classes_))))}")
        
        # Split the data
        X_temp, self.X_test, y_temp, self.y_test = train_test_split(
            self.X, self.y_encoded, test_size=0.2, random_state=42, stratify=self.y_encoded
        )
        
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(
            X_temp, y_temp, test_size=0.25, random_state=42, stratify=y_temp
        )
        
        print(f" Data split completed:")
        print(f"   Training set: {self.X_train.shape[0]} samples")
        print(f"   Validation set: {self.X_val.shape[0]} samples")
        print(f"   Test set: {self.X_test.shape[0]} samples")
        
        # Normalize features
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_val_scaled = self.scaler.transform(self.X_val)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        
        print(f" Feature normalization completed")
        print(f"   Feature means: {self.scaler.mean_[:5]}...")
        print(f"   Feature stds: {self.scaler.scale_[:5]}...")
        
        return self.X_train_scaled, self.X_val_scaled, self.X_test_scaled, self.y_train, self.y_val, self.y_test
    
    def save_preprocessed_data(self):
        """Save preprocessed data for model training."""
        print("\n" + "=" * 50)
        print("SAVING PREPROCESSED DATA")
        print("=" * 50)
        
        # Save training data
        np.save('/home/ubuntu/drone_classification_research/data/X_train.npy', self.X_train_scaled)
        np.save('/home/ubuntu/drone_classification_research/data/X_val.npy', self.X_val_scaled)
        np.save('/home/ubuntu/drone_classification_research/data/X_test.npy', self.X_test_scaled)
        np.save('/home/ubuntu/drone_classification_research/data/y_train.npy', self.y_train)
        np.save('/home/ubuntu/drone_classification_research/data/y_val.npy', self.y_val)
        np.save('/home/ubuntu/drone_classification_research/data/y_test.npy', self.y_test)
        
        # Save original data splits
        self.X_train.to_csv('/home/ubuntu/drone_classification_research/data/X_train_original.csv', index=False)
        self.X_val.to_csv('/home/ubuntu/drone_classification_research/data/X_val_original.csv', index=False)
        self.X_test.to_csv('/home/ubuntu/drone_classification_research/data/X_test_original.csv', index=False)
        
        # Save metadata
        metadata = {
            'feature_columns': list(self.X.columns),
            'label_encoding': dict(zip(self.label_encoder.classes_, range(len(self.label_encoder.classes_)))),
            'data_splits': {
                'train_size': len(self.X_train),
                'val_size': len(self.X_val),
                'test_size': len(self.X_test)
            },
            'normalization_params': {
                'mean': self.scaler.mean_.tolist(),
                'scale': self.scaler.scale_.tolist()
            }
        }
        
        import json
        with open('/home/ubuntu/drone_classification_research/data/metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(" All preprocessed data saved successfully!")
        print("   Normalized arrays: X_train.npy, X_val.npy, X_test.npy")
        print("    Target arrays: y_train.npy, y_val.npy, y_test.npy")
        print("    Original splits: X_train_original.csv, X_val_original.csv, X_test_original.csv")
        print("    Metadata: metadata.json")
    
    def generate_analysis_report(self, feature_cols):
        """Generate comprehensive analysis report."""
        print("\n" + "=" * 50)
        print("GENERATING ANALYSIS REPORT")
        print("=" * 50)
        
        report = []
        report.append("# Drone Data Leakage Classification - Dataset Analysis Report")
        report.append("=" * 70)
        report.append("")
        report.append("## Dataset Overview")
        report.append(f"- **Total Samples**: {self.df.shape[0]}")
        report.append(f"- **Total Features**: {len(feature_cols)}")
        report.append(f"- **Target Classes**: {len(self.df['label'].unique())}")
        report.append("")
        
        # Class distribution
        report.append("## Class Distribution")
        label_counts = self.df['label'].value_counts()
        for label, count in label_counts.items():
            percentage = (count / len(self.df)) * 100
            report.append(f"- **{label}**: {count} samples ({percentage:.2f}%)")
        report.append("")
        
        # Feature summary
        report.append("## Feature Summary")
        report.append("### IAT-based Features:")
        iat_features = [f for f in feature_cols if 'IAT' in f]
        for feat in iat_features:
            report.append(f"- {feat}")
        report.append("")
        
        report.append("### Other Features:")
        other_features = [f for f in feature_cols if 'IAT' not in f]
        for feat in other_features:
            report.append(f"- {feat}")
        report.append("")
        
        # Data quality
        report.append("## Data Quality Assessment")
        missing_count = self.df[feature_cols].isnull().sum().sum()
        report.append(f"- **Missing Values**: {missing_count}")
        report.append(f"- **Data Completeness**: {((len(self.df) * len(feature_cols) - missing_count) / (len(self.df) * len(feature_cols))) * 100:.2f}%")
        report.append("")
        
        # Statistical insights
        report.append("## Statistical Insights")
        for label in self.df['label'].unique():
            subset = self.df[self.df['label'] == label][feature_cols]
            report.append(f"### {label.upper()}")
            report.append(f"- **Sample Count**: {len(subset)}")
            report.append(f"- **Mean IAT**: {subset['Mean_IAT'].mean():.6f}")
            report.append(f"- **Mean Entropy**: {subset['Entropy_IAT'].mean():.6f}")
            report.append("")
        
        # Save report
        with open('/home/ubuntu/drone_classification_research/outputs/dataset_analysis_report.md', 'w') as f:
            f.write('\n'.join(report))
        
        print(" Analysis report generated: outputs/dataset_analysis_report.md")
        
        return report

def main():
    """Main execution function."""
    # Initialize analyzer
    analyzer = DroneDataAnalyzer('/home/ubuntu/drone_classification_research/data/dataset_64.csv')
    
    # Load and analyze data
    df = analyzer.load_data()
    
    # Analyze target distribution
    analyzer.analyze_target_distribution()
    
    # Analyze features
    feature_cols = analyzer.analyze_features()
    
    # Create visualizations
    analyzer.create_feature_visualizations(feature_cols)
    
    # Preprocess data
    analyzer.preprocess_data(feature_cols)
    
    # Save preprocessed data
    analyzer.save_preprocessed_data()
    
    # Generate report
    analyzer.generate_analysis_report(feature_cols)
    
    print("\n" + "=" * 80)
    print(" DATASET ANALYSIS AND PREPROCESSING COMPLETED SUCCESSFULLY!")
    print("=" * 80)
    print("\n Next Steps:")
    print("   1. Implement GPT model for classification")
    print("   2. Evaluate pre-tuning performance")
    print("   3. Fine-tune the model")
    print("   4. Compare pre vs post-tuning results")

if __name__ == "__main__":
    main()


