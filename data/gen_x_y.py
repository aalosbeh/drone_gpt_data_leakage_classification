import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import os

# Load dataset
df = pd.read_csv("dataset_64.csv")

# Feature and label selection
features = [
    'Mean_IAT', 'Median_IAT', 'Min_IAT', 'Max_IAT', 'STD_IAT',
    'Variance_IAT', 'Entropy_IAT', 'Packet_Count', 'IAT_Range',
    'IAT_CV', 'IAT_Ratio', 'Log_Mean_IAT', 'Log_Entropy_IAT', 'Log_Packet_Count'
]
X = df[features].values
y = df['label'].values

# Encode labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Normalize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split into train, val, test
X_train, X_temp, y_train, y_temp = train_test_split(X_scaled, y_encoded, test_size=0.3, random_state=42, stratify=y_encoded)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

# Save .npy files
output_path = "preprocessed"
os.makedirs(output_path, exist_ok=True)

np.save(f"{output_path}/X_train.npy", X_train)
np.save(f"{output_path}/y_train.npy", y_train)
np.save(f"{output_path}/X_val.npy", X_val)
np.save(f"{output_path}/y_val.npy", y_val)
np.save(f"{output_path}/X_test.npy", X_test)
np.save(f"{output_path}/y_test.npy", y_test)
