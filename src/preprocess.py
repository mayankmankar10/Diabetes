
import pandas as pd
import numpy as np
import yaml
import os
from sklearn.preprocessing import StandardScaler
import joblib

def handle_missing_values(data):
    """Handle missing and invalid values in the dataset"""
    # Replace 0 values with NaN for columns where 0 doesn't make sense
    zero_columns = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
    data[zero_columns] = data[zero_columns].replace(0, np.nan)
    
    # Fill missing values with median for each column
    for column in zero_columns:
        median_value = data[column].median()
        data[column] = data[column].fillna(median_value)
        
    return data

def remove_outliers(data, columns, n_std=3):
    """Remove outliers using the z-score method"""
    for column in columns:
        z_scores = np.abs((data[column] - data[column].mean()) / data[column].std())
        data = data[z_scores < n_std]
    return data

def preprocess(input_path, output_path, scaler_path):
    """Main preprocessing function"""
    # Load data
    print("Loading data...")
    data = pd.read_csv(input_path)
    
    # Handle missing values
    print("Handling missing values...")
    data = handle_missing_values(data)
    
    # Remove outliers from numerical columns
    print("Removing outliers...")
    numerical_columns = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction']
    data = remove_outliers(data, numerical_columns)
    
    # Split features and target
    X = data.drop(columns=['Outcome'])
    y = data['Outcome']
    
    # Scale features
    print("Scaling features...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Convert back to DataFrame
    X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)
    
    # Add target back
    final_data = X_scaled_df.copy()
    final_data['Outcome'] = y.values
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    os.makedirs(os.path.dirname(scaler_path), exist_ok=True)
    
    # Save preprocessed data and scaler
    print(f"Saving preprocessed data to {output_path}")
    final_data.to_csv(output_path, index=False)
    
    print(f"Saving scaler to {scaler_path}")
    joblib.dump(scaler, scaler_path)
    
    # Print preprocessing summary
    print("\nPreprocessing Summary:")
    print(f"Original data shape: {len(data)}")
    print(f"Preprocessed data shape: {len(final_data)}")
    print("Feature statistics after preprocessing:")
    print(final_data.describe())

if __name__ == "__main__":
    # Load parameters
    params = yaml.safe_load(open("params.yaml"))["preprocess"]
    
    # Execute preprocessing
    preprocess(
        input_path=params["input"],
        output_path=params["output"],
        scaler_path=params["scaler_path"]
    )
