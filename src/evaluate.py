import pandas as pd
import numpy as np
import pickle
import yaml
import mlflow
import os
from urllib.parse import urlparse
from sklearn.metrics import (
    accuracy_score, 
    classification_report, 
    confusion_matrix,
    roc_curve, 
    auc,
    precision_recall_curve,
    average_precision_score
)
import json
from datetime import datetime

# Set MLflow tracking URI
os.environ["MLFLOW_TRACKING_URI"] = "https://dagshub.com/mayankmankar10/machinelearningpipeline.mlflow"
os.environ["MLFLOW_TRACKING_USERNAME"] = "mayankmankar10"
os.environ["MLFLOW_TRACKING_PASSWORD"] = "ee3a1d10d9253999191b3320f4d6ae4a1fce587c"

def calculate_metrics(y_true, y_pred, y_prob):
    """Calculate and return comprehensive metrics"""
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'confusion_matrix': confusion_matrix(y_true, y_pred),
        'classification_report': classification_report(y_true, y_pred),
        'average_precision': average_precision_score(y_true, y_prob),
    }
    
    # Calculate ROC curve and AUC
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    metrics['roc_auc'] = auc(fpr, tpr)
    
    # Calculate Precision-Recall curve
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    metrics['pr_auc'] = auc(recall, precision)
    
    return metrics

def analyze_predictions(y_true, y_pred, data):
    """Analyze predictions to identify patterns in errors"""
    # Create DataFrame with actual and predicted values
    analysis_df = data.copy()
    analysis_df['actual'] = y_true
    analysis_df['predicted'] = y_pred
    analysis_df['correct'] = y_true == y_pred
    
    # Analyze errors by feature ranges
    error_analysis = {}
    for column in data.columns:
        if column != 'Outcome':
            # Calculate error rate by feature quartiles
            quartiles = pd.qcut(data[column], q=4, duplicates='drop')
            error_rates = 1 - analysis_df.groupby(quartiles)['correct'].mean()
            error_analysis[column] = error_rates.to_dict()
    
    return error_analysis

def evaluate(data_path, model_path):
    """Comprehensive model evaluation function"""
    print("Loading test data and model...")
    data = pd.read_csv(data_path)
    
    # Ensure 'Outcome' is in the dataset
    if 'Outcome' not in data.columns:
        raise ValueError(f"Target column 'Outcome' not found. Available columns: {data.columns.tolist()}")
    
    # Split features and target
    X = data.drop(columns=['Outcome'])
    y = data['Outcome']
    
    # Load model
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    
    # Get predictions and probabilities
    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)[:, 1]
    
    # Calculate all metrics
    print("\nCalculating metrics...")
    metrics = calculate_metrics(y, y_pred, y_prob)
    
    # Print detailed evaluation results
    print("\n=== Model Evaluation Results ===")
    print(f"\nAccuracy: {metrics['accuracy']:.4f}")
    print(f"ROC AUC: {metrics['roc_auc']:.4f}")
    print(f"PR AUC: {metrics['pr_auc']:.4f}")
    print(f"Average Precision: {metrics['average_precision']:.4f}")
    
    print("\nConfusion Matrix:")
    print(metrics['confusion_matrix'])
    
    print("\nClassification Report:")
    print(metrics['classification_report'])
    
    # Analyze prediction errors
    print("\nAnalyzing prediction errors...")
    error_analysis = analyze_predictions(y, y_pred, data)
    
    print("\nError Analysis by Feature Ranges:")
    for feature, error_rates in error_analysis.items():
        print(f"\n{feature}:")
        for range_label, error_rate in error_rates.items():
            print(f"  {range_label}: {error_rate:.4f}")
    
    # Log results to MLflow
    print("\nLogging results to MLflow...")
    with mlflow.start_run(run_name=f"model_evaluation_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
        # Log metrics
        mlflow.log_metric("test_accuracy", metrics['accuracy'])
        mlflow.log_metric("test_roc_auc", metrics['roc_auc'])
        mlflow.log_metric("test_pr_auc", metrics['pr_auc'])
        mlflow.log_metric("test_avg_precision", metrics['average_precision'])
        
        # Log confusion matrix and classification report
        mlflow.log_text(str(metrics['confusion_matrix']), "confusion_matrix.txt")
        mlflow.log_text(metrics['classification_report'], "classification_report.txt")
        
        # Log error analysis
        with open("error_analysis.json", "w") as f:
            json.dump(error_analysis, f, indent=2)
        mlflow.log_artifact("error_analysis.json")
    
    print("\nEvaluation complete! Results have been logged to MLflow.")

# ... existing code ...

if __name__ == "__main__":
    # Load parameters from params.yaml
    try:
        params = yaml.safe_load(open("params.yaml"))["evaluate"]
        print("Starting evaluation...")
        print(f"Data path: {params['data']}")
        print(f"Model path: {params['model']}")
        
        evaluate(params["data"], params["model"])
    except Exception as e:
        print(f"Error during evaluation: {str(e)}")
