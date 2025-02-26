
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score
import mlflow
import pickle
import yaml
import os
from urllib.parse import urlparse
import json
from datetime import datetime

# Set MLflow tracking URI
os.environ["MLFLOW_TRACKING_URI"] = "https://dagshub.com/mayankmankar10/machinelearningpipeline.mlflow"
os.environ["MLFLOW_TRACKING_USERNAME"] = "mayankmankar10"
os.environ["MLFLOW_TRACKING_PASSWORD"] = "ee3a1d10d9253999191b3320f4d6ae4a1fce587c"

def create_model(random_state):
    """Create base model with initial parameters"""
    return RandomForestClassifier(
        random_state=random_state,
        n_jobs=-1,
        oob_score=True  # Enable out-of-bag score estimation
    )

def hyperparameter_tuning(X_train, y_train, random_state):
    """Perform hyperparameter tuning with cross-validation"""
    model = create_model(random_state)
    
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [5, 10, 15, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2'],
        'class_weight': ['balanced', 'balanced_subsample', None]
    }
    
    # Use stratified k-fold to maintain class distribution
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    
    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=cv,
        n_jobs=-1,
        verbose=2,
        scoring=['accuracy', 'roc_auc', 'precision', 'recall'],
        refit='roc_auc'  # Use ROC-AUC for selecting best model
    )
    
    grid_search.fit(X_train, y_train)
    return grid_search

def log_feature_importance(model, feature_names, artifact_path):
    """Log feature importance plot and scores"""
    importance_dict = dict(zip(feature_names, model.feature_importances_))
    sorted_importance = dict(sorted(importance_dict.items(), key=lambda x: x[1], reverse=True))
    
    with open(artifact_path, 'w') as f:
        json.dump(sorted_importance, f, indent=2)
    
    return sorted_importance

def train(data_path, model_path, random_state):
    """Main training function with improved validation and logging"""
    # Load preprocessed data
    print("Loading preprocessed data...")
    data = pd.read_csv(data_path)
    
    X = data.drop(columns=["Outcome"])
    y = data["Outcome"]
    
    # Split data with stratification
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=0.20, 
        random_state=random_state,
        stratify=y
    )
    
    # Start MLflow run
    mlflow.set_tracking_uri("https://dagshub.com/mayankmankar10/machinelearningpipeline.mlflow")
    
    with mlflow.start_run(run_name=f"RF_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
        # Initial cross-validation before tuning
        base_model = create_model(random_state)
        cv_scores = cross_val_score(base_model, X_train, y_train, cv=5)
        print(f"Initial CV scores: {cv_scores}")
        print(f"Mean CV score: {cv_scores.mean():.4f} (±{cv_scores.std() * 2:.4f})")
        
        # Perform hyperparameter tuning
        print("\nPerforming hyperparameter tuning...")
        grid_search = hyperparameter_tuning(X_train, y_train, random_state)
        best_model = grid_search.best_estimator_
        
        # Get predictions
        train_pred = best_model.predict(X_train)
        test_pred = best_model.predict(X_test)
        
        # Calculate metrics
        train_accuracy = accuracy_score(y_train, train_pred)
        test_accuracy = accuracy_score(y_test, test_pred)
        test_roc_auc = roc_auc_score(y_test, best_model.predict_proba(X_test)[:, 1])
        
        print(f"\nFinal Metrics:")
        print(f"Training Accuracy: {train_accuracy:.4f}")
        print(f"Test Accuracy: {test_accuracy:.4f}")
        print(f"Test ROC-AUC: {test_roc_auc:.4f}")
        
        # Log metrics
        mlflow.log_metric("train_accuracy", train_accuracy)
        mlflow.log_metric("test_accuracy", test_accuracy)
        mlflow.log_metric("test_roc_auc", test_roc_auc)
        mlflow.log_metric("cv_mean_accuracy", cv_scores.mean())
        mlflow.log_metric("cv_std_accuracy", cv_scores.std())
        
        # Log best parameters
        for param, value in grid_search.best_params_.items():
            mlflow.log_param(param, value)
        
        # Log feature importance
        importance_dict = log_feature_importance(
            best_model, 
            X.columns, 
            "feature_importance.json"
        )
        mlflow.log_artifact("feature_importance.json")
        
        # Log confusion matrix and classification report
        test_cm = confusion_matrix(y_test, test_pred)
        test_cr = classification_report(y_test, test_pred)
        
        mlflow.log_text(str(test_cm), "test_confusion_matrix.txt")
        mlflow.log_text(test_cr, "test_classification_report.txt")
        
        # Save model
        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme
        if tracking_url_type_store != "file":
            mlflow.sklearn.log_model(
                best_model, 
                "model",
                registered_model_name="DiabetesPredictionModel"
            )
        else:
            mlflow.sklearn.log_model(best_model, "model")
        
        # Save model locally
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        with open(model_path, "wb") as f:
            pickle.dump(best_model, f)
        
        print(f"\nModel saved to {model_path}")
        print("Feature Importance:", importance_dict)

if __name__ == "__main__":
    # Load parameters
    params = yaml.safe_load(open("params.yaml"))["train"]
    train(params['data'], params['model'], params['random_state'])
