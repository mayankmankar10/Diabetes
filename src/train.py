import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import pickle
import yaml
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from mlflow.models import infer_signature
from sklearn.model_selection import train_test_split, GridSearchCV
import os
from urllib.parse import urlparse
import mlflow

# Set MLflow Tracking URI
os.environ["MLFLOW_TRACKING_URI"] = "https://dagshub.com/mayankmankar10/machinelearningpipeline.mlflow"
os.environ["MLFLOW_TRACKING_USERNAME"] = "mayankmankar10"
os.environ["MLFLOW_TRACKING_PASSWORD"] = "ee3a1d10d9253999191b3320f4d6ae4a1fce587c"

# Function to perform hyperparameter tuning
def hyperparameter_tuning(X_train, y_train, param_grid):
    rf = RandomForestClassifier()  # Fixed instantiation
    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2)
    grid_search.fit(X_train, y_train)
    return grid_search

# Load parameters from params.yaml
params = yaml.safe_load(open("params.yaml"))["train"]

# Train function
def train(data_path, model_path, random_state, n_estimators, max_depth):
    data = pd.read_csv(data_path)
    X = data.drop(columns=["Outcome"])  # Fixed typo in 'columns'
    y = data["Outcome"]

    # Set MLflow tracking
    mlflow.set_tracking_uri("https://dagshub.com/mayankmankar10/machinelearningpipeline.mlflow")

    with mlflow.start_run():
        # Train-Test Split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=random_state)
        signature = infer_signature(X_train, y_train)

        # Define parameter grid
        param_grid = {
            'n_estimators': [100, 200],  # Fixed key name
            'max_depth': [5, 10, None],
            'min_samples_leaf': [1, 2],  # Fixed key name
            'min_samples_split': [2, 5]  # Fixed key name
        }

        # Perform hyperparameter tuning
        grid_search = hyperparameter_tuning(X_train, y_train, param_grid)
        best_model = grid_search.best_estimator_

        # Predictions
        y_pred = best_model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)  # Fixed variable name
        print(f"Accuracy: {accuracy}")

        # Log metrics & parameters
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_param("best_n_estimators", grid_search.best_params_['n_estimators'])
        mlflow.log_param("best_max_depth", grid_search.best_params_['max_depth'])
        mlflow.log_param("best_min_samples_leaf", grid_search.best_params_['min_samples_leaf'])
        mlflow.log_param("best_min_samples_split", grid_search.best_params_['min_samples_split'])

        # Log confusion matrix & classification report
        cm = confusion_matrix(y_test, y_pred)
        cr = classification_report(y_test, y_pred)

        mlflow.log_text(str(cm), "confusion_matrix.txt")
        mlflow.log_text(cr, "classification_report.txt")

        # Log model based on MLflow tracking type
        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme
        if tracking_url_type_store != 'file':
            mlflow.sklearn.log_model(best_model, "model", registered_model_name="Best Model")
        else:
            mlflow.sklearn.log_model(best_model, "model", signature=signature)

        # Save model locally
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        pickle.dump(best_model, open(model_path, "wb"))

# Run the training function
if __name__ == "__main__":
    train(params['data'], params['model'], params['random_state'], params['n_estimators'], params['max_depth'])


