import pandas as pd 
import pickle
from sklearn.metrics import accuracy_score
import yaml
import mlflow
import os
from urllib.parse import urlparse
import pickle

os.environ["MLFLOW_TRACKING_URI"] = "https://dagshub.com/mayankmankar10/machinelearningpipeline.mlflow"
os.environ["MLFLOW_TRACKING_USERNAME"] = "mayankmankar10"
os.environ["MLFLOW_TRACKING_PASSWORD"] = "ee3a1d10d9253999191b3320f4d6ae4a1fce587c"

params = yaml.safe_load(open("params.yaml"))["train"]

def evaluate(data_path, model_path):
    data = pd.read_csv(data_path)

    X = data.drop(columns=['Outcome'])
    y = data["Outcome"]

    mlflow.set_tracking_uri("https://dagshub.com/mayankmankar10/machinelearningpipeline.mlflow")

    model = pickle.load(open(model_path, 'rb'))

    predictions = model.predict(X)

    accuracy = accuracy_score(y, predictions)

if __name__ == "__main__":
    evaluate(params["data"], params["model"])



