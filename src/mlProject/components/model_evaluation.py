import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from urllib.parse import urlparse
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, roc_curve, classification_report
from sklearn.metrics import ConfusionMatrixDisplay
import mlflow
import mlflow.sklearn
from mlProject.entity.config_entity import ModelEvaluationConfig
from mlProject.utils.common import save_json
from pathlib import Path


class ModelEvaluation:
    def __init__(self, config: ModelEvaluationConfig):
        self.config = config

    
    def eval_metrics(self, actual, pred):
        acc = accuracy_score(actual, pred)
        prec = precision_score(actual, pred)
        rec = recall_score(actual, pred)
        cm = confusion_matrix(actual, pred)
        cm_nor = confusion_matrix(actual, pred, normalize='true')
        cr = classification_report(actual, pred)
        return acc, prec, rec, cm, cm_nor, cr
    


    def log_into_mlflow(self):

        test_data = pd.read_csv(self.config.test_data_path)
        model = joblib.load(self.config.model_path)

        X_test = test_data.drop([self.config.target_column], axis=1)
        y_test = test_data[[self.config.target_column]]


        mlflow.set_registry_uri(self.config.mlflow_uri)
        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme


        with mlflow.start_run():

            predicted_qualities = model.predict(X_test)

            (acc, prec, rec, cm, cm_nor, cr) = self.eval_metrics(y_test, predicted_qualities)
            
            # Saving metrics as local
            #scores = {"Accuracy": acc, "Precision": prec, "Recall": rec, "Confusion Mat": cm, "C_report": cr}
            scores = {"Accuracy": acc, "Precision": prec, "Recall": rec, "Confusion Mat": np.array(cm).tolist()}
            save_json(path=Path(self.config.metric_file_name), data=scores)

            mlflow.log_params(self.config.all_params)

            mlflow.log_metric("Accuracy", acc)
            mlflow.log_metric("Precision", prec)
            mlflow.log_metric("Recall", rec)
            #mlflow.log_metric("Classification report", cr)

            mlflow.log_dict(np.array(cm).tolist(), "confusion_matrix.json")

            disp = ConfusionMatrixDisplay(confusion_matrix=cm)
            disp.plot(cmap=plt.cm.Blues, xticks_rotation=45)
            plt.savefig("ConfusionMatrix.png")
            mlflow.log_artifact("ConfusionMatrix.png")
            plt.close()

            disp = ConfusionMatrixDisplay(confusion_matrix=cm_nor)
            disp.plot(cmap=plt.cm.Blues, xticks_rotation=45)
            plt.savefig("NormalizedConfusionMatrix.png")
            mlflow.log_artifact("NormalizedConfusionMatrix.png")
            plt.close()


            # Model registry does not work with file store
            if tracking_url_type_store != "file":

                # Register the model
                # There are other ways to use the Model Registry, which depends on the use case,
                # please refer to the doc for more information:
                # https://mlflow.org/docs/latest/model-registry.html#api-workflow
                
                mlflow.sklearn.log_model(model, "model", registered_model_name="RandomForestClassifier")
            else:
                mlflow.sklearn.log_model(model, "model")

    