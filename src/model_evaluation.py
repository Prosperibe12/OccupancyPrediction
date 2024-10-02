import logging 
from abc import ABC, abstractmethod
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report 
import mlflow
import matplotlib.pyplot as plt
import seaborn as sns 


class EvaluationStrategy(ABC):

    @abstractmethod
    def evaluate():
        raise NotImplementedError("This method must be implemented")


class Accuracy(EvaluationStrategy):

    def evaluate(self, model, x_test, y_test):
        """
        Evaluate model accuracy, log confusion matrix and classification report to the current run.
        Args:
            model: Trained model
            x_test: Test dataset features
            y_test: True labels for the test dataset

        Returns:
            accuracy: Accuracy score of the model
        """
        # Get the active MLflow run
        active_run = mlflow.last_active_run()

        # get the model predictions
        pred = model.predict(x_test)

        # get the class with highest probability
        if pred.shape[1] > 1:
            y_pred = pred.argmax(axis=1)
        
        # calculate accuracy
        accuracy = accuracy_score(y_test, y_pred)
        accuracy = round(accuracy * 100, 2)

        # Log accuracy as a metric in the current active run
        mlflow.log_metric(key="Accuracy Score", value=accuracy, run_id=active_run.info.run_id)

        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        cm_display = sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title("Confusion Matrix")
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")

        # Save the confusion matrix as a PNG file
        plt.savefig('confusion_matrix.png')
        plt.close()

        # Log confusion matrix as an artifact in the active run
        mlflow.log_artifact('confusion_matrix.png',run_id=active_run.info.run_id)

        # Classification Report
        report = classification_report(y_test, y_pred, output_dict=True)
        report_str = classification_report(y_test, y_pred)

        # Log classification report as a text artifact
        with open("classification_report.txt", "w") as f:
            f.write(report_str)

        mlflow.log_artifact("classification_report.txt",run_id=active_run.info.run_id)

        return accuracy

class ConfusionMatrix(EvaluationStrategy):
    """ 
    Confusion Matrix Strategy for model evaluation
    """
    
    def evaluate(self, model, x_test, y_test):
        """ 
        Evaluate model with Confusion Matrix
        Args:
            model: Trained model
            x_test: Test dataset features
            y_test: True labels for the test dataset

        Returns:
            cm: Confusion matrix of the model predictions
        """
        # Get the model predictions
        y_pred = model.predict(x_test)

        # If it's a classification model, convert probabilities to class predictions
        if y_pred.shape[1] > 1:
            y_pred = y_pred.argmax(axis=1)

        # Calculate the confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        # mlflow.log_artifact(cm)

        return cm 
    
class ClassificationReport(EvaluationStrategy):
    """ 
    Classification Report Strategy for model evaluation
    """
    def evaluate(self, model, x_test, y_test):
        """ 
        Evaluate model with Classification Report
        Args:
            model: Trained model
            x_test: Test dataset features
            y_test: True labels for the test dataset

        Returns:
            report: Classification report of the model
        """
        # Get the model predictions
        y_pred = model.predict(x_test)

        # If it's a classification model, convert probabilities to class predictions
        if y_pred.shape[1] > 1:
            y_pred = y_pred.argmax(axis=1)

        # Generate the classification report
        report = classification_report(y_test, y_pred)
        return report 
    

class ModelEvaluation:

    def __init__(self, strategy: EvaluationStrategy, model, x_test, y_test):
        self.strategy = strategy 
        self.model = model
        self.x_test = x_test 
        self.y_test = y_test 

    def evaluate(self):
        """ 
        execute the evaluate method for any given Strategy
        """
        return self.strategy.evaluate(self.model, self.x_test, self.y_test)