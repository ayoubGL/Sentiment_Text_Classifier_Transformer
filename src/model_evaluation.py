from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import numpy as np


def calculate_metics(predictions, labels):
    """
        Calculate accurcay, precision, recall and F1-score for a classification task.
        
        Args:
            predictions: predicted class labels
            labels: True class labels
            
        returns:    
            dict: A dict containing the metrics
    """
    
    accuracy = accuracy_score(labels, predictions)
    
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average="weighted", zero_division=0)
    
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1
    }
    
    return metrics
    