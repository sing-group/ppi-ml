import numpy as np

from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score, accuracy_score, recall_score, precision_score, matthews_corrcoef

# balanced_accuracy With adjusted=True = Youden
def youden(y_true, y_pred):
     return balanced_accuracy_score(y_true, y_pred, adjusted=True)

def specificity(y_true, y_pred):
    tn, fp, _, _ = confusion_matrix(y_true, y_pred).ravel()
    return tn / (tn + fp)

DEFAULT_SCORING_DICT = {
    'recall': recall_score,
    'precision': precision_score,
    'f1': f1_score,
    'accuracy': accuracy_score, 
    'specificity': specificity,
    'matthews_corrcoef': matthews_corrcoef,
    'youden': youden
}

def weighted_avg_std(values, weights):
    """
    Compute the weighted average and standard deviation of a given set of values and weights.
    
    Parameters:
    values (array-like): The data values.
    weights (array-like): The weights corresponding to the data values.
    
    Returns:
        tuple(float, float): The weighted average and standard deviation.
    """
    average = np.average(values, weights=weights)
    variance = np.average((values-average)**2, weights=weights)
    return average, np.sqrt(variance)
