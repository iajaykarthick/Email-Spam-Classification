# evaluation metrics for classification
import numpy as np


def accuracy(y_true, y_pred):
    return np.mean(y_true == y_pred)

def precision(y_true, y_pred):
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    
    if tp + fp == 0:
        return 0 
    
    return tp / (tp + fp)

def recall(y_true, y_pred):
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    
    if tp + fn == 0:
        return 0
    
    return tp / (tp + fn)

def f1(y_true, y_pred):
    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)
    
    if p + r == 0:
        return 0
    
    return 2 * p * r / (p + r)

def confusion_matrix(y_true, y_pred):
    tp = np.sum((y_true == 1) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    return np.array([[tn, fp], [fn, tp]])

def classification_report(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)
    f1_score = f1(y_true, y_pred)
    return {'precision': p, 'recall': r, 'f1-score': f1_score, 'support': np.sum(cm)}

def roc_curve(y_true, y_pred):
    # True positive rate
    tpr = recall(y_true, y_pred)
    
    # False positive rate
    fp = np.sum((y_true == 0) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fpr = fp / (fp + tn)
    
    return fpr, tpr

def auc(y_true, y_pred):
    fpr, tpr = roc_curve(y_true, y_pred)
    return np.trapz(tpr, fpr)

def precision_recall_curve(y_true, y_pred):
    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)
    return r, p

def average_precision(y_true, y_pred):
    r, p = precision_recall_curve(y_true, y_pred)
    return np.trapz(p, r)

def evaluate(y_true, y_pred):
    acc = accuracy(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)
    report = classification_report(y_true, y_pred)
    roc = roc_curve(y_true, y_pred)
    auc_score = auc(y_true, y_pred)
    pr = precision_recall_curve(y_true, y_pred)
    ap = average_precision(y_true, y_pred)
    return {'accuracy': acc, 'confusion_matrix': cm, 'classification_report': report, 'roc_curve': roc, 'auc': auc_score, 'precision_recall_curve': pr, 'average_precision': ap}

