import numpy as np

from sklearn.metrics import roc_curve, auc


class ClassificationMetrics:
    def __init__(self, y_true=None, y_pred=None, y_pred_proba=None, round_digits=None, exp_id=None, **kwargs):
        if y_true is None and y_pred is None and exp_id is None:
            raise ValueError('At least one of y_true, y_pred or exp_id must be provided')
        self.y_true = y_true
        self.y_pred = y_pred
        self.y_pred_proba = y_pred_proba
        self.exp_id = exp_id
        self.round_digits = round_digits
        
        for k, v in kwargs.items():
            setattr(self, k, v)
        
    @staticmethod
    def from_results(exp_id, results, round_digits=None):
        class_metrics = ClassificationMetrics(exp_id=exp_id, round_digits=round_digits, **results)
        return class_metrics
        
    
    def _round(self, value):
        if self.round_digits is not None:
            return round(value, self.round_digits)
        return value

    def _accuracy(self):
        return self._round(np.mean(self.y_true == self.y_pred))

    def _precision(self):
        tp = np.sum((self.y_true == 1) & (self.y_pred == 1))
        fp = np.sum((self.y_true == 0) & (self.y_pred == 1))
        return self._round(0 if tp + fp == 0 else tp / (tp + fp))

    def _recall(self):
        tp = np.sum((self.y_true == 1) & (self.y_pred == 1))
        fn = np.sum((self.y_true == 1) & (self.y_pred == 0))
        return self._round(0 if tp + fn == 0 else tp / (tp + fn))

    def _f1(self):
        p = self._precision()
        r = self._recall()
        return self._round(0 if p + r == 0 else 2 * p * r / (p + r))

    def _confusion_matrix(self):
        tp = np.sum((self.y_true == 1) & (self.y_pred == 1))
        tn = np.sum((self.y_true == 0) & (self.y_pred == 0))
        fp = np.sum((self.y_true == 0) & (self.y_pred == 1))
        fn = np.sum((self.y_true == 1) & (self.y_pred == 0))
        return np.array([[tn, fp], [fn, tp]])  # Not rounded, as it contains counts

    def _classification_report(self):
        return {
            'precision': self._precision(),
            'recall': self._recall(),
            'f1-score': self._f1(),
            'support': len(self.y_true)  # Not rounded, as it is a count
        }

    def _roc_curve(self):
        fpr, tpr, _ = roc_curve(self.y_true, self.y_pred_proba)
        return fpr, tpr

    def _auc(self):
        fpr, tpr = self._roc_curve()
        return self._round(auc(fpr, tpr))

    def _precision_recall_curve(self):
        return self._round(self._recall()), self._round(self._precision())

    def _average_precision(self):
        r, p = self._precision_recall_curve()
        return self._round(np.trapz([p], [r]))

    def _calculate_metrics(self):
        self.accuracy = self._accuracy()
        self.precision = self._precision()
        self.recall = self._recall()
        self.f1 = self._f1()
        self.confusion_matrix = self._confusion_matrix()
        self.classification_report = self._classification_report()
        self.roc_curve = self._roc_curve()
        self.auc = self._auc()
        self.precision_recall_curve = self._precision_recall_curve()
        self.average_precision = self._average_precision()
    
    def evaluate(self, return_metrics=False, select_metrics=None):
        self._calculate_metrics()
        if return_metrics:
            results = {
                'accuracy': self.accuracy,
                'precision': self.precision,
                'recall': self.recall,
                'f1': self.f1,
                'confusion_matrix': self.confusion_matrix,
                'classification_report': self.classification_report,
                'roc_curve': self.roc_curve,
                'auc': self.auc,
                'precision_recall_curve': self.precision_recall_curve,
                'average_precision': self.average_precision
            }
            if select_metrics is not None:
                return {k: v for k, v in results.items() if k in select_metrics}
            
            return results
