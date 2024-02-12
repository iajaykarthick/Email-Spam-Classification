import numpy as np

class ClassificationMetrics:
    def __init__(self, y_true, y_pred, round_digits=None, exp_id=None):
        self.y_true = y_true
        self.y_pred = y_pred
        self.exp_id = exp_id
        self.round_digits = round_digits
        
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
        tpr = self._recall()
        fp = np.sum((self.y_true == 0) & (self.y_pred == 1))
        tn = np.sum((self.y_true == 0) & (self.y_pred == 0))
        fpr = self._round(fp / (fp + tn) if fp + tn > 0 else 0)
        return fpr, self._round(tpr)

    def _auc(self):
        fpr, tpr = self._roc_curve()
        return self._round(np.trapz([tpr], [fpr]))

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
    
    def evaluate(self, return_metrics=False):
        self._calculate_metrics()
        if return_metrics:
            return {
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
