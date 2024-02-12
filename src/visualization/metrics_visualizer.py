import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

class MetricsVisualizer:
    def __init__(self, metrics_objects):
        """
        Initializes the visualizer with one or more ClassificationMetrics objects.
        
        :param metrics_objects: A single ClassificationMetrics object or a list of them.
        """
        self.metrics_objects = metrics_objects if isinstance(metrics_objects, list) else [metrics_objects]

    def plot_roc_curve(self):
        plt.figure(figsize=(10, 8))
        
        for metrics in self.metrics_objects:
            fpr, tpr = metrics._roc_curve()
            roc_auc = metrics._auc()
            label = f'{metrics.exp_id} AUC = {roc_auc}' if metrics.exp_id else f'AUC = {roc_auc}'
            plt.plot(fpr, tpr, label=label)
        
        plt.plot([0, 1], [0, 1], 'k--', label='Chance')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic')
        plt.legend(loc="lower right")
        plt.show()

    def plot_precision_recall_curve(self):
        plt.figure(figsize=(10, 8))
        
        for metrics in self.metrics_objects:
            precision, recall = metrics._precision_recall_curve()
            ap = metrics._average_precision()
            label = f'{metrics.exp_id} AP = {ap}' if metrics.exp_id else f'AP = {ap}'
            plt.plot(recall, precision, label=label)
        
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend(loc="best")
        plt.show()
        
        
    def plot_accuracy_comparison(self):
        accuracies = [metrics._accuracy() for metrics in self.metrics_objects]
        labels = [metrics.exp_id if metrics.exp_id else "Unknown" for metrics in self.metrics_objects]
        
        plt.figure(figsize=(10, 8))
        plt.bar(range(len(accuracies)), accuracies, tick_label=labels)
        
        plt.xlabel('Experiment ID')
        plt.ylabel('Accuracy')
        plt.title('Accuracy Comparison')
        plt.ylim(0, 1.05)  # Extend y-axis to make room for labels if needed
        plt.xticks(rotation=45)  # Rotate labels if they overlap
        
        # Adding the text labels on the bars
        for i, acc in enumerate(accuracies):
            plt.text(i, acc + 0.01, f'{acc:.2f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.show()
        
    def plot_confusion_matrix(self):
        for metrics in self.metrics_objects:
            cm = metrics.confusion_matrix
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
            plt.xlabel('Predicted labels')
            plt.ylabel('True labels')
            plt.title(f'Confusion Matrix for {metrics.exp_id if metrics.exp_id else "Unknown"}')
            plt.show()
            
    def plot_f1_scores_comparison(self):
        f1_scores = [metrics._f1() for metrics in self.metrics_objects]
        labels = [metrics.exp_id if metrics.exp_id else "Unknown" for metrics in self.metrics_objects]

        plt.figure(figsize=(10, 8))
        plt.bar(range(len(f1_scores)), f1_scores, tick_label=labels)

        plt.xlabel('Experiment ID')
        plt.ylabel('F1 Score')
        plt.title('F1 Score Comparison')
        plt.ylim(0, 1.05)
        plt.xticks(rotation=45)

        for i, score in enumerate(f1_scores):
            plt.text(i, score + 0.01, f'{score:.2f}', ha='center', va='bottom')

        plt.tight_layout()
        plt.show()
        
        
    def plot_feature_importance(self, feature_names):
        for metrics in self.metrics_objects:
            if hasattr(metrics.model, 'feature_importances_'):
                importances = metrics.model.feature_importances_
                indices = np.argsort(importances)

                plt.figure(figsize=(10, 8))
                plt.title(f'Feature Importances for {metrics.exp_id if metrics.exp_id else "Unknown"}')
                plt.barh(range(len(indices)), importances[indices], color='b', align='center')
                plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
                plt.xlabel('Relative Importance')
                plt.show()
            else:
                print(f"{metrics.exp_id if metrics.exp_id else 'Unknown'} model does not support feature importance.")
        
    
