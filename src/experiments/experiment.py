import os
import json
import importlib

import numpy as np

from src.config import EXPERIMENT_DIR
from src.data.load_dataset import load_spambase
from src.evaluation.classification_metrics import ClassificationMetrics

from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split



class Experiment:
    def __init__(self, name, model_class, model_params, metrics, description=None, n_splits=5, random_state=None):
        self.name = name
        self.description = description or ''
        self.model_class = model_class
        self.model_params = model_params
        self.metrics = metrics
        self.n_splits = n_splits
        self.random_state = random_state    
        
        self.experiment_dir = os.path.join(EXPERIMENT_DIR, self.name)
        self.config_path = os.path.join(self.experiment_dir, 'config.json')
        if not os.path.exists(self.config_path):
            self.setup()
        self.results_path = os.path.join(self.experiment_dir, 'results.json')

    @staticmethod
    def from_config(config_path):
        if not os.path.exists(config_path):
            raise FileNotFoundError(f'Config file not found: {config_path}')
        
        with open(config_path, 'r') as f:
            config = json.load(f)
        return Experiment(
            name=config['name'],
            description=config['description'],
            model_class=config['model_class'],
            model_params=config['model_params'],
            metrics=config['metrics']
        )


    def setup(self):
        os.makedirs(self.experiment_dir, exist_ok=True)
        self.save_config()

    def save_config(self):
        config = {
            'name': self.name,
            'description': self.description,
            'model_class': self.model_class,
            'model_params': self.model_params,
            'metrics': self.metrics
        }
        with open(self.config_path, 'w') as f:
            json.dump(config, f, indent=4)
        print(f'Experiment Configuration saved to {self.config_path}')
        
    def save_results(self):
        """Save the results of the experiment to a json file"""
        results = {
            'name': self.name,
            'description': self.description,
            'results': self.results
        }
        with open(self.results_path, 'w') as f:
            json.dump(results, f, indent=4)
        print(f'Experiment Results saved to {self.results_path}')
        
    def run(self):
        try:
            self.setup()
            print(f'Running experiment {self.name}...')
            # Run the experiment
            self.results = self._run_experiment()
            self.save_results()
            print(f'Experiment {self.name} completed successfully!')
        except Exception as e:
            print(f'Error running experiment {self.name}: {e}')
            raise e
        
    def load_model(self):
        module_name, class_name = self.model_class.rsplit('.', 1) 
        module = importlib.import_module(module_name)
        model_class = getattr(module, class_name)  
        self.model = model_class(**self.model_params)
                
    def _run_experiment(self):
        # Run the model on the data and return the results
        self.load_model()
        X, y = load_spambase()
        
        kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)
        fold_results = []
        
        for i, (train_index, test_index) in enumerate(kf.split(X)):
            print(f'Running fold {i+1}...')
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            
            self.model.fit(X_train, y_train)
            metrics_obj: ClassificationMetrics = self.model.evaluate(X_test, y_test)
            fold_results.append(metrics_obj.evaluate(return_metrics=True))
        
        aggregate_results = self._aggregate_results(fold_results)
        
        print(f'Experiment {self.name} results: {aggregate_results}')
        return aggregate_results
    
    def _aggregate_results(self, fold_results):

        accumulated = {
            'accuracy': [],
            'precision': [],
            'recall': [],
            'f1': [],
            'auc': [],
            'average_precision': [],
            'confusion_matrix': np.zeros((2, 2), dtype=int)
        }
        
        # Process each fold's results
        for result in fold_results:
            accumulated['accuracy'].append(result['accuracy'])
            accumulated['precision'].append(result['precision'])
            accumulated['recall'].append(result['recall'])
            accumulated['f1'].append(result['f1'])
            accumulated['auc'].append(result['auc'])
            accumulated['average_precision'].append(result['average_precision'])
            # Sum the confusion matrix
            accumulated['confusion_matrix'] += result['confusion_matrix']
        
        # Calculate mean for numeric metrics
        aggregated_results = {metric: np.mean(values) for metric, values in accumulated.items() if metric != 'confusion_matrix'}
        # Confusion matrix is already accumulated, so just assign it
        aggregated_results['confusion_matrix'] = accumulated['confusion_matrix'].tolist()

        return aggregated_results