import os

from .experiment import Experiment
from src.visualization.metrics_visualizer import MetricsVisualizer


class ExperimentRun:
    def __init__(self, experiment_dir, run_id=None):
        self.run_id = run_id
        self.experiment_dir = experiment_dir
        self.experiment = Experiment.from_config(os.path.join(self.experiment_dir, 'config.json'))

    def __str__(self):
        return f"ExperimentRun(run_id={self.run_id}, run_dir={self.run_dir})"

    def __repr__(self):
        return str(self)
    
    def execute(self):
        self.experiment.run()
        
    def save_plots(self):
        metrics_visualizer = MetricsVisualizer(self.experiment.metrics_obj)
        metrics_visualizer.plot_roc_curve(save_fig=True, fig_path=os.path.join(self.experiment_dir, 'roc_curve.png'))