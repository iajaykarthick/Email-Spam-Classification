import os

from src.config import EXPERIMENT_DIR
from .experiment import Experiment
from .experiment_run import ExperimentRun


def run_experiment(experiment_id: str):
    """Run an experiment and return the results"""
    exp_dir_path = os.path.join(EXPERIMENT_DIR, experiment_id)

    if not os.path.exists(exp_dir_path):
        raise FileNotFoundError(f'Experiment directory not found: {exp_dir_path}')

    exp_run = ExperimentRun(
        experiment_dir=exp_dir_path
    )
    exp_run.execute()
    exp_run.save_plots()
    print(f'Experiment {experiment_id} completed successfully!')