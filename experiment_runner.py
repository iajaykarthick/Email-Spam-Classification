import json

from src.experiments import run_experiment

def main():
    # read run_config.json file and get the experiment_id to run
    with open('run_config.json') as f:
        run_config = json.load(f)

    experiment_id = run_config['experiment_id']
    run_experiment(experiment_id=experiment_id)
    
if __name__ == '__main__':
    main()