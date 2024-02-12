from .experiment import Experiment


def create_experiment(name, config, model, optimizer, loss, metrics, callbacks, train_loader, val_loader, test_loader):
    return Experiment(name, config, model, optimizer, loss, metrics, callbacks, train_loader, val_loader, test_loader)