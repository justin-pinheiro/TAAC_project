class Trainer:
    """
    A class for training a model on a dataset using a DataLoader and saving the trained weights.

    Attributes:
    ----------
    model : Model
        The model instance to be trained.
    train_loader : DataLoader
        DataLoader for the training dataset.
    val_loader : DataLoader
        DataLoader for the validation dataset
    optimizer : torch.optim.Optimizer
        The optimizer for training.
    scheduler : torch.optim.lr_scheduler, optional
        Learning rate scheduler (optional).
    device : torch.device
        The device to which the training is assigned (CPU or GPU).
    batch_size : int
        Number of samples per batch.
    epochs : int
        Number of training epochs.
    checkpoint_path : str
        Path to save model checkpoints.

    Methods:
    -------
    __init__(model, train_loader, val_loader=None, optimizer=None, scheduler=None, batch_size=32, epochs=10, device='cpu'):
        Initializes the Trainer with the model, data loaders, and training parameters.

    train():
        Executes the training process over the specified number of epochs.

    evaluate(data_loader):
        Evaluates the model's performance on a given dataset.

    save_checkpoint(file_path):
        Saves the current state of the model and optimizer to a file.

    load_checkpoint(file_path):
        Loads the model and optimizer states from a checkpoint file.

    set_optimizer(optimizer):
        Sets or updates the optimizer for the training process.

    set_scheduler(scheduler):
        Sets or updates the learning rate scheduler.

    log_training_progress(epoch, loss, accuracy):
        Logs the training progress for the current epoch.

    to_device(device):
        Moves the trainer and its components (model, data) to the specified device.

    get_training_metrics():
        Returns the training metrics (e.g., loss, accuracy) for analysis.

    adjust_hyperparameters(batch_size=None, learning_rate=None, epochs=None):
        Adjusts the hyperparameters of the training process dynamically.
    """