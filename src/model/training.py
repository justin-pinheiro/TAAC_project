from evaluation import Evaluator
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt


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
    prediction_threshold : float
        Threshold to know when a prediction is considered as accepted.

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

    plot_training_loss():
        Show the training and validation loss in a graph.

    to_device(device):
        Moves the trainer and its components (model, data) to the specified device.

    get_training_metrics():
        Returns the training metrics (e.g., loss, accuracy) for analysis.
    """
    def __init__(self, model, train_loader, val_loader=None, optimizer=None, scheduler=None, batch_size=32, epochs=10, prediction_threshold=0.5, device='cpu', context_aware=False):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer or torch.optim.Adam(self.model.layers.parameters(), lr=0.0015)
        self.scheduler = scheduler
        self.batch_size = batch_size
        self.epochs = epochs
        self.prediction_threshold = prediction_threshold
        self.device = torch.device(device)
        self.model.to_device(self.device)
        self.checkpoint_path = None
        self.context_aware=context_aware
        self.metrics = {'train_mAP': [], 'val_mAP': [], 'train_loss': [], 'val_loss': []}
    
    def train(self):
        for epoch in range(self.epochs):
            self.model.layers.train()
            total_loss = 0

            for inputs, targets in self.train_loader:

                inputs, targets = inputs.to(self.device), targets.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model.forward(inputs)
                loss = self.model.loss_function(outputs, targets)

                loss.backward()
                self.optimizer.step()
                total_loss += loss.item() * inputs.size(0)

            train_mAP = self.evaluate(self.train_loader)
            self.metrics['train_mAP'].append(train_mAP)
            self.metrics['train_loss'].append(total_loss / len(self.train_loader.dataset))
            
            if self.val_loader:
                val_mAP = self.evaluate(self.val_loader)
                self.metrics['val_mAP'].append(val_mAP)
            
            if self.scheduler:
                self.scheduler.step()
            
            self.log_training_progress(epoch, total_loss / len(self.train_loader.dataset), train_mAP, val_mAP)
    
    def evaluate(self, data_loader):
        evaluator = Evaluator(self.model, data_loader, self.device, self.batch_size, "mAP", False)
        metrics = evaluator.evaluate(context_aware=self.context_aware)
        return metrics["mAP"]

    def save_checkpoint(self, file_path):
        checkpoint = {
            'model_state': self.model.layers.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'scheduler_state': self.scheduler.state_dict() if self.scheduler else None,
            'metrics': self.metrics
        }
        torch.save(checkpoint, file_path)
    
    def load_checkpoint(self, file_path):
        checkpoint = torch.load(file_path)
        self.model.layers.load_state_dict(checkpoint['model_state'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state'])
        if self.scheduler and checkpoint['scheduler_state']:
            self.scheduler.load_state_dict(checkpoint['scheduler_state'])
        self.metrics = checkpoint['metrics']
    
    def set_optimizer(self, optimizer):
        self.optimizer = optimizer
    
    def set_scheduler(self, scheduler):
        self.scheduler = scheduler
    
    def log_training_progress(self, epoch, loss, train_mAP, val_map):
        print(f"Epoch {epoch+1}/{self.epochs}: Loss = {loss:.4f}, training mAP = {train_mAP:.4f}, validation mAP = {val_map:.4f}")
    
    def plot_training_loss(self):
        epochs = range(1, self.epochs + 1)
        plt.figure(figsize=(10, 6))
        plt.plot(epochs, self.metrics['train_loss'], label='Training Loss')
        plt.plot(epochs, self.metrics['train_mAP'], label='Training mAP')
        plt.plot(epochs, self.metrics['val_mAP'], label='Validation mAP')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Metrics')
        plt.legend()
        plt.show()

    def to_device(self, device):
        self.device = torch.device(device)
        self.model.to_device(self.device)
    
    def get_training_metrics(self):
        return self.metrics