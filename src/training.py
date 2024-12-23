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
    def __init__(self, model, train_loader, val_loader=None, optimizer=None, scheduler=None, batch_size=32, epochs=10, prediction_threshold=0.5, device='cpu'):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer or torch.optim.Adam(self.model.layers.parameters(), lr=0.001)
        self.scheduler = scheduler
        self.batch_size = batch_size
        self.epochs = epochs
        self.prediction_threshold = prediction_threshold
        self.device = torch.device(device)
        self.model.to_device(self.device)
        self.checkpoint_path = None
        self.metrics = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    
    def train(self):
        for epoch in range(self.epochs):
            self.model.layers.train()
            total_loss = 0
            correct_predictions = 0
            total_samples = 0

            for inputs, targets in self.train_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model.forward(inputs)
                loss = self.model.loss_function(outputs, targets)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item() * inputs.size(0)
                predictions = torch.sigmoid(outputs)
                predictions = (predictions > self.prediction_threshold).float()
                correct_predictions += (predictions == targets).sum().item()
                total_samples += inputs.size(0)

            epoch_loss = total_loss / total_samples
            epoch_acc = correct_predictions / total_samples
            self.metrics['train_loss'].append(epoch_loss)
            self.metrics['train_acc'].append(epoch_acc)

            if self.val_loader:
                val_loss, val_acc = self.evaluate(self.val_loader)
                self.metrics['val_loss'].append(val_loss)
                self.metrics['val_acc'].append(val_acc)

            if self.scheduler:
                self.scheduler.step()
            self.log_training_progress(epoch, epoch_loss, epoch_acc)
        self.plot_training_loss()
    
    def evaluate(self, data_loader):
        self.model.layers.eval()
        total_loss = 0
        correct_predictions = 0
        total_samples = 0
        with torch.no_grad():
            for inputs, targets in data_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.model.forward(inputs)
                loss = self.model.loss_function(outputs, targets)
                total_loss += loss.item() * inputs.size(0)
                predictions = torch.sigmoid(outputs)
                predictions = (predictions > self.prediction_threshold).float()
                correct_predictions += (predictions == targets).sum().item()
                total_samples += inputs.size(0)
        return total_loss / total_samples, correct_predictions / total_samples

    def save_checkpoint(self, file_path):
        checkpoint = {
            'model_state': self.model.model_architecture.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'scheduler_state': self.scheduler.state_dict() if self.scheduler else None,
            'metrics': self.metrics
        }
        torch.save(checkpoint, file_path)
    
    def load_checkpoint(self, file_path):
        checkpoint = torch.load(file_path)
        self.model.model_architecture.load_state_dict(checkpoint['model_state'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state'])
        if self.scheduler and checkpoint['scheduler_state']:
            self.scheduler.load_state_dict(checkpoint['scheduler_state'])
        self.metrics = checkpoint['metrics']
    
    def set_optimizer(self, optimizer):
        self.optimizer = optimizer
    
    def set_scheduler(self, scheduler):
        self.scheduler = scheduler
    
    def log_training_progress(self, epoch, loss, accuracy):
        print(f"Epoch {epoch+1}/{self.epochs}: Loss = {loss:.4f}, Accuracy = {accuracy:.4f}")
    
    def plot_training_loss(self):
        epochs = range(1, self.epochs + 1)
        plt.figure(figsize=(10, 6))
        plt.plot(epochs, self.metrics['train_loss'], label='Training Loss')
        if 'val_loss' in self.metrics:
            plt.plot(epochs, self.metrics['val_loss'], label='Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        plt.show()

    def to_device(self, device):
        self.device = torch.device(device)
        self.model.to_device(self.device)
    
    def get_training_metrics(self):
        return self.metrics