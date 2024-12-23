import torch
import torch.nn as nn

class Model:
    """
    A class for defining a model that takes 512 input features and outputs predictions for 17 classes.

    Attributes:
    ----------
    input_dim : int
        Dimensionality of the input features (default: 512).
    num_classes : int
        Number of output classes (default: 17).
    layers : nn.Module
        The neural network architecture.
    loss_function : nn.Module
        The loss function used for training (default: CrossEntropyLoss).
    device : torch.device
        The device to which the model is assigned (CPU or GPU).

    Methods:
    -------
    __init__(input_dim=512, num_classes=17, loss_function=None):
        Initializes the model with specified input dimensions, number of classes, and loss function.

    define_model():
        Defines the architecture of the model.

    forward(x):
        Performs a forward pass through the model.

    save_weights(file_path):
        Saves the model weights to a specified file path.

    load_weights(file_path):
        Loads model weights from a specified file path.

    get_loss_function():
        Returns the current loss function.

    set_loss_function(loss_function):
        Updates the loss function for the model.

    predict(inputs):
        Generates predictions for the given inputs.

    to_device(device):
        Moves the model to the specified device (CPU or GPU).

    test_model():
        Test the model by setting random inputs and targets. Checks that saving and loading weights works.
    """
    def __init__(self, input_dim=512, num_classes=17, loss_function=None):
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.layers = self.define_model()
        self.loss_function = loss_function or nn.BCEWithLogitsLoss()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to_device(self.device)

    def define_model(self):
        return nn.Sequential(
            nn.Linear(self.input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, self.num_classes),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.layers(x)
    
    def save_weights(self, file_path):
        torch.save(self.layers.state_dict(), file_path)
    
    def load_weights(self, file_path):
        self.layers.load_state_dict(torch.load(file_path, weights_only=True))
        self.layers.to(self.device)
    
    def get_loss_function(self):
        return self.loss_function
    
    def set_loss_function(self, loss_function):
        self.loss_function = loss_function
    
    def predict(self, inputs):
        self.layers.eval()
        with torch.no_grad():
            outputs = self.forward(inputs.to(self.device))
        return outputs
    
    def to_device(self, device):
        self.device = device
        self.layers.to(self.device)

def test_model(filename):
    model = Model(input_dim=512, num_classes=17)
    batch_size = 8
    dummy_inputs = torch.randn(batch_size, model.input_dim).to(model.device)
    dummy_targets = torch.randint(0, model.num_classes, (batch_size,)).to(model.device)

    outputs = model.forward(dummy_inputs)
    assert outputs.shape == (batch_size, model.num_classes), f"Expected output shape ({batch_size}, {model.num_classes}), got {outputs.shape}"
    
    loss = model.loss_function(outputs, dummy_targets)
    assert loss.item() >= 0, "Loss should be non-negative"
    
    predictions = model.predict(dummy_inputs)
    assert predictions.shape == (batch_size,), f"Expected predictions shape ({batch_size},), got {predictions.shape}"
    assert predictions.max() < model.num_classes and predictions.min() >= 0, "Predictions should be class indices between 0 and num_classes-1"
    
    file_path = f"weights/{filename}.pth"
    model.save_weights(file_path)
    model.load_weights(file_path)
    
    return "Model test passed successfully"
