import torch 
import numpy as np
from sklearn.metrics import precision_recall_curve, average_precision_score


class Evaluator:
    """
    A class for evaluating a trained model on a given dataset using a DataLoader and calculating performance metrics.

    Attributes:
    ----------
    model : Model
        The trained model instance to be evaluated.
    data_loader : DataLoader
        DataLoader for the dataset on which the model will be evaluated.
    device : torch.device
        The device to which the evaluation will be assigned (CPU or GPU).
    metrics : dict
        Dictionary to store evaluation metrics (e.g., mAP).
    batch_size : int
        Number of samples per batch.
    score_type : str
        Type of evaluation score to compute (e.g., "mAP").
    verbose : bool
        Flag to enable/disable detailed output during evaluation.

    Methods:
    -------
    __init__(model, data_loader, device='cpu', batch_size=32, score_type='mAP', verbose=False):
        Initializes the Evaluator with the model, data loader, and evaluation parameters.

    evaluate():
        Evaluates the model on the dataset and computes the desired evaluation metric(s).

    compute_mAP():
        Computes the mean Average Precision (mAP) for multi-class or multi-label classification tasks.

    log_evaluation_results(metrics):
        Logs or prints the evaluation results, including the mAP or other relevant metrics.

    save_evaluation_report(file_path):
        Saves the evaluation results and metrics to a file for later analysis.

    to_device(device):
        Moves the evaluator and its components (model, data) to the specified device.

    get_evaluation_metrics():
        Returns the computed evaluation metrics for further analysis or reporting.

    set_score_type(score_type):
        Sets the type of evaluation score to compute (e.g., "mAP", "accuracy", etc.).

    set_batch_size(batch_size):
        Sets the batch size for evaluation, which may be useful when performing batch-wise operations.

    enable_verbose_output(verbose):
        Enables or disables verbose output during evaluation for debugging or detailed tracking.

    get_predictions():
        Retrieves the model's predictions for the evaluation dataset, which can be used for further analysis.
    """
    def __init__(self, model, data_loader, device='cpu', batch_size=32, score_type='mAP', verbose=False):
        self.model = model
        self.model.eval()
        self.model.to(device)

        self.data_loader = data_loader
        self.device = torch.device(device) if isinstance(device, str) else device
        self.metrics = {}
        self.batch_size = batch_size
        self.score_type = score_type
        self.verbose = verbose

        if self.verbose:
            print(f"Evaluator initialized with the following parameters:")
            print(f"Device: {self.device}, Batch Size: {self.batch_size}, Score Type: {self.score_type}")

    def evaluate(self):
        if self.verbose:
            print("Starting evaluation.")
        all_preds, all_labels = [], []

        with torch.no_grad():
            for batch in self.data_loader:
                inputs, labels = batch
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                all_preds.append(outputs)
                all_labels.append(labels)

        all_preds = torch.cat(all_preds)
        all_labels = torch.cat(all_labels)

        if self.score_type == 'mAP':
            self.metrics['mAP'] = self.compute_mAP(all_preds, all_labels)
        else:
            raise NotImplementedError(f"Score type '{self.score_type}' is not implemented.")

<<<<<<< HEAD
=======
        if (self.verbose):
            self.log_evaluation_results(self.metrics)
>>>>>>> 3aab354570d1ee219462459f503fd74ea3847a81
        return self.metrics
    
    def compute_mAP(self, preds, labels):
        if self.verbose:
            print("Computing mAP...")

        preds = np.array(preds)
        labels = np.array(labels)

        num_classes = labels.shape[1]
        APs = []

        for class_idx in range(num_classes):
            class_preds = preds[:, class_idx]
            class_labels = labels[:, class_idx]
            if self.verbose:
                print(f"Class {class_idx + 1} preds: {class_preds}")
                print(f"Class {class_idx + 1} labels: {class_labels}")

            if np.sum(class_preds) == 0 and np.sum(class_labels) > 0:
                APs.append(0)
                if self.verbose:
                    print(f"Class {class_idx + 1}: No positive predictions. AP = 0.")
                continue

            AP = average_precision_score(class_labels, class_preds)
            APs.append(AP)

            if self.verbose:
                print(f"Class {class_idx + 1}: AP = {AP:.4f}")

        mAP = np.mean(APs)

        if self.verbose:
            print(f"Computed mAP: {mAP:.4f}")

        return mAP
    
    def log_evaluation_results(self, metrics):
        print("Evaluation Results:")
        for metric, value in metrics.items():
            print(f"{metric}: {value:.2f}")

    def save_evaluation_report(self, file_path):
        with open(file_path, 'w') as f:
            f.write("Evaluation Report\n")
            for metric, value in self.metrics.items():
                f.write(f"{metric}: {value:.2f}\n")
        if self.verbose:
            print(f"Evaluation report saved to {file_path}")

    def to_device(self, device):
        self.device = torch.device(device) if isinstance(device, str) else device
        self.model.to(self.device)
        if self.verbose:
            print(f"Moved model to {self.device}")

    def get_evaluation_metrics(self):
        if not self.metrics:
            raise ValueError("No metrics computed yet. Call evaluate() first.")
        return self.metrics
    
    def set_score_type(self, score_type):
        self.score_type = score_type
        if self.verbose:
            print(f"Score type set to {self.score_type}.")

    def set_batch_size(self, batch_size):
        self.batch_size = batch_size
        self.data_loader.batch_size = batch_size
        if self.verbose:
            print(f"Batch size adjusted to {self.batch_size}")

    def enable_verbose_output(self, verbose):
        self.verbose = verbose
        print("Verbose output enabled" if self.verbose else "Verbose output disabled")

    def get_predictions(self):
        if self.verbose:
            print("Retrieving predictions.")
        all_preds = []

        with torch.no_grad():
            for batch in self.data_loader:
                inputs, _ = batch
                inputs = inputs.to(self.device)
                outputs = self.model(inputs)
                all_preds.append(outputs)

        all_preds = torch.cat(all_preds)
        return all_preds