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

    adjust_batch_size(batch_size):
        Adjusts the batch size for evaluation, which may be useful when performing batch-wise operations.

    enable_verbose_output(verbose):
        Enables or disables verbose output during evaluation for debugging or detailed tracking.

    get_predictions():
        Retrieves the model's predictions for the evaluation dataset, which can be used for further analysis.
    """
