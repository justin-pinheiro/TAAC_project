from model.evaluation import Evaluator
import torch
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from model.labels_manager import LabelsManager

class EventSpotter:
    """
    A class for detecting football events from extracted video features using a trained classifier model.
    
    Attributes:
    ----------
    model : nn.Module
        The trained classifier model capable of predicting 17 event classes from 512 features.
    fps : int
        The frames per second (fps) used to process the video features.
    detection_threshold : float
        Threshold for deciding if an event is detected.
    nms_window : int
        The time window in seconds for applying non-maximum suppression (NMS) to filter redundant predictions.
    predictions : dict
        Dictionary storing predictions for each second of the input video features.
        Format : {seconds:predictions_for_each_class} (ex:{0:[0,1,0,...,0]} )
    events : dict
        Dictionary storing predictions above detection threshold and filtered with NMS.
        Format : {seconds:predictions_for_each_class} (ex:{0:[0,1,0,...,0]} )

    Methods:
    -------
    __init__(model, fps, detection_threshold=0.5, nms_window=60):
        Initializes the EventSpotter with the trained model, fps, detection threshold, and NMS window.
    
    detect_events(features):
        Slides through the extracted video features second by second (taking fps into account) and uses the model to detect events. Save the raw predictions into predictions
        and the filtered detected events into events.
        
    apply_nms(predictions):
        Applies non-maximum suppression (NMS) to the raw second-by-second predictions using the specified window.
    
    evaluate_predictions(ground_truth):
        Evaluates the final event detection predictions against ground truth events using mean Average Precision (mAP) from the Evaluator class.
    
    get_predictions():
        Returns the raw second-by-second model's predictions as a dictionary mapping seconds to event probability.
    
    get_events():
        Returns the final predictions as a dictionary mapping seconds to detected events.
    
    reset_predictions():
        Clears the current predictions, allowing the instance to process a new video.
    
    save_predictions(file_path):
        Saves the predictions to a specified file for analysis or record-keeping.
    
    load_predictions(file_path):
        Loads predictions from a file into the current instance.
    
    show_predictions_summary(ground_truth=None, show_threshold=False, final_predictions=False):
        Prints a graph of predictions along time (in minutes). For each second, it should print the event probability from predictions. 
        If ground_truth, it should print where the ground truth events are. 
        If show_threshold, it should print a red horizontal line for event detection threshold.
        If final_predictions, it should use events instead of predictions to only show final predictions.
    """

    def __init__(self, labels_manager: LabelsManager, model, fps, detection_threshold=0.5, nms_window=60, delta=60):
        self.labels_manager = labels_manager
        self.model = model
        self.fps = fps
        self.detection_threshold = detection_threshold
        self.nms_window = nms_window
        self.predictions = {}
        self.events = {}
        self.delta = delta

    def detect_events(self, features):
        self.predictions = {}
        self.events = {}

        num_frames = features.shape[0]
        num_seconds = num_frames // self.fps

        self.model.eval()

        with torch.no_grad():
            for second in range(num_seconds):
                start_idx = second * self.fps
                end_idx = start_idx + self.fps

                # Extract features for the second and calculate the mean over the frames.
                feature_chunk = features[start_idx:end_idx]
                if feature_chunk.shape[0] == self.fps:
                    aggregated_feature = np.mean(feature_chunk, axis=0)
                    aggregated_feature_tensor = torch.tensor(aggregated_feature, dtype=torch.float32).unsqueeze(0) # Shape: [1, 512]

                    score = self.model(aggregated_feature_tensor).squeeze(0)
                    self.predictions[second] = score.cpu().detach().numpy().tolist()

        print(self.predictions[0])

        self.events = self.apply_nms(self.predictions)

    def apply_nms(self, predictions):
        nms_results = defaultdict(list)
        num_classes = len(next(iter(predictions.values())))
        print("num classes : ", num_classes)

        for class_idx in range(num_classes):
            scores = []
            for second, pred in predictions.items():
                if pred[class_idx] >= self.detection_threshold:
                    scores.append((second, pred[class_idx]))
            
            scores.sort(key=lambda x: x[1], reverse=True)

            while scores:
                best = scores.pop(0)  # Get the best score
                nms_results[best[0]].append(class_idx)  # Add the class index to results

                # Filter scores based on the NMS window
                scores = [score for score in scores if abs(score[0] - best[0]) > self.nms_window]

        # Convert indices to one-hot encoding
        one_hot_results = {}
        for second, indices in nms_results.items():
            one_hot_vector = [1 if i in indices else 0 for i in range(num_classes)]
            one_hot_results[second] = one_hot_vector

        return one_hot_results


    def evaluate_predictions(self, ground_truth):
        num_classes = max(
            max((len(events) for events in ground_truth.values()), default=0),
            max((len(events) for events in self.events.values()), default=0)
        )
        filtered_ground_truth = {second: events for second, events in ground_truth.items() if not np.all(events == 0)}
        # Align keys between ground_truth and self.events
        all_seconds = set(filtered_ground_truth.keys()).union(set(self.events.keys()))
        
        aligned_ground_truth = {
            second: filtered_ground_truth.get(second, [0] * num_classes) for second in all_seconds
        }
        aligned_events = {
            second: self.events.get(second, [0] * num_classes) for second in all_seconds
        }

        # Convert arrays to lists (ensure one-hot encoding as lists)
        aligned_ground_truth = {
            second: list(events) for second, events in aligned_ground_truth.items()
        }
        aligned_events = {
            second: list(events) for second, events in aligned_events.items()
        }
        
        print("Aligned Ground Truth:")
        print(aligned_ground_truth)
        print("Aligned Events:")
        print(aligned_events)
        
        evaluator = Evaluator(
            model=self.model, 
            data_loader=None, 
            device="cpu", 
            score_type="mAP"
        )
        
        mAP = evaluator.compute_mAP_event_spotting(aligned_events, aligned_ground_truth, self.delta)

        print(f"mAP: {mAP}")
        
        return {"mAP": mAP}

    def get_predictions(self):
        return self.predictions

    def get_events(self):
        return self.events

    def reset_predictions(self):
        self.predictions = {}
        self.events = {}

    def save_predictions(self, file_path):
        with open(file_path, "w") as f:
            f.write(str(self.predictions))

    def load_predictions(self, file_path):
        with open(file_path, "r") as f:
            self.predictions = eval(f.read())

    def show_predictions_summary(self, ground_truth, show_threshold=False):
        # Transpose raw_predictions for easier plotting
        classes = len(next(iter(self.predictions.values())))
        time_steps = list(self.predictions.keys())
        class_scores = {class_idx: [] for class_idx in range(classes)}
        
        labels_names = self.labels_manager.get_labels()

        for second, predictions in self.predictions.items():
            for class_idx, score in enumerate(predictions):
                class_scores[class_idx].append(score)

        plt.figure(figsize=(120, 6))
        class_colors = plt.cm.get_cmap('tab20', classes)

        for class_idx, scores in class_scores.items():
            color = class_colors(class_idx)
            plt.plot(time_steps, scores, label=labels_names[class_idx], color=color)

        if (ground_truth):
            for second, true_events in ground_truth.items():
                for class_idx, event in enumerate(true_events):
                    if event == 1:
                        color = class_colors(class_idx)
                        plt.scatter(second, 1.05, marker='o', color=color, zorder=5)


        if show_threshold and self.detection_threshold is not None:
            plt.axhline(y=self.detection_threshold, color="red", linestyle="--", label="Threshold")

        plt.title("Predictions Over Time")
        plt.xlabel("Time (seconds)")
        plt.ylabel("Prediction Scores")
        plt.legend(loc="upper right")
        plt.grid(True)
        plt.show()
