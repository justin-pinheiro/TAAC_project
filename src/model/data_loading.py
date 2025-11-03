from typing import Counter
from torch import float32, tensor
from torch.utils.data import DataLoader
import numpy as np
import os
import pandas as pd
from SoccerNet.utils import getListGames
import matplotlib.pyplot as plt

from domain.football_dataset import FootballDataset
from model.labels_manager import LabelsManager

class DataLoading:
    """
    Loads labels and features from football videos.
    Relies on a PyTorch Dataset.
    Prepares DataLoaders for model training and evaluation.

    Attributes:
    ----------
    data_path : str
        Path to the root directory containing league, season, and video subdirectories.
    fps : int
        The frames per second (fps) to use when processing video features.
    chunk_length : int
        Video chunks length in seconds.
    batch_size : int
        Batch size for the DataLoader.
    shuffle : bool
        Whether to shuffle the dataset during loading (default: True).
    split_type : str
        The split type for the dataset. Can be one of 'train', 'valid', or 'test'.
    video_names : list
        List of video names for the current split, retrieved using the `getListGames` utility.
    dataset : FootballDataset
        A PyTorch Dataset object containing features and labels for all videos.
    data_loader : DataLoader
        A PyTorch DataLoader object for loading the dataset in batches.

    Methods
    -------
    __init__(data_path, fps, batch_size, split_type, shuffle=True):
        Initializes the DataLoading class with paths, batch size, shuffle flag, and split type.
    
    get_video_names(split_type):
        Retrieves the list of video names for the specified split type ('train', 'valid', or 'test').

    get_video_path_from_video_name(video_name):
        Given a video name, locates its path in the directory structure.
    
    load_features_labels(video_name, half):
        Loads the features and labels for a given video and half (1 or 2), processes them, 
        and returns them in a format suitable for model training.
    
    preprocess_features(features):
        Placeholder method for preprocessing features (e.g., normalization).
    
    preprocess_labels(labels):
        Placeholder method for preprocessing labels (e.g., one-hot encoding).
    
    create_dataset():
        Creates a PyTorch Dataset by loading features and labels for all videos in the specified split type.
    
    create_dataloader():
        Creates a PyTorch DataLoader for the dataset, batching it according to the specified batch size 
        and shuffle flag.
    
    get_dataloader():
        Returns the DataLoader for the current dataset. If the DataLoader does not exist, it creates one 
        using the `create_dataloader` method.
    """

    def __init__(self, label_manager:LabelsManager, data_path, fps, chunk_length, batch_size, split_type, shuffle=True, context_aware=False):
        self.label_manager = label_manager
        self.data_path = data_path
        self.fps = fps
        self.chunk_length = chunk_length
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.split_type = split_type
        self.video_names = self.get_video_names(split_type)
        self.dataset: FootballDataset = None
        self.data_loader: DataLoader = None
        self.context_aware = context_aware
    
    def get_video_path_from_video_name(self, video_name):
        return os.path.join(self.data_path, video_name) 

    def get_video_names(self, split_type):
        if (split_type != "train" and split_type != "valid" and split_type != "test"):
            raise ValueError("The parameter split_type should be 'train', 'valid', or 'test'")
        return getListGames(split=split_type)

    def load_features_labels(self, video_name, half):
        if (half != 1 and half != 2):
            raise ValueError("'half' should be either '1' or '2'.")
        if (video_name not in self.video_names):
            raise ValueError(f'video {video_name} not in video list.')
        
        video_path = self.get_video_path_from_video_name(video_name)
        if video_path is None:
            raise FileNotFoundError(f"Video {video_name} not found.")
        
        feature_file = f"{half}_ResNET_TF2_PCA512.npy"
        feature_path = os.path.join(video_path, feature_file)

        if not os.path.exists(feature_path):
            raise FileNotFoundError(f"Feature file {feature_file} not found for video {video_name} (half {half}).")
        
        features = np.load(feature_path)
        print(f"Features shape = {features.shape}")

        labels_file = os.path.join(video_path, 'labels.csv')
        if not os.path.exists(labels_file):
            raise FileNotFoundError(f"Labels file 'labels.csv' not found for video {video_name}.")
        
        labels_df = pd.read_csv(labels_file)
        labels_df = labels_df[labels_df['half'] == half]

        frames_per_chunk = self.chunk_length * self.fps
        print(f"Frames per chunk = {frames_per_chunk}")

        num_chunks = features.shape[0] // frames_per_chunk
        print(f"Chunks count = {num_chunks}")

        features = features[:num_chunks * frames_per_chunk]
        features = features.reshape(num_chunks, frames_per_chunk, -1)  # Shape: (num_chunks, frames_per_chunk, 512)
        features = self.preprocess_features(features)

        print(f"Features shape : {features.shape}")
        print(f"Features : {features}")

        labels = []
        for chunk_idx in range(num_chunks):
            start_time = chunk_idx * self.chunk_length
            end_time = (chunk_idx + 1) * self.chunk_length

            if (self.context_aware):
                chunk_labels = self.preprocess_labels_with_context(start_time, end_time, labels_df)
            else:
                chunk_labels = labels_df[(labels_df['time'] >= start_time) & (labels_df['time'] < end_time)]['label'].values
                chunk_labels = self.encode_labels(chunk_labels)
            labels.append(chunk_labels)

        labels = np.array(labels)  # Shape: (num_chunks, num_categories)
        print(f"Labels shape : {labels.shape}")
        print(f"Labels : {labels}")

        return features, labels
    
    def preprocess_features(self, features):
        # Here we apply average pooling
        features = features.mean(1)
        return features
    
    def preprocess_labels_with_context(self, start_time, end_time, labels_df):
        categories = self.label_manager.get_labels() 
        distances = []

        for category in categories:
            category_times = labels_df[labels_df['label'] == category]['time'].values
            if len(category_times) == 0:
                distances.append(5400)
            else:
                signed_distances = [
                    time - start_time if time >= start_time and time < end_time else 
                    (time - start_time if time < start_time else time - end_time)
                    for time in category_times
                ]
                min_distance = min(signed_distances, key=abs)
                distances.append(min_distance)


        # print(f"Start time: '{start_time}', end time: '{end_time}', labels = '{distances}'")

        return distances

    def encode_labels(self, labels):

        encoding = [1 if category in labels else 0 for category in self.label_manager.get_labels()]
        return encoding
    
    def create_dataset(self):
        all_features = []
        all_labels = []

        for video_name in self.video_names:
            for half in (1,2):
                print(f"-----------------\nLoading features for game {video_name}, half={half}\n-----------------")
                features, labels = self.load_features_labels(video_name, half)
                
                num_chunks = features.shape[0]

                if num_chunks == 0:
                    raise Exception(f"No chunks loaded for game {video_name}")

                print(f"Features shape = {features.shape}")
                print(f"Labels shape = {labels.shape}")

                for chunk_idx in range(features.shape[0]):
                    all_features.append(tensor(features[chunk_idx], dtype=float32))
                    all_labels.append(tensor(labels[chunk_idx], dtype=float32))

        dataset = FootballDataset(features=all_features,labels=all_labels)

        self.dataset = dataset
    
    def create_dataloader(self):
        if self.dataset is None:
            self.create_dataset()
        self.data_loader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=self.shuffle, drop_last=True)
    
    def get_dataloader(self):
        if self.data_loader is None:
            self.create_dataloader()
            
        return self.data_loader

    def show_label_distribution(self):
        if self.dataset is None:
            print("Dataset is empty.")
            return

        labels = self.label_manager.get_labels()
        label_counts = [0] * len(labels)
        no_label_count = 0

        for _, label_tensor in self.dataset:
            label_array = label_tensor.numpy()
            if label_array.sum() == 0:  # Check if all labels are 0
                no_label_count += 1
            else:
                label_counts = [label_counts[i] + label_array[i] for i in range(len(labels))]

        labels.append("No Label")
        label_counts.append(no_label_count)

        plt.figure(figsize=(12, 6))
        plt.bar(labels, label_counts, color='skyblue')
        plt.xlabel('Labels', fontsize=12)
        plt.ylabel('Count', fontsize=12)
        plt.title('Label Distribution', fontsize=14)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.show()
