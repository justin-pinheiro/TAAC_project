from typing import Counter
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch
import numpy as np
import os
import pandas as pd
from SoccerNet.utils import getListGames
from utils import get_labels
import matplotlib.pyplot as plt

class DataLoading:
    """
    A class for loading features and labels for football videos, creating a PyTorch Dataset, 
    and preparing DataLoaders for model training and evaluation.

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

    class FootballDataset(Dataset):
        def __init__(self, features, labels):
            self.features = features
            self.labels = labels

        def __len__(self):
            return len(self.features)

        def __getitem__(self, idx):
            return self.features[idx], self.labels[idx]

    def get_video_names(self, split_type):
        if (split_type != "train" and split_type != "valid" and split_type != "test"):
            raise ValueError("The parameter split_type should be 'train', 'valid', or 'test'")
        return getListGames(split=split_type)

    def __init__(self, data_path, fps, chunk_length, batch_size, split_type, shuffle=True, context_aware=False):
        self.data_path = data_path
        self.fps = fps
        self.chunk_length = chunk_length
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.split_type = split_type
        self.video_names = self.get_video_names(split_type)
        self.dataset:self.FootballDataset = None
        self.data_loader:DataLoader = None
        self.context_aware=context_aware
    
    def get_video_path_from_video_name(self, video_name):
        return os.path.join(self.data_path, video_name) 

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

        labels_file = os.path.join(video_path, 'labels.csv')
        if not os.path.exists(labels_file):
            raise FileNotFoundError(f"Labels file 'labels.csv' not found for video {video_name}.")
        
        labels_df = pd.read_csv(labels_file)
        labels_df = labels_df[labels_df['half'] == half]

        frames_per_chunk = self.chunk_length * self.fps

        num_chunks = features.shape[0] // frames_per_chunk
        features = features[:num_chunks * frames_per_chunk]
        features = features.reshape(num_chunks, frames_per_chunk, -1)  # Shape: (num_chunks, frames_per_chunk, 512)
        features = self.preprocess_features(features)

        labels = []
        for chunk_idx in range(num_chunks):
            start_time = chunk_idx * self.chunk_length
            end_time = (chunk_idx + 1) * self.chunk_length

            if (self.context_aware):
                chunk_labels = self.preprocess_labels_with_context(start_time, end_time, labels_df)
            else:
                chunk_labels = labels_df[(labels_df['time'] >= start_time) & (labels_df['time'] < end_time)]['label'].values
                chunk_labels = self.preprocess_labels(chunk_labels)
            labels.append(chunk_labels)

        labels = np.array(labels)  # Shape: (num_chunks, num_categories)

        return features, labels
    
    def preprocess_features(self, features):
        # Here we apply average pooling
        features = features.mean(1)
        return features
    
    def preprocess_labels_with_context(self, start_time, end_time, labels_df):
        categories = get_labels() 
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

    def preprocess_labels(self, labels):
        categories = get_labels()
        encoding = [1 if category in labels else 0 for category in categories]
        return encoding
    
    def create_dataset(self):
        all_features = []
        all_labels = []

        for video_name in self.video_names:
            for half in range(1,3):
                features, labels = self.load_features_labels(video_name, half)
                
                for chunk_idx in range(features.shape[0]):
                    all_features.append(torch.tensor(features[chunk_idx], dtype=torch.float32))
                    all_labels.append(torch.tensor(labels[chunk_idx], dtype=torch.float32))

        dataset = self.FootballDataset(features=all_features,labels=all_labels)

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

        labels = get_labels()
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

def test_data_loader(data_path, fps, chunk_length, batch_size, split_type):

    # Instantiate the DataLoading object
    data_loader_instance = DataLoading(data_path, fps, chunk_length, batch_size, split_type)
    data_loader_instance.create_dataset()

    data_loader_instance.create_dataloader()
    data_loader: DataLoader = data_loader_instance.get_dataloader()

    # General DataLoader Information
    print("=" * 50)
    print("DataLoader Summary")
    print("=" * 50)
    print(f"Data path: {data_path}")
    print(f"FPS: {fps}")
    print(f"Batch size: {batch_size}")
    print(f"Split type: {split_type}")
    print(f"Shuffle: {data_loader_instance.shuffle}")
    print(f"Total samples in dataset: {len(data_loader_instance.dataset)}")
    print(f"Total batches: {len(data_loader)}")
    print("=" * 50)

    # Iterate through the DataLoader
    for i, (features, labels) in enumerate(data_loader):
        print(f"Batch {i+1} Details:")
        print("-" * 50)

        # Features information
        print("Features:")
        print(f"Shape: {features.shape}")
        print(f"Data type: {features.dtype}")
        print(f"Device: {features.device}")
        print(f"Sample data: {features[0][:5]}")  # Print first 5 values of the first sample
        print(f"Min: {features.min().item()}, Max: {features.max().item()}, Mean: {features.mean().item()}, Std: {features.std().item()}")
        
        # Labels information
        print("Labels:")
        print(f"Shape: {labels.shape}")
        print(f"Data type: {labels.dtype}")
        print(f"Device: {labels.device}")
        print(f"Sample data: {labels[0]}")  # Print the first label
        print("-" * 50)

        # Check if the batch size is correct
        assert features.size(0) == batch_size, f"Expected batch size {batch_size}, but got {features.size(0)}"
        print(f"Batch {i+1} has {features.size(0)} samples (correct batch size).")
        print("=" * 50)

        # Stop after the second batch for demonstration purposes
        if i == 1:
            break