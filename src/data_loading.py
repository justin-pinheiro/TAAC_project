from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch
import numpy as np
import os
import pandas as pd
from SoccerNet.utils import getListGames

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

    def get_video_names(split_type):
        if (split_type != "train" and split_type != "valid" and split_type != "test"):
            raise ValueError("The parameter split_type should be 'train', 'valid', or 'test'")
        return getListGames(split=split_type)

    def __init__(self, data_path, fps, batch_size, split_type, shuffle=True):
        self.data_path = data_path
        self.fps = fps
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.split_type = split_type
        self.video_names = self.get_video_names(split_type)
        self.dataset:self.FootballDataset = None
        self.data_loader:DataLoader = None
    
    def get_video_path_from_video_name(self, video_name):
        video_path = None
        for league_name in os.listdir(self.data_path):
            league_path = os.path.join(self.data_path, league_name)
            if os.path.isdir(league_path):
                # Iterate through each season folder
                for season_name in os.listdir(league_path):
                    season_path = os.path.join(league_path, season_name)
                    if os.path.isdir(season_path):
                        # Check if the video exists in the current season
                        potential_video_path = os.path.join(season_path, video_name)
                        if os.path.isdir(potential_video_path):
                            video_path = potential_video_path
                            break
                if video_path:
                    break
        return video_path

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
        features = self.preprocess_features(features)

        labels_file = os.path.join(video_path, 'labels.csv')
        if not os.path.exists(labels_file):
            raise FileNotFoundError(f"Labels file 'labels.csv' not found for video {video_name}.")
        
        labels_df = pd.read_csv(labels_file)
        labels_df = labels_df[labels_df['half'] == half]

        labels = labels_df[['label', 'time']].values
        labels = self.preprocess_labels(labels)

        return features, labels
    
    def preprocess_features(self, features):
        return features
    
    def preprocess_labels(self, labels):
        return labels
    
    def create_dataset(self):
        all_features = []
        all_labels = []

        for video_name in self.video_names:
            for half in range(1,3):
                features, labels = self.load_features_labels(video_name, half)
                
                all_features.append(torch.tensor(features, dtype=torch.float32))
                all_labels.append(torch.tensor(labels, dtype=torch.float32))

        dataset = self.FootballDataset(features=torch.tensor(all_features, dtype=torch.float32),
                                    labels=torch.tensor(all_labels, dtype=torch.float32))

        self.dataset = dataset
    
    def create_dataloader(self):
        if self.dataset is None:
            self.create_dataset()
        self.data_loader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=self.shuffle)
    
    def get_dataloader(self):
        if self.data_loader is None:
            self.create_dataloader()
            
        return self.data_loader