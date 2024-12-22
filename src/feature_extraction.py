class FeatureExtraction:
    """
    A class for extracting features from videos in a given database folder.

    Attributes:
    ----------
    base_path : str
        The base path of the database folder containing videos.
    fps : int
        The frames per second at which features will be extracted.
    model : callable
        A feature extraction model that takes a frame or a batch of frames 
        and outputs features of fixed dimensionality (e.g., 512).
    feature_dim : int
        The dimensionality of the extracted features (default: 512).

    Methods:
    -------
    __init__(base_path, fps, model, feature_dim=512):
        Initializes the class with the base path, fps, model, and feature dimensionality.

    get_video_paths():
        Returns a list of all video file paths in the database folder.

    load_video(video_path):
        Loads a video from the given path and returns its frames and metadata.

    preprocess_frame(frame):
        Preprocesses a single video frame to prepare it for feature extraction.

    extract_features_from_video(video_path):
        Extracts features from the frames of a video at the specified fps using the provided model.

    save_features(features, video_path):
        Saves the extracted features to a `.npy` file with a path based on the video name.

    process_database():
        Processes all videos in the database folder, extracts features, and saves them.
    """
    pass