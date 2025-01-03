import os
import shutil
import subprocess
import tempfile
from event_spotting import EventSpotter
from model import Model
import streamlit as st
import cv2
import numpy as np
from training import Trainer
from utils import get_labels

class App:
    """
    A class for running a Streamlit web application that allows users to upload a video,
    process it to detect events using a trained model, and display highlights with confidence scores.

    Attributes:
    ----------
    model : Model
        The trained model used for video highlight prediction.
    feature_extractor : FeatureExtraction
        The feature extractor used to extract features from the input video.
    fps : int
        Frames per second used for processing the video.
    output_dir : str
        Directory where video highlights will be stored.
    display_video : bool
        Flag to determine if video highlights should be displayed in the web app.
    verbose : bool
        Flag to enable or disable detailed output in the console.

    Methods:
    -------
    __init__(model, feature_extractor, fps=2, display_video=True, verbose=False):
        Initializes the app with the model, feature extractor, and other parameters.

    upload_video():
        Allows the user to upload a video file and displays it within the Streamlit interface.
        Returns the file path of the uploaded video.

    extract_features(video_file):
        Loads pre-extracted features for a test video. Returns the feature array or an error if not found.

    extract_video_segment(video_file, start_time, end_time, idx):
        Extracts a segment of the video based on the start and end times and saves it as a new video file.
        Returns the path to the extracted video.

    display_highlights(video_file, highlights, selected_classes, confidence_threshold, extract_length):
        Displays highlights based on detected events and user-selected classes, extracting relevant video clips.

    render_video_highlight(video_clip, idx, start_time, end_time):
        Displays a specific video highlight clip within the Streamlit interface.

    clear_highlights_folder():
        Clears all files in the highlights folder to prepare for new extractions.

    process_video(video_file, confidence_threshold):
        Processes the video by extracting features, detecting events, and returning the detected highlights.

    select_classes_to_show():
        Displays a set of checkboxes for users to select which event classes they want to display as highlights.
        Returns a list of indices for the selected classes.

    run_app(confidence_threshold, extract_length):
        Runs the application, allowing the user to configure parameters, select event classes, 
        and generate and display video highlights.
    """

    def __init__(self, model, feature_extractor, fps=2, display_video=True, verbose=False):
        self.model = model
        self.feature_extractor = feature_extractor
        self.fps = fps
        self.display_video = display_video
        self.verbose = verbose
        self.output_dir = "highlights"

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def upload_video(self):
        self.clear_highlights_folder()
        uploaded_file = st.file_uploader("Upload a video", type=["mp4", "mkv"])
        if uploaded_file is not None:
            temp_file_path = os.path.join(tempfile.gettempdir(), uploaded_file.name)
            with open(temp_file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            st.video(temp_file_path) # we show the full video
            return temp_file_path
        return None

    def extract_features(self, video_file):
        # as we could not extract the features ourselves, we use the already extracted features from the dataset.
        # here, we use the features from the first half of the game in test_folder for testing purposes.
        feature_path = "test_folder/1_ResNET_TF2_PCA512.npy"
        
        if os.path.exists(feature_path):
            st.write(f"Loading features from {feature_path}...")
            return np.load(feature_path)
        else:
            st.error(f"Feature path {feature_path} not found.")
            return None

    def extract_video_segment(self, video_file, start_time, end_time, idx):
        output_filename = os.path.join(self.output_dir, f"highlight_{start_time}_{end_time}_{idx}.mp4")
        cmd = ["ffmpeg", "-i", video_file, "-ss", str(start_time), "-to", str(end_time), "-c:v", "libx264", "-c:a", "aac", output_filename]
        subprocess.check_output(cmd)
        return output_filename

    def display_highlights(self, video_file, highlights, selected_classes, confidence_threshold, extract_length):
        for timestamp, predictions in highlights.items():
            for idx, pred in enumerate(predictions):
                if pred >= confidence_threshold and idx in selected_classes:
                    start_time = max(timestamp - extract_length // 2, 0)
                    end_time = start_time + extract_length
                    video_clip = self.extract_video_segment(video_file, start_time, end_time, idx)
                    self.render_video_highlight(video_clip, idx, start_time, end_time)


    def render_video_highlight(self, video_clip, idx, start_time, end_time):
        st.write(f"Displaying highlight '{get_labels()[idx]}' from {start_time}s to {end_time}s")
        st.video(video_clip)

    def clear_highlights_folder(self):
        for filename in os.listdir(self.output_dir):
            file_path = os.path.join(self.output_dir, filename)
            if os.path.isfile(file_path):
                os.remove(file_path)

    def process_video(self, video_file, confidence_threshold):
        st.write("Processing video...")
        features = self.extract_features(video_file)
        if features is None:
            st.error("Unable to load features. Please ensure the feature file exists.")
            return None
        
        spotter = EventSpotter(self.model, self.fps, confidence_threshold)
        spotter.detect_events(features)
        print(spotter.get_events())
        spotter.show_predictions_summary(None, True)
        return spotter.get_events()
    
    def select_classes_to_show(self):
        labels = get_labels()
        selected_classes = []

        for idx, label in enumerate(labels):
            if st.checkbox(label, key=label):
                selected_classes.append(idx)

        return selected_classes
    
    def run_app(self, confidence_threshold, extract_length):
        
        selected_classes = self.select_classes_to_show()

        if st.button("Generate Highlights"):
            highlights = self.process_video(video_file, confidence_threshold)
            if highlights:
                self.display_highlights(video_file, highlights, selected_classes, confidence_threshold, extract_length)


if __name__ == "__main__":
    classifier = Model()
    trainer = Trainer(classifier, None, None)
    trainer.load_checkpoint("weights/model_0_2.pth")

    app = App(
        classifier,
        None,
        fps=2,
        display_video=True,
        verbose=True
    )

    st.title("Configure parameters")
    detection_treshold = st.slider("Detection treshold", 0.0, 1.0, 0.5)
    extract_duration = st.number_input("Extract duration (in seconds)", 5, 120, 60)
    
    st.title("Video Highlight Detector")
    video_file = app.upload_video()

    if video_file:
        app.run_app(detection_treshold, extract_duration)