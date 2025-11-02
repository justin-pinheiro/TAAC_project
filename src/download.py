import os
import sys
import argparse
from dotenv import load_dotenv

try:
    from SoccerNet.Downloader import SoccerNetDownloader
except ImportError:
    print("Error: The SoccerNet package is not installed.")
    print("Please ensure SoccerNet is installed via 'pip install SoccerNet'.")
    sys.exit(1)

def main():
    parser = argparse.ArgumentParser(
        description="Download specific components of the SoccerNet dataset.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    
    parser.add_argument(
        "type", 
        choices=["annotations", "features", "all"],
        help="""Specify the type of data to download:
  'annotations' - Downloads Labels-v2.json (Split: train, valid, test, challenge)
  'features' - Downloads ResNET features (Split: train, valid, test, challenge)
  'videos' - Downloads games videos (Split: train, valid, test, challenge)
  'all' - Downloads annotations, features and videos.
  """
    )
    
    args = parser.parse_args()

    load_dotenv()
    local_directory = "SoccerNet"
    
    print(f"\n--- Initializing Downloader (Local Directory: {local_directory}) ---")
    
    try:
        mySoccerNetDownloader = SoccerNetDownloader(LocalDirectory=local_directory)
    except Exception as e:
        print(f"Error initializing SoccerNetDownloader: {e}")
        print("Ensure the library is configured correctly.")
        sys.exit(1)

    sn_key = os.getenv("SOCCERNET_KEY")
    if sn_key:
        mySoccerNetDownloader.password = sn_key
        print("SoccerNet key loaded successfully from .env file.")
    else:
        print("Warning: SOCCERNET_KEY not found in environment variables or .env file.")
        print("If required, please set it up for full dataset access.")

    download_type = args.type
    
    if download_type == "annotations" or download_type == "all":
        print("\n--- Starting Annotation Download ---")
        mySoccerNetDownloader.downloadGames(
            files=["Labels-v2.json"], 
            split=["train", "valid", "test"]
        )
        print("Annotation download finished.")

    if download_type == "features" or download_type == "all":
        print("\n--- Starting Feature Download (ResNET PCA512) ---")
        mySoccerNetDownloader.downloadGames(
            files=["1_ResNET_TF2_PCA512.npy", "2_ResNET_TF2_PCA512.npy"], 
            split=["train", "valid", "test", "challenge"]
        )
        print("Feature download finished.")
    
    if download_type == "videos" or download_type == "all":
        print("\n--- Starting Videos Download ---")
        mySoccerNetDownloader.downloadGames(
            files=["1_224p.mkv", "2_224p.mkv"], 
            split=["train", "valid", "test", "challenge"]
        )
        print("Videos download finished.")
    
    print(f"\n--- SoccerNet download process complete for type: {download_type} ---")


if __name__ == "__main__":
    main()
