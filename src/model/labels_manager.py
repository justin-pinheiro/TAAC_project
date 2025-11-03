import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
import json

class LabelsManager():
    __instance = None
    __SOCCERNET_DATA_PATH = "SoccerNet"
    __LABELS_JSON_FILE_NAME = "Labels-v2.json"
    __LABELS = (
        'Ball out of play', 
        'Clearance', 
        'Corner', 
        'Direct free-kick', 
        'Foul', 
        'Goal', 
        'Indirect free-kick', 
        'Kick-off', 
        'Offside', 
        'Penalty', 
        'Red card', 
        'Shots off target', 
        'Shots on target', 
        'Substitution', 
        'Throw-in', 
        'Yellow card', 
        'Yellow->red card'
    )
    __LABELS_COLORS = (
    '#558c89', 
    '#74af95', 
    '#9193f0', 
    '#874e4c', 
    '#444444', 
    '#e74c3c', 
    '#a26c48', 
    '#2ecc71', 
    '#cccccc', 
    '#666666', 
    '#ff0000', 
    '#5070d6', 
    '#1d59b3', 
    '#3498db', 
    '#7f8c8d', 
    '#f1c40f', 
    '#f39c12'
)

    def __new__(cls, *args, **kwargs):
        if cls.__instance is None:
            cls.__instance = super(LabelsManager, cls).__new__(cls)
        return cls.__instance

    def get_instance(self):
        return self.__instance

    def get_labels(self):
        return self.__LABELS

    def get_label_colors(self):
        return self.__LABELS_COLORS

    def get_annotations_dataframe(self):
        """
        Traverse the directory structure and extract annotations into a pandas DataFrame.
        Encodes data for machine learning processing.

        Returns:
            - pd.DataFrame: DataFrame containing all annotations with metadata and engineered features.
        """
        rows = []

        leagues = os.listdir(self.__SOCCERNET_DATA_PATH)
        for league in leagues:

            league_path = os.path.join(self.__SOCCERNET_DATA_PATH, league)
            if not os.path.isdir(league_path):
                print(f"{league_path} is not a valid directory.")
                continue

            seasons = os.listdir(league_path)
            for season in seasons:

                season_path = os.path.join(league_path, season)
                if not os.path.isdir(season_path):
                    print(f"{season_path} is not a valid directory.")
                    continue

                for game_id, game_folder in enumerate(os.listdir(season_path)):

                    game_path = os.path.join(season_path, game_folder)
                    labels_file = os.path.join(game_path, self.__LABELS_JSON_FILE_NAME)
                    
                    if not os.path.isfile(labels_file):
                        continue
                    
                    with open(labels_file, "r") as f:
                        data = json.load(f)

                    away_team = data.get("gameAwayTeam", "")
                    home_team = data.get("gameHomeTeam", "")
                    game_date = data.get("gameDate", "")
                    game_score = data.get("gameScore", "")
                    
                    date_part, hour_part = game_date.split(' - ') if game_date else ('', '')
                    
                    if game_score:
                        home_goals, away_goals = map(int, game_score.split(' - '))
                    else:
                        raise ValueError(f"Field 'game_score' is missing for file {labels_file}")

                    annotations = data.get("annotations", [])

                    for annotation_id, annotation in enumerate(annotations):
                        position = annotation.get("position", "")
                        visibility = annotation.get("visibility", "not shown")

                        game_time = annotation.get("gameTime", "")
                        if game_time:
                            half, time = game_time.split(' - ')
                            first_half = 1 if int(half) == 1 else 0
                            second_half = 1 if int(half) == 2 else 0
                        else:
                            raise ValueError(f"Field 'game_time' is missing for file {labels_file}")

                        label = annotation.get("label", "")
                        if not label:
                            raise ValueError(f"Field 'label' is missing for file {labels_file}")
                        label_one_hot_encoded = {label: 1}

                        if not league:
                            raise ValueError(f"Field 'league' is missing for file {labels_file}")
                        league_one_hot_encoded = {league: 1}
                        
                        team = annotation.get("team")
                        if not team:
                            raise ValueError(f"Field 'team' is missing for file {labels_file}")
                        if team == "away":
                            team_one_hot_encoded = {"is_away": 1, "is_home": 0}
                        elif team == "home":
                            team_one_hot_encoded = {"is_away": 0, "is_home": 1}

                        visibility_encoded = {"visibility": 1 if visibility == "visible" else 0}

                        row = {
                            "game_id": game_id,
                            "annotation_id": annotation_id,
                            "away_team": away_team,
                            "home_team": home_team,
                            "date": date_part,
                            "hour": hour_part,
                            "home_goals": home_goals,
                            "away_goals": away_goals,
                            "time": time,
                            "first_half": first_half,
                            "second_half": second_half,
                            **league_one_hot_encoded,
                            **label_one_hot_encoded,
                            **team_one_hot_encoded,
                            **visibility_encoded,
                        }
                        rows.append(row)

        df = pd.DataFrame(rows)

        return df.copy()

    def __parse_labels_json(self, file_path):
        rows = []
        with open(file_path, "r") as f:
            data = json.load(f)

            away_team = data.get("gameAwayTeam", "")
            home_team = data.get("gameHomeTeam", "")
            game_date = data.get("gameDate", "")
            game_score = data.get("gameScore", "")
            url_local = data.get("UrlLocal", "")
            url_youtube = data.get("UrlYoutube", "")
            
            annotations = data.get("annotations", [])
            for annotation in annotations:
                row = {
                    "away_team": away_team,
                    "home_team": home_team,
                    "date": game_date,
                    "score": game_score,
                    "url_local": url_local,
                    "url_youtube": url_youtube,
                    "game_time": annotation.get("gameTime", ""),
                    "label": annotation.get("label", ""),
                    "position": annotation.get("position", ""),
                    "team": annotation.get("team", ""),
                    "visibility": annotation.get("visibility", ""),
                }
                rows.append(row)

        df = pd.DataFrame(rows)
        return df

    def save_labels_to_csv(self):

        for league in os.listdir(self.__SOCCERNET_DATA_PATH):
            league_path = os.path.join(self.__SOCCERNET_DATA_PATH, league)
            if not os.path.isdir(league_path):
                continue 

            for season in os.listdir(league_path):
                season_path = os.path.join(league_path, season)
                if not os.path.isdir(season_path):
                    continue

                for game_folder in os.listdir(season_path):
                    game_path = os.path.join(season_path, game_folder)
                    labels_file = os.path.join(game_path, self.__LABELS_JSON_FILE_NAME)
                    
                    if not os.path.isfile(labels_file):
                        continue
                    
                    df = self.__parse_labels_json(labels_file)
                    
                    labels = []
                    halves = []
                    times = []
                    
                    for _, row in df.iterrows():
                        game_time = row['game_time']
                        
                        half, time_str = game_time.split(' - ')
                        
                        labels.append(row['label'])
                        halves.append(int(half))
                        times.append(int(time_str.split(":")[0]) * 60 + int(time_str.split(":")[1]))
                    
                    output_dataframe = pd.DataFrame({'label': labels, 'half': halves, 'time': times})
                    csv_path = os.path.join(game_path, "labels.csv")
                    output_dataframe.to_csv(csv_path, index=False)

                    print(f"Labels saved to {csv_path}.")
