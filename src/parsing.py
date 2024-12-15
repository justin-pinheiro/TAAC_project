import os
import json
import pandas as pd

base_path = "SoccerNet"

def parse_labels_json():
    """
    Traverse the directory structure and extract annotations into a pandas DataFrame. 
    Obsolete function as parse_labels_json_with_feature_engineering() 
    was created to performs feature engineering and add features to the dataframe.
    
    Parameters:
    - base_path (str): Root directory containing the league folders.
    
    Returns:
    - pd.DataFrame: DataFrame containing all annotations with relevant metadata.
    """
    rows = []

    for league in os.listdir(base_path):
        league_path = os.path.join(base_path, league)
        if not os.path.isdir(league_path):
            continue 

        for season in os.listdir(league_path):
            season_path = os.path.join(league_path, season)
            if not os.path.isdir(season_path):
                continue

            for game_folder in os.listdir(season_path):
                game_path = os.path.join(season_path, game_folder)
                labels_file = os.path.join(game_path, "Labels-v2.json")
                
                if not os.path.isfile(labels_file):
                    continue
                
                with open(labels_file, "r") as f:
                    data = json.load(f)

                league_name = league
                season_name = season
                game_name = game_folder
                away_team = data.get("gameAwayTeam", "")
                home_team = data.get("gameHomeTeam", "")
                game_date = data.get("gameDate", "")
                game_score = data.get("gameScore", "")
                url_local = data.get("UrlLocal", "")
                url_youtube = data.get("UrlYoutube", "")
                
                annotations = data.get("annotations", [])
                for annotation in annotations:
                    row = {
                        "league": league_name,
                        "season": season_name,
                        "game_name": game_name,
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

def parse_labels_json_with_feature_engineering():
    """
    Traverse the directory structure and extract annotations into a pandas DataFrame,
    applying feature engineering and ensuring no missing values (NaNs).
    
    Parameters:
    - base_path (str): Root directory containing the league folders.
    
    Returns:
    - pd.DataFrame: DataFrame containing all annotations with relevant metadata and engineered features.
    """
    rows = []

    for league in os.listdir(base_path):
        league_path = os.path.join(base_path, league)
        if not os.path.isdir(league_path):
            continue

        for season in os.listdir(league_path):
            season_path = os.path.join(league_path, season)
            if not os.path.isdir(season_path):
                continue
            for game_id, game_folder in enumerate(os.listdir(season_path)):
                game_path = os.path.join(season_path, game_folder)
                labels_file = os.path.join(game_path, "Labels-v2.json")
                
                if not os.path.isfile(labels_file):
                    continue
                
                with open(labels_file, "r") as f:
                    data = json.load(f)

                away_team = data.get("gameAwayTeam", "")
                home_team = data.get("gameHomeTeam", "")
                game_date = data.get("gameDate", "")
                game_score = data.get("gameScore", "")
                url_local = data.get("UrlLocal", "")
                url_youtube = data.get("UrlYoutube", "")
                
                date_part, hour_part = game_date.split(' - ') if game_date else ('', '')
                
                # Split score into home and away goals, handle missing scores
                if game_score:
                    try:
                        home_goals, away_goals = map(int, game_score.split(' - '))
                    except ValueError:
                        home_goals, away_goals = 0, 0  # Default if score is malformed
                else:
                    home_goals, away_goals = 0, 0

                annotations = data.get("annotations", [])
                for annotation_id, annotation in enumerate(annotations):
                    # Extract label, position, team, and visibility
                    label = annotation.get("label", "")
                    position = annotation.get("position", "")
                    team = annotation.get("team", "")
                    visibility = annotation.get("visibility", "not visible")

                    # Split game time into half and time, handle missing game time
                    game_time = annotation.get("gameTime", "")
                    if game_time:
                        try:
                            half, time = game_time.split(' - ')
                            first_half = 1 if int(half) == 1 else 0
                            second_half = 1 if int(half) == 2 else 0
                        except ValueError:
                            first_half = second_half = 0
                            time = ""
                    else:
                        first_half = second_half = 0
                        time = ""

                    # One-hot encoding for leagues
                    league_encoded = {f"league_{league}": 1}

                    # One-hot encoding for labels
                    label_encoded = {f"label_{label}": 1} if label else {}

                    # One-hot encoding for team
                    if team == "away":
                        team_encoded = {"is_away": 1, "is_home": 0, "team_unknown": 0}
                    elif team == "home":
                        team_encoded = {"is_away": 0, "is_home": 1, "team_unknown": 0}
                    else:
                        team_encoded = {"is_away": 0, "is_home": 0, "team_unknown": 1}

                    # Convert visibility into binary (visible = 1, not shown = 0)
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
                        **league_encoded,
                        **label_encoded,
                        **team_encoded,
                        **visibility_encoded,
                    }
                    rows.append(row)
    
    df = pd.DataFrame(rows)
    df.fillna(0, inplace=True)
    
    return df
