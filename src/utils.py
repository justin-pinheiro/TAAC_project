import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
import json

class Utilities:
    """
    Contains utility functions for logging, visualization, and data handling.

    Methods:
    --------
    - log_metrics(metrics, epoch): Log training and evaluation metrics.
    - plot_training_curves(metrics): Plot loss and accuracy curves for training.
    """

    def __init__(self, data_path):
        """
        Constructor method to initialize the class attributes.

        Parameters:
        ----------
        - attribute1: Path to the SoccerNet downloaded database
        """
        self.data_path = data_path

    def print_labels_distribution(annotations_df):
        label_columns = [col for col in annotations_df.columns if col.startswith('label_')]
        
        label_distribution = annotations_df[label_columns].sum()
        label_distribution_sorted = label_distribution.sort_values(ascending=False)

        plt.figure(figsize=(10, 6))
        sns.barplot(x=label_distribution_sorted.index, y=label_distribution_sorted.values, palette="viridis")
        plt.title("Distribution of labels across all games")
        plt.xlabel("Label")
        plt.ylabel("Count")
        plt.xticks(rotation=45)
        plt.show()

    def print_labels_frequency(annotations_df):
        label_columns = [col for col in annotations_df.columns if col.startswith('label_')]

        label_counts_per_game = annotations_df.groupby('game_id')[label_columns].sum()
        avg_label_freq = label_counts_per_game.mean()
        avg_label_freq_sorted = avg_label_freq.sort_values(ascending=False)

        plt.figure(figsize=(10, 6))
        sns.barplot(x=avg_label_freq_sorted.index, y=avg_label_freq_sorted.values, palette="viridis")
        plt.title("Average frequency of labels per game")
        plt.xlabel("Label")
        plt.ylabel("Average frequency")
        plt.xticks(rotation=45)
        plt.show()

    def print_labels_frequency_per_league(annotations_df):
        label_columns = [col for col in annotations_df.columns if col.startswith('label_')]
        league_columns = [col for col in annotations_df.columns if col.startswith('league_')]

        league_label_counts = {}

        for league in league_columns:
            league_data = annotations_df[annotations_df[league] == 1]
            league_label_counts[league] = league_data[label_columns].sum()

        games_per_league = annotations_df[league_columns].sum()

        # Normalize the event counts by dividing by the number of games per league
        normalized_counts = {
            league: league_label_counts[league] / games_per_league[league] for league in league_columns
        }

        normalized_df = pd.DataFrame(normalized_counts)

        nrows = (len(label_columns) // 2) + 1  # Dynamically calculate rows based on the number of labels
        fig, axes = plt.subplots(nrows=nrows, ncols=2, figsize=(18, nrows * 6), constrained_layout=True)
        axes = axes.flatten() 

        for i, label in enumerate(label_columns):
            ax = axes[i]
            normalized_df.loc[label].plot(kind='bar', ax=ax, color='skyblue', edgecolor='black')
            
            ax.set_title(f"Average '{label}' label across leagues")
            ax.set_ylabel("Average labels per game")
            ax.set_xlabel("League")
            ax.tick_params(axis='x', rotation=45)
            ax.grid(axis='y', linestyle='--', alpha=0.5)

        # Hide unused subplots if any
        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])

        plt.suptitle("Comparison of labels frequency accross leagues", fontsize=16, y=1.02)
        plt.show()

    def print_labels_distribution_over_game_intervals(annotations_df):
        
        # convert 'time' to minutes
        def convert_time_to_minutes(row):
            time_parts = row['time'].split(':') if isinstance(row['time'], str) else []
            if len(time_parts) == 2:
                minutes = int(time_parts[0])
                if row['second_half'] == 1:
                    minutes += 45
                return minutes
            return None

        annotations_df['time_minutes'] = annotations_df.apply(convert_time_to_minutes, axis=1)

        annotations_df['time_interval'] = (annotations_df['time_minutes'] // 5) * 5

        label_columns = [col for col in annotations_df.columns if col.startswith('label_')]

        event_counts = annotations_df.groupby(['time_interval'])[label_columns].sum()

        plt.figure(figsize=(12, 8))
        event_counts.plot(kind='bar', stacked=True, colormap='tab20', figsize=(12, 8))

        plt.title("Event Time Distribution Across Matches (Stacked by Label)")
        plt.xlabel("Time Interval (minutes)")
        plt.ylabel("Event Frequency")
        plt.xticks(rotation=45)
        plt.legend(title="Event Labels", bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(axis='y', linestyle='--', alpha=0.7)

        plt.tight_layout()
        plt.show()

    def parse_labels_json_with_feature_engineering(self):
        """
        Traverse the directory structure and extract annotations into a pandas DataFrame,
        applying feature engineering and ensuring no missing values (NaNs).
        
        Parameters:
        - base_path (str): Root directory containing the league folders.
        
        Returns:
        - pd.DataFrame: DataFrame containing all annotations with relevant metadata and engineered features.
        """
        rows = []

        for league in os.listdir(self.data_path):
            league_path = os.path.join(self.data_path, league)
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

    def save_labels_in_csv(self):
        
        for league in os.listdir(self.data_path):
            league_path = os.path.join(self.data_path, league)
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
                    
                    df = self.__parse_labels_json(labels_file)
                    
                    labels = []
                    halves = []
                    times = []
                    
                    for index, row in df.iterrows():
                        game_time = row['game_time']
                        
                        half, time_str = game_time.split(' - ')
                        
                        labels.append(row['label'])
                        halves.append(int(half))
                        times.append(time_str)
                    
                    output_dataframe = pd.DataFrame({
                        'label': labels,
                        'half': halves,
                        'time': times
                    })
                    
                    output_dataframe.to_csv(os.path.join(game_path, "labels.csv"), index=False)
                    print(f"Labels saved to {os.path.join(game_path, 'labels.csv')}")




