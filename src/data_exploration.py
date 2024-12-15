import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

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