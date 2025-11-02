
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
import pandas as pd
import seaborn as sns

from model.labels_manager import LabelsManager

class LabelsPrinter():
    
    __instance = None

    def __new__(cls, *args, **kwargs):
        if cls.__instance is None:
            cls.__instance = super(LabelsPrinter, cls).__new__(cls)
        return cls.__instance

    def __init__(self, labels_manager: LabelsManager = None):
        if not labels_manager:
            raise ValueError("Argument labels_manager is missing.")
        self.__annotations_df = labels_manager.get_annotations_dataframe()
        self.__labels = labels_manager.get_labels()
        self.__label_colors = labels_manager.get_label_colors()
        

    def get_instance(self):
        return self.__instance

    def print_labels_distribution(
            self, 
            title="Distribution of events across all games.",
            fig_size=(10, 6)        
        ):
        label_distribution = self.__annotations_df[list(self.__labels)].sum()
        label_distribution_sorted = label_distribution.sort_values(ascending=True)

        color_map_dict = dict(zip(list(self.__labels), self.__label_colors))
        y_labels = label_distribution_sorted.index
        bar_colors = [color_map_dict[label] for label in y_labels]
        
        plt.figure(figsize=fig_size)
        ax = sns.barplot(
            x=label_distribution_sorted.values, 
            y=label_distribution_sorted.index,
            palette=bar_colors
        )

        for p in ax.patches:
            ax.text(
                p.get_width(),
                p.get_y() + p.get_height() / 2,
                '{:.0f}'.format(p.get_width()),
                ha='left',
                va='center',
                fontsize=10,
                color='black'
            )

        plt.title(title)
        plt.xlabel("Count")
        plt.ylabel("Label")
        plt.show()

    def print_labels_distribution_over_game_intervals(self, fig_size=(10, 6)):
    
        def convert_time_to_minutes(row):
            time_parts = row['time'].split(':') if isinstance(row['time'], str) else []
            if len(time_parts) == 2:
                minutes = int(time_parts[0])
                if row['second_half'] == 1:
                    minutes += 45
                return minutes
            return None

        self.__annotations_df['time_minutes'] = self.__annotations_df.apply(convert_time_to_minutes, axis=1)
        self.__annotations_df['time_interval_start'] = (self.__annotations_df['time_minutes'] // 5) * 5
        self.__annotations_df['time_interval'] = (
            self.__annotations_df['time_interval_start'].astype(str) + "-" + 
            (self.__annotations_df['time_interval_start'] + 5).astype(str)
        )

        event_counts = self.__annotations_df.groupby(['time_interval'])[list(self.__labels)].sum()
        
        event_counts.sort_index(ascending=False).plot(
            kind='barh', 
            stacked=True, 
            colormap=ListedColormap(self.__label_colors), 
            ax=plt.gca(), 
            figsize=fig_size
        )

        plt.title("Distribution of events per time intervals (stacked across all games)")
        plt.xlabel("Count") 
        plt.ylabel("Time interval (minutes)") 
        
        plt.legend(title="Event labels", bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.tight_layout()
        plt.show()

