import os
import random
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
from collections import Counter

from ..base import BaseDataset
from .schemas import LLCCDatasetSchema

'''
A large-scale dataset for land cover classification and representation learning, featuring global 
coverage with multi-spectral satellite imagery and 66 land cover classes.
'''

LABELS = [
    "Corn",
    "Cotton",
    "Rice",
    "Sorghum",
    "Sweet Corn",
    "Barley",
    "Durum Wheat",
    "Spring Wheat",
    "Winter Wheat",
    "Rye",
    "Oats",
    "Safflower",
    "Alfalfa",
    "Other Hay/Non Alfalfa",
    "Dry Beans",
    "Other Crops",
    "Watermelons",
    "Onions",
    "Peas",
    "Tomatoes",
    "Sod/Grass Seed",
    "Fallow/Idle Cropland",
    "Cherries",
    "Peaches",
    "Grapes",
    "Other Tree Crops",
    "Citrus",
    "Pecans",
    "Almonds",
    "Walnuts",
    "Pears",
    "Open Water",
    "Developed/Open Space",
    "Developed/Low Intensity",
    "Developed/Med Intensity",
    "Developed/High Intensity",
    "Barren",
    "Deciduous Forest",
    "Evergreen Forest",
    "Shrubland",
    "Grassland/Pasture",
    "Woody Wetlands",
    "Herbaceous Wetlands",
    "Pistachios",
    "Triticale",
    "Carrots",
    "Asparagus",
    "Garlic",
    "Cantaloupes",
    "Olives",
    "Oranges",
    "Honeydew Melons",
    "Pomegranates",
    "Nectarines",
    "Plums",
    "Dbl Crop WinWht/Corn",
    "Dbl Crop Oats/Corn",
    "Lettuce",
    "Dbl Crop WinWht/Sorghum",
    "Dbl Crop Barley/Corn",
    "Blueberries"
]


class LLCCDataset(BaseDataset):

    schema = LLCCDatasetSchema
    name = "Representation Learning for Land Cover Classification"
    url = "https://sustainlab-group.github.io/sustainbench/docs/datasets/sdg15/land_cover_representation.html"
    labels = LABELS

    def __init__(self, config):
        # now call the constructor to validate the schema and split the data
        super().__init__(config)
        self.data_dir = self.config.data_dir
        self.csv_file = self.config.csv_file
        self.selection = self.config.selection
        self.task = self.config.task
        self.annotations, self.data = self.load_dataset(
            self.data_dir, self.csv_file
        )


    def __getitem__(self, index):

        if self.selection == "triplets":

            img_anchor, img_neighbor, img_distant = self.data[index]

            # Handle 5-channel images by taking first 3 channels for RGB display
            img_a = np.load(os.path.join(self.data_dir, img_anchor + ".npy"))[:, :, :3]
            img_n = np.load(os.path.join(self.data_dir, img_neighbor + ".npy"))[:, :, :3]
            img_d = np.load(os.path.join(self.data_dir, img_distant + ".npy"))[:, :, :3]

            target_a, target_n, target_d = self.annotations[index]

            return (img_a, img_n, img_d), (target_a,target_n,target_d)

        if self.selection == "individual":

            image = self.data[index]
            img = np.load(os.path.join(self.data_dir, image + ".npy"))[:, :, :3]
            target = self.annotations[index]

            if self.transform:
                img = self.transform(img)

            return img, target


    def load_dataset(self, data_dir, csv_file):
        annotations = []
        data = []

        df = pd.read_csv(csv_file)

        for i in range(0, len(df), 3):
            anchor = df.iloc[i]["y"]
            neighbor = df.iloc[i + 1]["y"]
            distant = df.iloc[i + 2]["y"]

            name_anchor = os.path.splitext(os.path.basename(df.iloc[i]['file_name']))[0]
            name_neighbor = os.path.splitext(os.path.basename(df.iloc[i + 1]['file_name']))[0]
            name_distant = os.path.splitext(os.path.basename(df.iloc[i + 2]['file_name']))[0]

            if self.selection == "triplets":
                # annotations and data are list of triples (anchor, neighbor, distant)
                data.append((name_anchor, name_neighbor, name_distant))
                annotations.append((anchor, neighbor, distant))

            if self.selection == "individual":
                if self.task == "train":
                    if df.iloc[i]["split_str"] == "train":
                        names = [name_anchor, name_neighbor, name_distant]
                        label_list = [anchor, neighbor, distant]
                        for i in names:
                            data.append(i)
                        for i in label_list:
                            annotations.append(i)
                if self.task == "val":
                    if df.iloc[i]["split_str"] == "val":
                        names = [name_anchor, name_neighbor, name_distant]
                        label_list = [anchor, neighbor, distant]
                        for i in names:
                            data.append(i)
                        for i in label_list:
                            annotations.append(i)

        return annotations, data

    def __len__(self):
        return len(self.data)
    
    def get_labels(self):
        return self.labels

    def show_image(self, index):
        if self.selection == "triplets":
            (anchor, neighbor, distant), (ta, tn, td) = self.__getitem__(index)

            fig, axs = plt.subplots(1, 3, figsize=(15, 5))

            images = [anchor, neighbor, distant]
            titles = ['anchor', 'neighbor', 'distant']
            labels = [ta, tn, td]

            fig.suptitle(
                f"Images with index {index} from dataset {self.get_name()}",
                fontsize=16
            )

            for ax, img, title, label in zip(axs, images, titles, labels):

                ax.imshow(img)
                ax.set_title(f"{title} (label: {LABELS[label]})")
                ax.axis('off')

            plt.tight_layout(rect=[0, 0, 1, 0.95])

            return fig
        else:
            print("Change config.selection to 'triplets' for show_image() and show_batch() functions.")

    def show_batch(self, size, show_title=True):
        if self.selection == "triplets":
            image_indices = random.sample(range(len(self.data)), size)
            figure, axs = plt.subplots(size, 3, figsize=(15, 5 * size))

            if show_title:
                figure.suptitle(
                    "Example images with labels from {}".format(self.get_name()),
                    fontsize=32,
                    y=1.006,
                )
            titles = ['anchor', 'neighbor', 'distant']

            for row_idx, idx in enumerate(image_indices):
                (anchor, neighbor, distant), (ta, tn, td) = self.__getitem__(idx)
                images = [anchor, neighbor, distant]
                labels = [ta, tn, td]

                for col_idx, (ax, img, t, label) in enumerate(zip(axs[row_idx], images, titles, labels)):
                    if img.shape[-1] == 5:
                        img_display = img[:, :, :3]
                    else:
                        img_display = img
                    ax.imshow(img_display)
                    ax.set_title(f"{t} (label: {LABELS[label]})", fontsize=12)
                    ax.axis('off')

            plt.tight_layout(rect=[0, 0, 1, 0.95])
            return figure
        else:
            print("Change config.selection to 'triplets' for show_image() and show_batch() functions.")


    def show_samples(self):
        sample_data = []
        for i in range(min(20, len(self.annotations))):
            anchor_label, neighbor_label, distant_label = self.annotations[i]
            anchor_file, neighbor_file, distant_file = self.data[i]
            sample_data.append({
                'Anchor_File': anchor_file,
                'Anchor_Label': LABELS[anchor_label],
                'Neighbor_File': neighbor_file,
                'Neighbor_Label': LABELS[neighbor_label],
                'Distant_File': distant_file,
                'Distant_Label': LABELS[distant_label],
            })

        df = pd.DataFrame(sample_data)
        return df

    def data_distribution_table(self):
        all_labels = []
        for anchor_label, neighbor_label, distant_label in self.annotations:
            all_labels.extend([anchor_label, neighbor_label, distant_label])

        # Count unique labels
        label_counts = Counter(all_labels)

        df = pd.DataFrame({
            'Label': [LABELS[label_idx] for label_idx in label_counts.keys()],
            'Count': list(label_counts.values())
        })

        return df

    def data_distribution_barchart(self):
        label_count = self.data_distribution_table()
        fig, ax = plt.subplots(figsize=(12, 8))
        sns.barplot(data=label_count, y="Label", x="Count", ax=ax)
        ax.set_title(
            "Labels distribution for {}".format(self.get_name()), pad=20, fontsize=18
        )
        return fig