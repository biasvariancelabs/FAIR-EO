import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import csv 
import random

from aitlas.base import BaseDataset
from .schemas import FAIREOSchema

"""
This dataset supports weakly supervised cropland classification in the Midwestern United States. The study 
region spans from 37°N to 41°30'N from 94°W to 86°W, covering over 450,000 km². It contains pixel-level
and image-level labels derived from USDA land cover data, which provide high-quality reference labels 
across the US. This dataloader handles the image-level labels.
"""


class WSCSImageClassificationDataset(BaseDataset):
    url = ["https://sustainlab-group.github.io/sustainbench/docs/datasets/sdg2/weakly_supervised_cropland.html"]
    name = "Weakly Supervised Cropland Segmentation"
    schema = FAIREOSchema
    labels = ["non-cropland", "cropland"]
    color_mapping = [[0, 0, 0],[255, 255, 255]]

    def __init__(self, config):
        # now call the constructor to validate the schema and split the data
        super().__init__(config)
        self.data_dir = self.config.data_dir
        self.mode = self.config.mode
        self.img_ids = self.load_dataset(self.data_dir, self.mode)

    def __getitem__(self, index):
        img_id = self.img_ids[index]

        image = np.load(os.path.join(self.data_dir, f"images/{img_id}.npy"))
        target = int(np.load(os.path.join(self.data_dir, f"image_level_labels/{img_id}_label_crop.npy"))[0])

        image = image[:, :, [3, 2, 1]]
        image = image.astype(np.float32)
        if image.max() > image.min():
            image = (image - image.min()) / (image.max() - image.min())

        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            target = self.target_transform(target)

        return (image, target)

    def load_dataset(self, data_dir, mode=None):
        img_ids = []
        test_files_path = os.path.join(data_dir, "dataset_splits", "test_files.txt")
        test_files = []
        with open(test_files_path, mode ='r')as file:
            csvFile = csv.reader(file)
            for lines in csvFile:
                test_files.append(lines[0])

        for filename in os.listdir(os.path.join(data_dir, "images")):
            if filename.endswith(".npy"):
                img_id = filename[:-4]
                if mode == 'train':
                    if img_id not in test_files:
                        img_ids.append(img_id)
                elif mode == 'test' or mode == 'val':
                    if img_id in test_files:
                        img_ids.append(img_id)
                elif mode == None:
                    img_ids.append(img_id)

        annot = []
        for lab in os.listdir(os.path.join(data_dir, "image_level_labels")):
            label = int(np.load(os.path.join(data_dir, "image_level_labels", lab))[0])
            annot.append({"Label": label})
        self.annot = annot

        return img_ids

    def __len__(self):
        return len(self.img_ids)
    
    def get_labels(self):
        return self.labels

    def data_distribution_table(self):
        df = pd.DataFrame(self.annot)
        df_count = df.groupby("Label").size().reset_index(name="Count")
        df_count["Label"] = [self.labels[i] for i in df_count["Label"]]
        return df_count[["Label", "Count"]]

    def data_distribution_barchart(self):
        label_count = self.data_distribution_table()
        fig, ax = plt.subplots(figsize=(12, 10))
        sns.barplot(x="Label", y="Count", data=label_count, ax=ax)
        ax.set_title(
            "Labels distribution for {}".format(self.get_name()), pad=20, fontsize=18
        )
        return fig
    
    def show_image(self, index):
        label = self.labels[self[index][1]]
        fig = plt.figure(figsize=(8, 6))
        plt.title(
            f"Image with index {index} from the dataset {self.get_name()}, with label {label}\n",
            fontsize=14,
        )
        plt.axis("off")
        plt.imshow(self[index][0])
        return fig
    
    def show_batch(self, size, show_title=True):
        if size % 5:
            raise ValueError("The provided size should be divided by 5!")
        image_indices = random.sample(range(0, len(self.img_ids)), size)
        figure, ax = plt.subplots(
            int(size / 5), 5, figsize=(13.75, 2.8 * int(size / 5))
        )
        if show_title:
            figure.suptitle(
                "Example images with labels from {}".format(self.get_name()),
                fontsize=32,
                y=1.006,
            )
        for axes, image_index in zip(ax.flatten(), image_indices):
            axes.imshow(self[image_index][0])
            axes.set_title(self.labels[self[image_index][1]], fontsize=18, pad=10)
            axes.set_xticks([])
            axes.set_yticks([])
        figure.tight_layout()
        return figure