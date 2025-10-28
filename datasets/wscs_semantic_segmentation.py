import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import csv 
from matplotlib.patches import Patch
import math

from ..base import BaseDataset
from .schemas import FAIREOSchema


"""
This dataset supports weakly supervised cropland classification in the Midwestern United States. It contains 
single pixel-level and image-level labels derived from USDA land cover data, which provide high-quality 
reference labels across the US. This dataloader considers the pixel-level labels for the semantic segmentation 
task.
"""


class WSCSSemanticSegmentationDataset(BaseDataset):
    url = ["https://sustainlab-group.github.io/sustainbench/docs/datasets/sdg2/weakly_supervised_cropland.html"]
    name = "Weakly Supervised Cropland Segmentation"
    schema = FAIREOSchema
    labels = ["background", "cropland"]
    color_mapping = [[0, 0, 0], [195, 195, 195]]

    def __init__(self, config):
        # now call the constructor to validate the schema and split the data
        super().__init__(config)
        self.data_dir = self.config.data_dir
        self.mode = self.config.mode
        self.img_ids = self.load_dataset(self.data_dir, self.mode)

    def __getitem__(self, index):
        img_id = self.img_ids[index]
        image = np.load(os.path.join(self.data_dir, f"images/{img_id}.npy"))

        mask = np.load(os.path.join(self.data_dir, f"single_pixel_masks/mask_{img_id}.npy"))
        masks = [(mask == v) for v, label in enumerate(self.labels)]
        mask = np.stack(masks, axis=-1).astype("float32")

        image = image[:, :, [3, 2, 1]]
        image = image.astype(np.float32)
        # normalize the image
        if image.max() > image.min():
            image = (image - image.min()) / (image.max() - image.min())

        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            mask = self.target_transform(mask)
        return image, mask

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

        return img_ids

    def __len__(self):
        return len(self.img_ids)
    
    def get_labels(self):
        return self.labels
    
    def data_distribution_table(self):
        label_dist = {key: 0 for key in self.labels}
        for image, mask in self.dataloader():
            for index, label in enumerate(self.labels):
                label_dist[self.labels[index]] += mask[:, :, :, index].sum()
        label_count = pd.DataFrame.from_dict(label_dist, orient='index')
        label_count.columns = ["Number of pixels"]
        label_count = label_count.astype(float)
        return label_count

    def data_distribution_barchart(self, show_title=True):
        label_count = self.data_distribution_table()
        fig, ax = plt.subplots(figsize=(12, 10))
        sns.barplot(data=label_count, x=label_count.index, y='Number of pixels', ax=ax)
        if show_title:
            ax.set_title(
                "Labels distribution for {}".format(self.get_name()), pad=20, fontsize=18
            )
        return fig
    
    def show_image(self, index, show_title=False):
        img, mask = self[index]
        img_mask = np.zeros([mask.shape[0], mask.shape[1], 3], np.uint8)
        legend_elements = []
        for i, label in enumerate(self.labels):
            legend_elements.append(
                Patch(
                    facecolor=tuple([x / 255 for x in self.color_mapping[i]]),
                    label=self.labels[i],
                )
            )
            img_mask[np.where(mask[:, :, i] == 1)] = self.color_mapping[i]

        fig = plt.figure(figsize=(10, 8))
        height_factor = math.ceil(len(self.labels) / 3)
        if height_factor == 4:
            height_factor = 0.73
        elif height_factor == 2:
            height_factor = 0.80
        else:
            height_factor = 0.81
        fig.legend(
            handles=legend_elements,
            bbox_to_anchor=(0.2, height_factor, 0.6, 0.2),
            ncol=3,
            mode="expand",
            loc="lower left",
            prop={"size": 12},
        )
        plt.subplot(1, 2, 1)

        img_normalized = (img - img.min()) / (img.max() - img.min())
        if img.max() == img.min():  # Handle edge case
            img_normalized = np.zeros_like(img)

        plt.imshow(img_normalized)
        plt.axis("off")
        plt.subplot(1, 2, 2)
        plt.imshow(img_mask)
        plt.axis("off")
        fig.tight_layout()
        plt.show()
        return fig