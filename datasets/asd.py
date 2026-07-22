import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.io
import seaborn as sns
import math
import os
from matplotlib.patches import Patch


from aitlas.base import BaseDataset
from .schemas import FAIREOSchema
from aitlas.utils import image_loader


"""
This dataset is provided for research on aerial image segmentation. The dataset contains 80 high-resolution 
aerial images. To each image is associated a segmentation mask where objects were delineated.
"""

LABELS = ["background", "object boundary"]
color_mapping = [[0, 0, 0], [255, 255, 255]]

class ASDDataset(BaseDataset):

    schema = FAIREOSchema
    url = "http://jiangyeyuan.com/ASD/Aerial%20Image%20Segmentation%20Dataset.html"
    labels = LABELS
    name = "ASD"

    def __init__(self, config):
        # now call the constructor to validate the schema
        super().__init__(config)
        self.data_dir = self.config.data_dir
        self.csv_file = self.config.csv_file
        self.images, self.masks = self.load_dataset(self.data_dir, self.csv_file)


    def __getitem__(self, index):
        image = image_loader(self.images[index])
        mat = scipy.io.loadmat(self.masks[index])
        mask = mat['groundTruth']
        # the file contains 4 segmentation masks created by 4 different labelers
        # we consider the first mask and the boundary segmentation
        first = mask[0][0]
        mask = first[0][0][1]

        masks = [(mask == v) for v, label in enumerate(self.labels)]
        mask = np.stack(masks, axis=-1).astype("float32")

        return self.apply_transformations(image, mask)


    def load_dataset(self, data_dir, csv_file):
        if not self.labels:
            raise ValueError("You need to provide the list of labels for the dataset")

        ids = os.listdir(os.path.join(data_dir, 'Images'))

        if csv_file:
            ids = []
            df = pd.read_csv(csv_file, header=None, names=["path", "folder"])
            df = df["path"]
            for i in np.arange(len(df)):
                ids.append(df[i].split('/')[1])

        self.images = [os.path.join(data_dir, "Images", image_id) for image_id in ids]
        self.masks = [os.path.join(data_dir, "Human", image_id[: -4] + '.mat') for image_id in ids]         
        
        return self.images, self.masks

    def show_image(self, index, show_title=False):
        img, mask = self[index]

        img_mask = np.zeros([mask.shape[0], mask.shape[1], 3], np.uint8)
        legend_elements = []

        for i, label in enumerate(self.labels):
            legend_elements.append(
                Patch(
                    facecolor=tuple([x / 255 for x in color_mapping[i]]),
                    label=self.labels[i],
                )
            )
            img_mask[np.where(mask[:, :, i] == 1)] = color_mapping[i]

        fig = plt.figure(figsize=(15, 10))
        height_factor = math.ceil(len(self.labels)/3)
        if height_factor == 5:
            height_factor = 0.68
        elif height_factor == 4:
            height_factor = 0.70
        elif height_factor == 3:
            height_factor = 0.75
        else:
            height_factor = 0.78
        fig.legend(handles=legend_elements, bbox_to_anchor=(0.3, 0.9, 0.4, 0.4), ncol=2, mode='expand',
                   loc='lower left', prop={'size': 14})
        plt.subplot(1, 2, 1)
        img_normalized = (img - img.min()) / (img.max() - img.min()) if img.max() > img.min() else img
        plt.imshow(img_normalized)
        plt.axis("off")
        plt.subplot(1, 2, 2)
        plt.imshow(img_mask)
        plt.axis("off")
        fig.tight_layout()
        plt.show()
        return fig

    def __len__(self):
        return len(self.images)
    
    def get_labels(self):
        return self.labels

    def data_distribution_table(self):
        label_dist = {key: 0 for key in self.labels}
        for i in np.arange(len(self.images)):
            image, mask = self.__getitem__(i)
            for index, label in enumerate(self.labels):
                label_dist[self.labels[index]] += mask[:, :, index].sum()
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

    def apply_transformations(self, image, mask):
        if self.joint_transform:
            image, mask = self.joint_transform((image, mask))
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            mask = self.target_transform(mask)
        return image, mask