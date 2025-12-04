from ..base import BaseDataset
from .schemas import SSL4EOS12MLSchema
from ..utils import image_loader

import random
from itertools import compress
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
import cv2
import math
import os
import csv
import json

'''
SSL4EO-S12-ML dataset is a large-scale multi-label land cover land use classification dataset derived from 
SSL4EO-S12 images and Dynamic World segmentation maps. It consists of 780,371 multispectral Sentinel-2 images 
divided into 247,377 non-overlapping scenes each with 1-4 multi-seasonal patches. Each image has a multi-label 
annotation from one or more categories in 9 land cover land use classes.
'''

LABELS = ["water","forest","grass","flooded vegetation","crops","shrub & scrub","built","bare ground","snow & ice"]
NUM_CLASSES = 9


class SSL4EOS12MLDataset(BaseDataset):
    url = "https://github.com/zhu-xlab/softcon"
    schema = SSL4EOS12MLSchema
    labels = LABELS
    name = "SSL4EO-S12-ML"

    def __init__(self, config):
        # now call the constructor to validate the schema and load the data
        super().__init__(config)

        # load the data
        self.image_loader = image_loader
        self.data_dir = self.config.data_dir
        self.csv_file = self.config.csv_file
        self.label_file = self.config.label_file
        self.selection = self.config.selection
        self.imagery = self.config.imagery
        self.images = self.load_dataset(self.data_dir, self.label_file)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is a multi-hot vector.
        """

        # Sentinel-2 L1C
        img_path = os.path.join(self.data_dir, 's2_l1c/s2c', self.images[index][0])
        if self.selection == 'rgb':
            b02 = self.image_loader(os.path.join(img_path, 'B2.tif'))
            b03 = self.image_loader(os.path.join(img_path, 'B3.tif'))
            b04 = self.image_loader(os.path.join(img_path, 'B4.tif'))
            s2c_image = np.stack((b02,b03,b04))
        elif self.selection == 'all':
            b01 = self.image_loader(os.path.join(img_path, 'B1.tif'))
            b02 = self.image_loader(os.path.join(img_path, 'B2.tif'))
            b03 = self.image_loader(os.path.join(img_path, 'B3.tif'))
            b04 = self.image_loader(os.path.join(img_path, 'B4.tif'))
            b05 = self.image_loader(os.path.join(img_path, 'B5.tif'))
            b06 = self.image_loader(os.path.join(img_path, 'B6.tif'))
            b07 = self.image_loader(os.path.join(img_path, 'B7.tif'))
            b08 = self.image_loader(os.path.join(img_path, 'B8.tif'))
            b8a = self.image_loader(os.path.join(img_path, 'B8A.tif'))
            b09 = self.image_loader(os.path.join(img_path, 'B9.tif'))
            b10 = self.image_loader(os.path.join(img_path, 'B10.tif'))
            b11 = self.image_loader(os.path.join(img_path, 'B11.tif'))
            b12 = self.image_loader(os.path.join(img_path, 'B12.tif'))
            s2c_image = np.stack((b01,b02,b03,b04,b05,b06,b07,b08,b8a,b09,b10,b11,b12))
        # normalize image for better visualization
        img_min, img_max = s2c_image.min(), s2c_image.max()
        s2c_image = (s2c_image - img_min) / (img_max - img_min + 1e-8)

        # Sentinel-2 L2A
        # The Sentinel-2 L1C images were used for labeling. Therefore some of the Sentinel-2 L2A and Sentinel-1 images do not have any corresponding label.
        if os.path.exists(os.path.join(self.data_dir, 's2_l2a/s2a', self.images[index][0])):
            img_path = os.path.join(self.data_dir, 's2_l2a/s2a', self.images[index][0])
            if self.selection == 'rgb':
                b02 = self.image_loader(os.path.join(img_path, 'B2.tif'))
                b03 = self.image_loader(os.path.join(img_path, 'B3.tif'))
                b04 = self.image_loader(os.path.join(img_path, 'B4.tif'))
                s2a_image = np.stack((b02,b03,b04))
            elif self.selection == 'all':
                b01 = self.image_loader(os.path.join(img_path, 'B1.tif'))
                b02 = self.image_loader(os.path.join(img_path, 'B2.tif'))
                b03 = self.image_loader(os.path.join(img_path, 'B3.tif'))
                b04 = self.image_loader(os.path.join(img_path, 'B4.tif'))
                b05 = self.image_loader(os.path.join(img_path, 'B5.tif'))
                b06 = self.image_loader(os.path.join(img_path, 'B6.tif'))
                b07 = self.image_loader(os.path.join(img_path, 'B7.tif'))
                b08 = self.image_loader(os.path.join(img_path, 'B8.tif'))
                b8a = self.image_loader(os.path.join(img_path, 'B8A.tif'))
                b09 = self.image_loader(os.path.join(img_path, 'B9.tif'))
                b10 = self.image_loader(os.path.join(img_path, 'B10.tif'))
                b11 = self.image_loader(os.path.join(img_path, 'B11.tif'))
                b12 = self.image_loader(os.path.join(img_path, 'B12.tif'))
                s2a_image = np.stack((b01,b02,b03,b04,b05,b06,b07,b08,b8a,b09,b10,b11,b12))
            # normalize image
            img_min, img_max = s2a_image.min(), s2a_image.max()
            s2a_image = (s2a_image - img_min) / (img_max - img_min + 1e-8)
        else: # the s2c image has no correpsonding s2a image
            s2a_image = np.zeros((3, 264, 264))

        # Sentinel-1
        if os.path.exists(os.path.join(self.data_dir, 's1/s1', self.images[index][0])):
            img_path = os.path.join(self.data_dir, 's1/s1', self.images[index][0])
            vh = self.image_loader(os.path.join(img_path, 'VH.tif'))
            vv = self.image_loader(os.path.join(img_path, 'VV.tif'))
            vh_vv = np.array(vh/vv)
            s1_image = np.stack((vh,vv,vh_vv))
            # normalize image
            img_min, img_max = s1_image.min(), s1_image.max()
            s1_image = (s1_image - img_min) / (img_max - img_min + 1e-8)
        else: # the s2c image has no correpsonding s1 image
            s1_image = np.zeros((3, 264, 264))

        if self.imagery == 'S2 L2A':
            img = s2a_image.transpose(1,2,0)
        elif self.imagery == 'S2 L1C':
            img = s2c_image.transpose(1,2,0)
        elif self.imagery == 'S1':
            img = s1_image.transpose(1,2,0)
        elif self.imagery == 'all':
            img = [s2a_image.transpose(1,2,0),s2c_image.transpose(1,2,0),s1_image.transpose(1,2,0)]
        
        if self.transform:
            img = self.transform(img)
        target = self.images[index][1]
        if self.target_transform:
            target = self.target_transform(target)

        return img, target

    def load_dataset(self, data_dir, label_file):      
        images = []

        # csv_files created by BVLabs provide a 60/20/20 train/val/test split
        allowed_tiles = None
        if self.csv_file:
            with open(self.csv_file, "r") as f:
                reader = csv.reader(f)
                allowed_tiles = set(row[0] for row in reader) 

        with open(label_file, "r") as f:
            data = json.load(f) 
            for tile_id, timestamps in data.items():
                if allowed_tiles is not None and tile_id not in allowed_tiles:
                    continue
                for timestamp, labels in timestamps.items():
                    if len(labels)==0:
                        continue
                    # start with a multi-hot vector with all zeros, positive labels will be replaced by a 1
                    multi_hot = np.zeros(NUM_CLASSES, dtype=float)
                    for label in labels:
                        idx = int(label)
                        multi_hot[idx] = 1
                    path = os.path.join(tile_id, timestamp)
                    item = (path, multi_hot)
                    images.append(item)
        
        return images
    
    def __len__(self):
        return len(self.images)

    def get_labels(self):
        return self.labels
    
    def show_samples(self):
        df = pd.DataFrame(self.images, columns=('file name','labels'))
        return df.head(20)
    
    def data_distribution_table(self):
        df = pd.DataFrame(self.images, columns=('file name','labels'))
        labels_matrix = np.stack(df['labels'].values) 
        label_counts = labels_matrix.sum(axis=0)
        label_count_df = pd.DataFrame({
            "Label": self.labels,
            "Count": label_counts.astype(int)
        })
        return label_count_df
    
    def data_distribution_barchart(self):
        label_count = self.data_distribution_table()
        fig, ax = plt.subplots(figsize=(12, 10))
        sns.barplot(y="Label", x="Count", data=label_count, ax=ax)
        ax.set_title(
            "Labels distribution for {}".format(self.get_name()), pad=20, fontsize=18
        )
        return fig
    
    def show_image(self, index):
        labels_list = list(compress(self.labels, self[index][1]))
        if self.imagery == 'all':
            fig, axes = plt.subplots(1, 3, figsize=(18, 6))
            fig.suptitle(
            f"Image with index {index} from the dataset {self.get_name()}, with labels:\n"
            f"{', '.join(labels_list)}",
            fontsize=14,
            )

            axes[0].imshow(self[index][0][0])
            axes[0].set_title('Sentinel-2 L2A', fontsize=12)
            axes[0].axis("off")

            axes[1].imshow(self[index][0][1])
            axes[1].set_title('Sentinel-2 L1C', fontsize=12)
            axes[1].axis("off")

            axes[2].imshow(self[index][0][2])
            axes[2].set_title('Sentinel-1', fontsize=12)
            axes[2].axis("off")
        else: 
            fig = plt.figure(figsize=(8, 6))
            plt.title(
                f"Image with index {index} from the dataset {self.get_name()}, with labels:\n "
                f"{str(labels_list).strip('[]')}\n",
                fontsize=14,
            )
            plt.axis("off")
            plt.imshow(self[index][0])

        return fig
    
    def show_batch(self, size, show_title=True):
        if size % 4:
            raise ValueError("The provided size should be divided by 4!")
        image_indices = random.sample(range(0, len(self.images)), size)
        figure, ax = plt.subplots(int(size / 4), 4, figsize=(13.75, 2.0*int(size/4)))
        if show_title:
            figure.suptitle(
                "Example images with labels from {}".format(self.get_name()),
                fontsize=32,
                y=1.006,
            )
        for axes, image_index in zip(ax.flatten(), image_indices):
            labels_list = list(compress(self.labels, self[image_index][1]))
            height, width, depth = self[image_index][0].shape
            white_image = np.ones([height, width, 3], dtype=np.float32)
            text = '\n'. join(labels_list)
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_size = width/600 * 2.5
            font_thickness = math.ceil(width/600 * 4)
            x = 30

            for i, line in enumerate(text.split('\n')):
                textsize = cv2.getTextSize(line, font, font_size, font_thickness)[0]
                gap = textsize[1] + 5
                y = textsize[1] + i * gap
                cv2.putText(white_image, line, (x, y), font,
                            font_size,
                            (0, 0, 0),
                            font_thickness,
                            lineType=cv2.LINE_AA)

            display_image = np.hstack((self[image_index][0], white_image))
            axes.imshow(display_image)
            axes.set_xticks([])
            axes.set_yticks([])
            axes.axis('off')
        figure.tight_layout()
        return figure