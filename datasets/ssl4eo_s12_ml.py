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
from skimage.transform import resize
import fnmatch

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
        self.bands_s2 = self.config.bands_s2
        self.bands_s1 = self.config.bands_s1
        self.images = self.load_dataset(self.data_dir, self.label_file)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is a multi-hot vector.
        """
        # The S2 L2A and S1 images don't have the exact same date of acquisition as S2 L1C
        # We find the matching S2 L2A and S1 images knowing that they have the same position index in the images folder
        tile = self.images[index][0].split('/')[0]
        file_name = self.images[index][0].split('/')[1]
        timestamps = []
        for timestamp in os.listdir(os.path.join(self.data_dir, 's2_l1c/s2c', tile)):
            timestamps.append(timestamp)
        timestamp_index = timestamps.index(file_name)

        # Sentinel-2 L1C
        if self.imagery == "S2 L1C":
            img_path = os.path.join(self.data_dir, 's2_l1c/s2c', self.images[index][0])
            if self.selection == 'rgb':
                b02 = self.image_loader(os.path.join(img_path, 'B2.tif'))
                b03 = self.image_loader(os.path.join(img_path, 'B3.tif'))
                b04 = self.image_loader(os.path.join(img_path, 'B4.tif'))
                s2c_image = np.stack((b02,b03,b04))
            elif self.selection == 'all':
                band_filenames = {
                    "B01": "B1.tif",  "B02": "B2.tif",  "B03": "B3.tif",
                    "B04": "B4.tif",  "B05": "B5.tif",  "B06": "B6.tif",
                    "B07": "B7.tif",  "B08": "B8.tif",  "B8A": "B8A.tif",
                    "B09": "B9.tif",  "B10": "B10.tif", "B11": "B11.tif",
                    "B12": "B12.tif"
                }
                # Load a 10m band as reference shape to resize images (all images must have the same shape for stacking)
                reference = self.image_loader(os.path.join(img_path, "B2.tif"))
                target_shape = reference.shape
                loaded_bands = []
                for filename in band_filenames.values():
                    arr = self.image_loader(os.path.join(img_path, filename))
                    if arr.shape != target_shape:
                        arr = resize(arr, target_shape, order=1, preserve_range=True).astype(arr.dtype)
                    loaded_bands.append(arr)
                s2c_image = np.stack(loaded_bands)
            elif self.selection == 'bands':
                if self.bands_s2 == None:
                    print("The config must contain a bands_s2 field with a list of chosen bands")
                s2_bands = ["B01","B02","B03","B04","B05","B06","B07","B08","B8A","B09","B10","B11","B12"] # all 13 accepted s2 bands
                band_filenames = {
                    "B01": "B1.tif",  "B02": "B2.tif",  "B03": "B3.tif",
                    "B04": "B4.tif",  "B05": "B5.tif",  "B06": "B6.tif",
                    "B07": "B7.tif",  "B08": "B8.tif",  "B8A": "B8A.tif",
                    "B09": "B9.tif",  "B10": "B10.tif", "B11": "B11.tif",
                    "B12": "B12.tif"
                }
                for band in self.bands_s2:
                    if band not in s2_bands:
                        print("The bands must be valid Sentinel-2 bands, i.e. one of B01, B02, B03, B04, B05, B06, B07, B08, B8A, B09, B10, B11, B12")
                        break  
                # Load a 10m band as reference shape to resize images
                reference = self.image_loader(os.path.join(img_path, "B2.tif")) 
                target_shape = reference.shape
                loaded_bands = []
                for band in self.bands_s2:
                    arr = self.image_loader(os.path.join(img_path, band_filenames[band]))
                    if arr.shape != target_shape:
                        arr = resize(arr, target_shape, order=1, preserve_range=True).astype(arr.dtype)
                    loaded_bands.append(arr)
                s2c_image = np.stack(loaded_bands)
            # normalize image 
            img_min, img_max = s2c_image.min(), s2c_image.max()
            s2c_image = (s2c_image - img_min) / (img_max - img_min + 1e-8)

        # Sentinel-2 L2A
        if self.imagery == "S2 L2A" or self.imagery == "all":
            # Find the corresponding L2A product using the index of the L1C
            s2a_timestamps = []
            for s2a_timestamp in os.listdir(os.path.join(self.data_dir, 's2_l2a/s2a', tile)):
                s2a_timestamps.append(s2a_timestamp)
            s2a = s2a_timestamps[timestamp_index]

            img_path = os.path.join(self.data_dir, 's2_l2a/s2a', tile, s2a)
            if self.selection == 'rgb':
                b02 = self.image_loader(os.path.join(img_path, 'B2.tif'))
                b03 = self.image_loader(os.path.join(img_path, 'B3.tif'))
                b04 = self.image_loader(os.path.join(img_path, 'B4.tif'))
                s2a_image = np.stack((b02,b03,b04))
            elif self.selection == 'all':
                band_filenames = {
                    "B01": "B1.tif",  "B02": "B2.tif",  "B03": "B3.tif",
                    "B04": "B4.tif",  "B05": "B5.tif",  "B06": "B6.tif",
                    "B07": "B7.tif",  "B08": "B8.tif",  "B8A": "B8A.tif",
                    "B09": "B9.tif",  "B11": "B11.tif", "B12": "B12.tif"
                }
                # Load a 10m band as reference shape to resize images 
                reference = self.image_loader(os.path.join(img_path, "B2.tif"))
                target_shape = reference.shape
                loaded_bands = []
                for filename in band_filenames.values():
                    arr = self.image_loader(os.path.join(img_path, filename))
                    if arr.shape != target_shape:
                        arr = resize(arr, target_shape, order=1, preserve_range=True).astype(arr.dtype)
                    loaded_bands.append(arr)
                s2a_image = np.stack(loaded_bands)
            elif self.selection == 'bands':
                if self.bands_s2 == None:
                    print("The config must contain a bands_s2 field with a list of chosen bands")
                s2_bands = ["B01","B02","B03","B04","B05","B06","B07","B08","B8A","B09","B11","B12"] 
                band_filenames = {
                    "B01": "B1.tif",  "B02": "B2.tif",  "B03": "B3.tif",
                    "B04": "B4.tif",  "B05": "B5.tif",  "B06": "B6.tif",
                    "B07": "B7.tif",  "B08": "B8.tif",  "B8A": "B8A.tif",
                    "B09": "B9.tif",  "B11": "B11.tif", "B12": "B12.tif"
                }
                for band in self.bands_s2:
                    if band not in s2_bands:
                        print("The bands must be valid Sentinel-2 bands, i.e. one of B01, B02, B03, B04, B05, B06, B07, B08, B8A, B09, B11, B12")
                        break
                # Load a 10m band as reference shape to resize images
                reference = self.image_loader(os.path.join(img_path, "B2.tif")) 
                target_shape = reference.shape
                loaded_bands = []
                for band in self.bands_s2:
                    arr = self.image_loader(os.path.join(img_path, band_filenames[band]))
                    if arr.shape != target_shape:
                        arr = resize(arr, target_shape, order=1, preserve_range=True).astype(arr.dtype)
                    loaded_bands.append(arr)
                s2a_image = np.stack(loaded_bands)
            # normalize image
            img_min, img_max = s2a_image.min(), s2a_image.max()
            s2a_image = (s2a_image - img_min) / (img_max - img_min + 1e-8)

        # Sentinel-1
        if self.imagery == "S1" or self.imagery == "all":
            # Find the corresponding S1 product using the index of the S2 L1C
            s1_timestamps = []
            for s1_timestamp in os.listdir(os.path.join(self.data_dir, 's1/s1', tile)):
                s1_timestamps.append(s1_timestamp.split('_')[4]) # keep the acquisition date only to have a similar ordering as S2 products
            s1 = s1_timestamps[timestamp_index]
            for s1_timestamp in os.listdir(os.path.join(self.data_dir, 's1/s1', tile)):
                if fnmatch.fnmatch(s1_timestamp, f"*{s1}*"):
                    s1_path = s1_timestamp

            img_path = os.path.join(self.data_dir, 's1/s1', tile, s1_path)
            if self.selection == 'rgb': # the bands will be are vv, vh and vv/vh
                vv = self.image_loader(os.path.join(img_path, 'VV.tif'))
                vh = self.image_loader(os.path.join(img_path, 'VH.tif'))
                vv_vh = np.array(vv/vh)
                s1_image = np.stack((vh,vv,vv_vh))
            elif self.selection == 'all':
                vv = self.image_loader(os.path.join(img_path, 'VV.tif'))
                vh = self.image_loader(os.path.join(img_path, 'VH.tif'))
                s1_image = np.stack((vh,vv))
            elif self.selection == 'bands':
                if self.bands_s1 == None:
                    print("The config must contain a bands_s1 field with a list of chosen bands")
                s1_bands = ["VV","VH"]
                for band in self.bands_s1:
                    if band not in s1_bands:
                        print("The bands must be valid Sentinel-1 bands, i.e. one of VV and VH")
                        break
                loaded_bands = [self.image_loader(os.path.join(img_path, f"{band}.tif")) for band in self.bands_s1]
                s1_image = np.stack(loaded_bands)
            # normalize image
            img_min, img_max = s1_image.min(), s1_image.max()
            s1_image = (s1_image - img_min) / (img_max - img_min + 1e-8)

        if self.imagery == 'S2 L2A':
            img = s2a_image.transpose(1,2,0)
        elif self.imagery == 'S2 L1C':
            img = s2c_image.transpose(1,2,0)
        elif self.imagery == 'S1':
            img = s1_image.transpose(1,2,0)
        elif self.imagery == 'all': # returns S1 and S2 L2A
            s1_t = s1_image.transpose(1, 2, 0)
            s2_t = s2a_image.transpose(1, 2, 0)
            # Images do not always have the exact same size (few pixels difference)
            # Find the maximum height and width
            max_h = max(s1_t.shape[0], s2_t.shape[0])
            max_w = max(s1_t.shape[1], s2_t.shape[1])
            # Calculate padding required (top, bottom), (left, right), (channels)
            pad_s1 = ((0, max_h - s1_t.shape[0]), (0, max_w - s1_t.shape[1]), (0, 0))
            pad_s2 = ((0, max_h - s2_t.shape[0]), (0, max_w - s2_t.shape[1]), (0, 0))
            # Apply padding if necessary
            if pad_s1[0][1] > 0 or pad_s1[1][1] > 0:
                s1_t = np.pad(s1_t, pad_s1, mode='constant', constant_values=0)
            if pad_s2[0][1] > 0 or pad_s2[1][1] > 0:
                s2_t = np.pad(s2_t, pad_s2, mode='constant', constant_values=0)
                
            img = np.dstack((s1_t, s2_t))

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
        if self.selection != "rgb":
            print("The selection parameter must be set to rgb for visualization")
        labels_list = list(compress(self.labels, self[index][1]))
        if self.imagery == 'all':
            fig, axes = plt.subplots(1, 2, figsize=(16, 8))
            fig.suptitle(
            f"Image with index {index} from the dataset {self.get_name()}, with labels:\n"
            f"{', '.join(labels_list)}",
            fontsize=14,
            )

            axes[0].imshow(self[index][0][:,:,:3])
            axes[0].set_title('Sentinel-1', fontsize=12)
            axes[0].axis("off")

            axes[1].imshow(self[index][0][:,:,3:])
            axes[1].set_title('Sentinel-2 L2A', fontsize=12)
            axes[1].axis("off")
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