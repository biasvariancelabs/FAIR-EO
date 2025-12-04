import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import json
from skimage.draw import polygon
from matplotlib.patches import Patch
import math

from ..utils import image_loader
from .schemas import FAIREOObjectDetectionYoloSchema
from ..base import BaseDataset

'''
The Urban Building Classification (UBC) dataset was created for building detection and classification from very 
high-resolution satellite imagery with the focus on object-level interpretation of individual buildings. It is 
meant to provide not only a flexible test platform for object detection algorithms but also a solid basis for 
the comparison of city morphologies and the investigation of urban planning.
'''

class UBCDataset(BaseDataset):
    schema = FAIREOObjectDetectionYoloSchema
    name = "UBC dataset"
    url = "https://drive.google.com/file/d/1XnKFKqjoa95PLXFw01HcXx4Az49Qw37i/view"
    labels = ["no-data","flat","shed","gable","row","hipped","arched","revolved","freeshape","other"]
    color_mapping = [[0,0,0],[255,0,0],[0,0,255],[255,0,255],[0,255,0],[255,255,0],[0,255,255],[192,192,192],[192,50,50],[255,255,255]]

    def __init__(self, config):
        # now call the constructor to validate the schema and split the data
        super().__init__(config)
        self.data_dir = self.config.data_dir
        self.annotation_dir = self.config.annotation_dir
        self.data = self.load_dataset(
            self.data_dir, self.annotation_dir
        )


    def __getitem__(self, index):
        img_name = self.data[index]
        image = image_loader(os.path.join(self.data_dir, img_name))
        img_h, img_w = image.shape[:2]
        mask_2d = np.zeros((img_h, img_w), dtype = np.int64)

        with open(self.annotation_dir, "r") as f:
            annot_json = json.load(f)

        img_id= None
        for img in annot_json["images"]:
            if img["file_name"] == img_name:
                img_id = img["id"]
                break

        for ann in annot_json["annotations"]:
            if ann["image_id"] == img_id:
                coords = ann["segmentation"][0]
                label = ann["category_id"]
                mask_2d = self.add_building(mask_2d, coords, label)

        num_classes = len(self.labels)
        mask = np.eye(num_classes)[mask_2d]
            
        image, mask = self.apply_transformations(image, mask)

        return image, mask


    def load_dataset(self, data_dir, annotation_dir):
        data = []

        with open(annotation_dir, "r") as f:
            annot = json.load(f)

        for ann in annot["images"]:
            data.append(ann["file_name"])

        return data
    
    def __len__(self):
        return len(self.data)
    
    def get_labels(self):
        return self.labels

    def apply_transformations(self, image, mask):
        if self.joint_transform:
            image, mask = self.joint_transform((image, mask))
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            mask = self.target_transform(mask)
        return image, mask
    
    def add_building(self, mask:np.ndarray, coords:list, label)->np.ndarray:
        # Convert flat list [x1, y1, x2, y2, ...] to [(x1, y1), (x2, y2), ...]
        pixel_coords = [(int(round(coords[i])), int(round(coords[i + 1]))) for i in range(0, len(coords), 2)]

        if len(pixel_coords) < 4:
            return mask

        # Extract x and y coordinates for polygon filling
        x_coords = [coord[0] for coord in pixel_coords[:-1]]
        y_coords = [coord[1] for coord in pixel_coords[:-1]]

        rr, cc = polygon(y_coords, x_coords, shape=mask.shape)
        mask[rr, cc] = int(label)
        return mask

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
        height_factor = math.ceil(len(self.labels)/3)
        if height_factor == 4:
            height_factor = 0.73
        elif height_factor == 2:
            height_factor = 0.80
        else:
            height_factor = 0.81
        fig.legend(handles=legend_elements, bbox_to_anchor=(0.2, height_factor, 0.6, 0.2), ncol=3, mode='expand',
                   loc='lower left', prop={'size': 12})
        plt.subplot(1, 2, 1)
        plt.imshow(img)
        plt.axis("off")
        plt.subplot(1, 2, 2)
        plt.imshow(img_mask)
        plt.axis("off")
        fig.tight_layout()
        plt.show()
        return fig
    
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