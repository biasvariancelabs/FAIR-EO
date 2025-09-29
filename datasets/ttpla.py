import os
import math
import numpy as np
import json
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import csv
from matplotlib.patches import Patch
from skimage.draw import polygon

from .semantic_segmentation import SemanticSegmentationDataset
from .schemas import SegmentationDatasetSchema
from ..utils import image_loader

"""
TTPLA is an aerial-image dataset for detection and segmentation of transmission towers (TTs) 
and power lines (PLs). The dataset consists of 1100 images, as well as manually labeled 8987 
instances of TTs and PLs.
"""


class TTPLADataset(SemanticSegmentationDataset):

    url = ["https://github.com/R3ab/ttpla_dataset?tab=readme-ov-file"]
    name = "Transmission Towers and Powers Lines (TTPLA)"
    schemas = SegmentationDatasetSchema
    labels = [
        "void",
        "cable",
        "tower_lattice",
        "tower_tucohy",
        "tower_wooden"
    ]
    color_mapping = [
        [0,0,0],
        [232,12,12],
        [225,232,12],
        [48,232,12],
        [12, 192, 232]
    ]

    def __init__(self, config):
        # now call the constructor to validate the schema and split the data
        super().__init__(config)
        self.data_dir = self.config.data_dir
        self.csv_file = self.config.csv_file

    def __getitem__(self, index):
        img_id = self.data[index]
        image = image_loader(os.path.join(self.data_dir,f'{img_id}.jpg'))
        image = image.astype(np.float32)

        img_h, img_w = image.shape[:2]
        mask_2d = np.zeros((img_h, img_w), dtype = np.int64)
        annot_file_path = os.path.join(self.data_dir, f'{img_id}.json')
        # objects (TT and PL) are annotated with point coordinates delineating the polygons
        with open(annot_file_path, "r") as f:
            annot_json = json.load(f)

        for shape in annot_json['shapes']:
            label = shape['label'].lower()
            coords = shape['points']
            # convert outer point coordinates into an object polygon with every pixel labeled
            mask_2d = self.add_object(mask_2d, coords, label)

        num_classes = len(self.labels)
        mask = np.eye(num_classes)[mask_2d]

        return self.apply_transformations(image, mask)


    def load_dataset(self, data_dir, csv_file=None):
        img_ids = []
        split_list = []
        if csv_file: 
            with open(csv_file, mode ='r')as file:
                csvFile = csv.reader(file)
                for lines in csvFile:
                    split_list.append(lines[0].split('/')[1])

        for filename in os.listdir(data_dir):
            if filename.endswith('.jpg'):
                img_id = filename[:-4]
                if csv_file:
                    if filename in split_list:
                        img_ids.append(img_id)
                else:
                    img_ids.append(img_id)

        self.data = img_ids

    
    def add_object(self, mask:np.ndarray, coords:list, label)->np.ndarray:
        """
        Fills the target object polygons with labeled pixels
        """
        # Handle 3D coordinates (x, y, z) by ignoring z
        pixel_coords = [(int(round(x)), int(round(y))) for x, y in coords]

        if len(pixel_coords) < 4:
            return mask

        # Extract x and y coordinates for polygon filling
        x_coords = [coord[0] for coord in pixel_coords[:-1]]
        y_coords = [coord[1] for coord in pixel_coords[:-1]]

        rr, cc = polygon(y_coords, x_coords, shape=mask.shape)

        int_label = [i for i,x in enumerate(self.labels) if label == x]
        int_label = int_label[0]
        mask[rr, cc] = int_label

        return mask


    def data_distribution_table(self):
        stats = {
            "void" : 0,
            "cable" : 0,
            "tower_lattice" : 0,
            "tower_tucohy" : 0,
            "tower_wooden" : 0
        }

        total_pixels_all_images = 3840 * 2160 * len(self.data)

        for img_id in self.data:
            total_object_pixels = 0
            annot_file_path = os.path.join(self.data_dir, f'{img_id}.json')
            with open(annot_file_path, "r") as f:
                annot_json = json.load(f)

            for shape in annot_json['shapes']:
                label = shape['label'].lower()
                coords = shape['points']
                area = self.calculate_polygon_area(coords)
                stats[label] += int(area)

                total_object_pixels += int(area)

            stats['void'] = total_pixels_all_images - total_object_pixels

        label_count = pd.DataFrame.from_dict(stats, orient='index')
        label_count.columns = ["Number of pixels"]

        return label_count


    def calculate_polygon_area(self, points):
        """
        Calculates the area using the Shoelace formula
        """
        n = len(points)
        area = 0.0
        for i in range(n - 1):
            area += points[i][0] * points[i + 1][1]
            area -= points[i + 1][0] * points[i][1]
        return abs(area) / 2.0


    def data_distribution_barchart(self, show_title=True):
        label_count = self.data_distribution_table()
        fig, ax = plt.subplots(figsize=(12, 10))
        sns.barplot(data=label_count, x=label_count.index, y='Number of pixels', ax=ax)
        if show_title:
            ax.set_title(
                "Labels distribution for {}".format(self.get_name()), pad=20, fontsize=18
            )
        return fig

    def __len__(self):
        return len(self.data)
    
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
        fig.legend(
            handles=legend_elements,
            bbox_to_anchor=(0.2, height_factor, 0.6, 0.2),
            ncol=3,
            mode='expand',
            loc='lower left',
            prop={'size': 12}
        )
        plt.subplot(1, 2, 1)

        img_normalized = (img - img.min()) / (img.max() - img.min())
        if img.max() == img.min():
            img_normalized = np.zeros_like(img)

        plt.imshow(img_normalized)
        plt.axis("off")
        plt.subplot(1, 2, 2)
        plt.imshow(img_mask)
        plt.axis("off")
        fig.tight_layout()
        plt.show()
        return fig