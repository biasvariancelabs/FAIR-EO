import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.io
import seaborn as sns
import math
from matplotlib.patches import Patch

from ..base import BaseDataset
from .schemas import MatDatasetSchema


"""
Qingdao UAV-borne HSI (QUH) dataset consists of three sub-datasets: QUH-Tangdaowan, QUH-Qingyun
and QUH-Pingan, for precise land cover classification. Each of these scenes has its own classification 
schema. 

The QUH-Tangdaowan dataset: it contains four vegetation species: Coniferous pine, Buxus sinica, Populus, 
Ulmus pumila L and three pavements: Flagging, Boardwalk, Gravel road. 
The QUH-Qingyun dataset: it has areas obscured by building shadows, including classes such as Trees, Car, 
and Asphalt road.
The QUH-Pingan dataset: it has landcovers with large differences in size: Seawater, Road at large scales, 
Trees, Floating pier, Brick houses, Steel houses at medium scales, and Ship, Car at small scales.
"""

LABELS_T = [ "Undefined", "Rubber track", "Flagging", "Sandy", "Asphalt", "Boardwalk", "Rocky shallows", "Grassland", 
            "Bulrush", "Gravel road","Ligustrum vicaryi", "Coniferous pine", "Spiraea", "Bare soil", "Buxus sinica", 
            "Photinia serrulata", "Populus", "Ulmus pumila L",  "Seawater"]
LABELS_Q = [ "Undefined", "Trees", "Concrete building", "Car", "Ironhide building", "Plastic playground", "Asphalt road"]
LABELS_P = [ "Undefined", "Ship", "Seawater", "Trees", "Concrete structure building", "Floating pier", "Brick houses", 
            "Steel houses", "Wharf construction land", "Car", "Road"]

color_mapping = [
    [0, 0, 0],          # 0: Black (undefined)
    [255, 0, 0],        # 1: Bright Red
    [0, 255, 0],        # 2: Bright Green
    [0, 0, 255],        # 3: Bright Blue
    [255, 255, 0],      # 4: Yellow
    [255, 0, 255],      # 5: Magenta
    [0, 255, 255],      # 6: Cyan
    [255, 128, 0],      # 7: Orange
    [128, 0, 255],      # 8: Purple
    [255, 192, 203],    # 9: Pink
    [0, 128, 0],        # 10: Dark Green
    [128, 128, 0],      # 11: Olive
    [0, 128, 128],      # 12: Teal
    [128, 0, 0],        # 13: Maroon
    [0, 0, 128],        # 14: Navy
    [255, 165, 0],      # 15: Orange Red
    [75, 0, 130],       # 16: Indigo
    [240, 230, 140],    # 17: Khaki
    [220, 20, 60]       # 18: Crimson
]

IMG_W, IMG_H = 256, 256

class QUHDataset(BaseDataset):
    schema = MatDatasetSchema
    name = "Qingdao UAV-borne HSI (QUH)"
    url = "https://zenodo.org/records/8223066"

    def __init__(self, config):
        # now call the constructor to validate the schema
        super().__init__(config)

        self.mat_file = self.config.mat_file
        self.filename = self.extract_dataset_name(self.mat_file)
        self.mode = self.config.mode
        self.labels, self.data = self.load_dataset(self.mat_file)


    def __getitem__(self, index):
        mat = scipy.io.loadmat(self.mat_file)
        scene = mat[self.label]

        lab_dir = os.path.join(os.path.dirname(self.mat_file), f'QUH-{self.filename}_GT.mat')
        label_mat = scipy.io.loadmat(lab_dir)
        label = label_mat[self.label+"GT"]

        x, y = self.data[index]

        image = scene[y:y+IMG_H, x:x+IMG_W, [75, 45, 25]]
        mask = label[y:y+IMG_H, x:x+IMG_W]
        masks = [(mask == v) for v, label in enumerate(self.labels)]
        mask = np.stack(masks, axis=-1).astype("float32")
        
        return self.apply_transformations(image, mask)


    def load_dataset(self, mat_file):

        if self.filename == "Pingan":
            self.label = "Haigang"
            labels = LABELS_P
        elif self.filename ==   "Qingyun":
            self.label = "Chengqu"
            labels = LABELS_Q
        else:
            self.label = "Tangdaowan"
            labels = LABELS_T

        mat = scipy.io.loadmat(self.mat_file)
        scene = mat[self.label]

        height, width, _ = scene.shape
        data = []
        stride = IMG_H

        for y in range(0, height - IMG_H + 1, stride):
            for x in range(0, width - IMG_W + 1, stride):
                if y + IMG_H <= height and x + IMG_W <= width:
                    data.append((x,y))

        train = round((len(data)/100)*80)
        val = round((len(data)/100)*10)
        test = round((len(data)/100)*10)
        if self.mode == 'train':
            data = data[: train]
        if self.mode == 'val':
            data = data[train : train+val]

        return labels, data

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
        fig.legend(handles=legend_elements, bbox_to_anchor=(0.2, height_factor, 0.6, 0.25), ncol=3, mode='expand',
                loc='lower left', prop={'size': 10})
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
        return len(self.data)
    
    def extract_dataset_name(self, mat_file):
        filename = os.path.basename(mat_file)
        name_without_ext = os.path.splitext(filename)[0]
        return name_without_ext.split('-')[1]
    
    def get_labels(self):
        return self.labels

    def data_distribution_table(self):

        lab_dir = os.path.join(os.path.dirname(self.mat_file), f'QUH-{self.filename}_GT.mat')
        label_mat = scipy.io.loadmat(lab_dir)
        label = label_mat[self.label+"GT"]

        label_dist = {key: 0 for key in self.labels}
        for index, label_name in enumerate(self.labels):
            label_dist[label_name] = (label == index).sum()

        label_count = pd.DataFrame.from_dict(label_dist, orient='index')
        label_count.columns = ["Number of pixels"]
        return label_count

    def data_distribution_barchart(self, show_title=True):
        label_count = self.data_distribution_table()
        fig, ax = plt.subplots(figsize=(12, 10))
        sns.barplot(data=label_count, x=label_count.index, y='Number of pixels', ax=ax)
        if show_title:
            ax.set_title(
                "Labels distribution for {}".format(self.filename), pad=20, fontsize=18
            )
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        return fig

    def apply_transformations(self, image, mask):
        if self.joint_transform:
            image, mask = self.joint_transform((image, mask))
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            mask = self.target_transform(mask)
        return image, mask