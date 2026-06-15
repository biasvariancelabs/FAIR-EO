import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import seaborn as sns
import csv

from ..base import BaseDataset
from ..utils import image_loader
from .schemas import Sen1Floods11Schema

"""
Sen1Floods11 is a georeferenced dataset to train and test deep learning flood algorithms for Sentinel-1.
"""


class Sen1Floods11Dataset(BaseDataset):
    url = "https://github.com/cloudtostreet/Sen1Floods11"
    name = "Sen1Floods11"
    schema = Sen1Floods11Schema

    labels = ["not water","water"]
    color_mapping = [[0,0,0],[255,255,255]]


    def __init__(self, config):
        # now call the constructor to validate the schema and split the data
        super().__init__(config)
        self.source_dir = self.config.source_dir
        self.annot_dir = self.config.annot_dir
        self.imagery = self.config.imagery
        self.annotations = self.config.annotations
        self.selection = self.config.selection
        self.csv_file = self.config.csv_file
        self.images, self.masks = self.load_dataset(self.source_dir, self.annot_dir, self.csv_file)


    def __getitem__(self, index):
        if self.imagery == 'S2' and self.selection == 'all': # 13 bands image
            image = image_loader(self.images[index])
            if image.shape[2] == 512:
                image = image.transpose(1,2,0)
        elif self.imagery == 'S2' and self.selection == 'rgb': # 3 bands image
            image = image_loader(self.images[index])
            if image.shape[2] == 512:
                image = image.transpose(1,2,0)[:,:,[3,2,1]]
            else:
                image = image[:,:,[3,2,1]]
                
        elif self.imagery == 'S1' and self.selection == 'all': # 2 or 3 bands image
            image = image_loader(self.images[index]).transpose(1,2,0)
        elif self.imagery == 'S1' and self.selection == 'rgb': # 3 bands image
            image = image_loader(self.images[index]).transpose(1,2,0)
            if image.shape[2] == 2: # some images are already rgb
                image = np.dstack((image[:,:,0], image[:,:,1], image[:,:,0]/image[:,:,1]))

        image = (image - image.min()) / (image.max() - image.min() + 1e-8)
            
        mask = image_loader(self.masks[index])
        masks = [(mask == v) for v, label in enumerate(self.labels)]
        mask = np.stack(masks, axis=-1).astype("float32")

        return self.apply_transformations(image, mask)

    def load_dataset(self, source_dir, annot_dir, csv_file):
        if not self.labels:
            raise ValueError("You need to provide the list of labels for the dataset")
        
        ids = []

        if csv_file == None:
            ids = os.listdir(source_dir)
            self.images = [os.path.join(source_dir, img_id) for img_id in ids]
            if self.imagery == 'S1' and self.annotations == 'hand labeled':
                self.masks = [os.path.join(annot_dir, img_id[: img_id.rfind('_S1Hand')]+'_LabelHand.tif') for img_id in ids]
            elif self.imagery == 'S2' and self.annotations == 'hand labeled':
                self.masks = [os.path.join(annot_dir, img_id[: img_id.rfind('_S2Hand')]+'_LabelHand.tif') for img_id in ids]
            elif self.imagery == 'S1' and self.annotations == 'weakly labeled':
                self.masks = [os.path.join(annot_dir, img_id[: img_id.rfind('_S1Weak')]+'_S1OtsuLabelWeak.tif') for img_id in ids]
            elif self.imagery == 'S2' and self.annotations == 'weakly labeled':
                self.masks = [os.path.join(annot_dir, img_id[: img_id.rfind('_S2Weak')]+'_S2IndexLabelWeak.tif') for img_id in ids]

        elif csv_file: # split csv files with a list of images and corresponding masks, only available for S1 and hand labeled annotations
            self.images = []
            self.masks = []
            with open(csv_file, mode ='r')as file:
                csvFile = csv.reader(file)
                for lines in csvFile:
                    self.images.append(os.path.join(source_dir, lines[0]))
                    self.masks.append(os.path.join(annot_dir, lines[1]))

        return self.images, self.masks

    def __len__(self):
        return len(self.images)
    
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

    def data_distribution_table(self):
        label_dist = {key: 0 for key in self.labels}
        for i in np.arange(len(self.images)): 
            mask = self[i][1]
            for index, label in enumerate(self.labels):
                label_dist[self.labels[index]] += mask[:, :, index].sum()
        label_count = pd.DataFrame.from_dict(label_dist, orient='index')
        label_count.columns = ["Number of pixels"]
        label_count = label_count.astype(float)
        return label_count

    def data_distribution_barchart(self, show_title=True):
        label_count = self.data_distribution_table()
        fig, ax = plt.subplots(figsize=(12, 12))
        sns.barplot(data=label_count, x=label_count.index, y='Number of pixels', ax=ax)
        if show_title:
            ax.set_title(
                "Labels distribution for {}".format(self.get_name()), pad=20, fontsize=18
            )
        return fig
    
    def show_image(self, index, show_title=False):
        image, mask = self[index]
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
        fig.legend(handles=legend_elements, bbox_to_anchor=(0.1, 0.8, 0.8, 0.1), ncol=3, mode='expand',
                   loc='lower left', prop={'size': 12})
        plt.subplot(1, 2, 1)
        plt.imshow(image)
        plt.axis("off")
        plt.subplot(1, 2, 2)
        plt.imshow(img_mask)
        plt.axis("off")
        fig.tight_layout()
        plt.show()
        return fig