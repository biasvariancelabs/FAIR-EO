import numpy as np
import os
import pandas as pd
import csv

from .semantic_segmentation import SemanticSegmentationDataset
from ..utils import image_loader
from .schemas import FAIREOSchema

"""
The dataset aims at global surface water detection in very-high-resolution (VHR) satellite imagery. 
It consists of 250 satellite images and 40.96 billion pixels labeled surface water annotations that 
are distributed globally and contain water bodies exhibiting a wide variety of types. 
The initial size of the images is 12800 x 12800. This notebook divides the images into 1280 x 1280 
patches to increase the amount of trainning samples.
"""

# Define sliding window size
PATCH_H, PATCH_W = 1280, 1280

class GLHWaterDataset(SemanticSegmentationDataset):
    url = "https://drive.google.com/drive/folders/1VACDh3aGx72hDdz7Taf7FyUrpZS31Ksx"

    labels = ["non-water","water"]
    color_mapping = [[0,0,0],[255,255,255]] 
    name = "GLH-Water"
    schema = FAIREOSchema

    def __init__(self, config):
        # now call the constructor to validate the schema and split the data
        super().__init__(config)
        self.data_dir = self.config.data_dir
        self.selection = self.config.selection
        self.csv_file = self.config.csv_file
        self.images, self.masks = self.load_dataset(self.data_dir, self.csv_file)


    def __getitem__(self, index):
        image = self.images[index]
        mask = self.masks[index]
        single_band_mask = np.zeros([mask.shape[0], mask.shape[1]], np.uint8)
        for i in np.arange(mask.shape[0]):
            for j in  np.arange(mask.shape[1]):
                if mask[i,j] == 0:
                    single_band_mask[i,j] = 0
                if mask[i,j] == 255:
                    single_band_mask[i,j] = 1

        masks = [(single_band_mask == v) for v, label in enumerate(self.labels)]
        mask = np.stack(masks, axis=-1).astype("float32")

        return self.apply_transformations(image, mask)

    def load_dataset(self, data_dir, csv_file):
        if not self.labels:
            raise ValueError("You need to provide the list of labels for the dataset")
        
        self.images = []
        self.masks = []

        ids = os.listdir(os.path.join(data_dir, "img"))

        # new 60/20/20 splits are provided by BVLabs
        if csv_file:
            ids = []
            with open(csv_file, mode ='r')as file:
                csvFile = csv.reader(file)
                for lines in csvFile:
                    ids.append(lines[0].split('/')[1])

        self.images_path = [os.path.join(data_dir, "img", image_id) for image_id in ids]
        self.masks_path = [os.path.join(data_dir, "label", image_id[: image_id.rfind('.jpg')]+'.png') for image_id in ids]

        stride = 1280
        # create patches of the original images to increase the amount of training data
        for i in np.arange(len(self.images_path)):
            image = image_loader(self.images_path[i]) 
            mask = image_loader(self.masks_path[i])
                        
            # (height, width, bands) format
            img_h, img_w = image.shape[:2]
            for y in range(0, img_h - PATCH_H + 1, stride):
                for x in range(0, img_w - PATCH_W + 1, stride):
                    if y + PATCH_H <= img_h and x + PATCH_W <= img_w:
                        self.images.append(image[y:y + PATCH_H, x:x + PATCH_W])
                        self.masks.append(mask[y:y + PATCH_H, x:x + PATCH_W])

        return self.images, self.masks

    def data_distribution_table(self):
        label_dist = {key: 0 for key in self.labels}
        for image, mask in self.dataloader():
            for index, label in enumerate(self.labels):
                label_dist[self.labels[index]] += mask[:, :, :, index].sum()
        label_count = pd.DataFrame.from_dict(label_dist, orient='index')
        label_count.columns = ["Number of pixels"]
        label_count = label_count.astype(float)
        return label_count