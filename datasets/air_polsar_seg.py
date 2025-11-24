import numpy as np
import os
import pandas as pd
import csv

from .semantic_segmentation import SemanticSegmentationDataset
from ..utils import image_loader
from .schemas import FAIREOSchema

"""
The AIR-PolSAR-Seg dataset is a challenging PolSAR terrain segmentation dataset with high 
data scale and scene complexity. It contains 500 full-polarization SAR images and each image 
has 4 kinds of polarization modes (HH, HV, VH and VV). Therefore, the total number is 2000. 
These full-polarization SAR images are provided by the Gaofen-3 satellite at quad-polarized 
strip I (QPSI) mode.
"""


class AIRPolSARSegDataset(SemanticSegmentationDataset):
    url = "https://drive.google.com/drive/folders/1sGrmLmknhQ28nvasKbCqKvVj6rY2fCqL"

    labels = ["industrial","natural","land use","water","other","housing"]
    color_mapping = [[0,0,255],[0,255,0],[255,0,0],[0,255,255],[255,255,255],[255,255,0]] 
    name = "AIR-PolSAR-Seg"
    schema = FAIREOSchema

    def __init__(self, config):
        # now call the constructor to validate the schema and split the data
        super().__init__(config)
        self.data_dir = self.config.data_dir
        self.selection = self.config.selection
        self.csv_file = self.config.csv_file

    def __getitem__(self, index):
        mask = image_loader(self.masks[index])
        single_band_mask = np.zeros([mask.shape[0], mask.shape[1]], np.uint8)
        
        for i in np.arange(mask.shape[0]):
            for j in  np.arange(mask.shape[1]):
                if mask[i,j,0] == 0 and mask[i,j,1] == 0 and mask[i,j,2] == 255:
                    single_band_mask[i,j] = 0
                if mask[i,j,0] == 0 and mask[i,j,1] == 255 and mask[i,j,2] == 0:
                    single_band_mask[i,j] = 1
                if mask[i,j,0] == 255 and mask[i,j,1] == 0 and mask[i,j,2] == 0:
                    single_band_mask[i,j] = 2
                if mask[i,j,0] == 0 and mask[i,j,1] == 255 and mask[i,j,2] == 255:
                    single_band_mask[i,j] = 3
                if mask[i,j,0] == 255 and mask[i,j,1] == 255 and mask[i,j,2] == 255:
                    single_band_mask[i,j] = 4
                if mask[i,j,0] == 255 and mask[i,j,1] == 255 and mask[i,j,2] == 0:
                    single_band_mask[i,j] = 5
        
        masks = [(single_band_mask == v) for v, label in enumerate(self.labels)]
        mask = np.stack(masks, axis=-1).astype("float32")

        if self.selection == "rgb":
            imageHH = image_loader(self.imagesHH[index])
            imageHV = image_loader(self.imagesHV[index])
            imageHH_HV = (imageHH/imageHV).astype("uint16")
            image = np.dstack((imageHH, imageHV, imageHH_HV))

        elif self.selection == "all bands":
            imageHH = image_loader(self.imagesHH[index])
            imageHV = image_loader(self.imagesHV[index])
            imageVH = image_loader(self.imagesVH[index])
            imageVV = image_loader(self.imagesVV[index])
            image = np.dstack((imageHH, imageHV, imageVH, imageVV))

        else: # HH images
            image = image_loader(self.images[index])

        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            mask = self.target_transform(mask)

        return image, mask

    def load_dataset(self, data_dir, csv_file):
        if not self.labels:
            raise ValueError("You need to provide the list of labels for the dataset")
        
        # the dataset originally comes with a 70/30 train/test split
        # csv_files were created by BVLabs with a 60/20/20 train/val/test split
        if csv_file:
            ids = []
            with open(csv_file, "r", encoding="utf-8-sig") as f:
                reader = csv.reader(f)
                next(reader, None) # skip header
                for row in reader:
                    img_id = row[0] 
                    original_split = row[1] 
                    ids.append((img_id, original_split))
            self.images = [os.path.join(data_dir, f'{image_id[1]}_set', "images/HH", f'{image_id[1]}_{image_id[0]}_HH.tiff') for image_id in ids]
            self.imagesHH = [os.path.join(data_dir, f'{image_id[1]}_set', "images/HH", f'{image_id[1]}_{image_id[0]}_HH.tiff') for image_id in ids]
            self.imagesHV = [os.path.join(data_dir, f'{image_id[1]}_set', "images/HV", f'{image_id[1]}_{image_id[0]}_HV.tiff') for image_id in ids]
            self.imagesVH = [os.path.join(data_dir, f'{image_id[1]}_set', "images/VH", f'{image_id[1]}_{image_id[0]}_VH.tiff') for image_id in ids]
            self.imagesVV = [os.path.join(data_dir, f'{image_id[1]}_set', "images/VV", f'{image_id[1]}_{image_id[0]}_VV.tiff') for image_id in ids]
            self.masks = [os.path.join(data_dir, f'{image_id[1]}_set', "labels", f'{image_id[1]}_{image_id[0]}_gt.png') for image_id in ids]

        else: # original split
            ids = os.listdir(os.path.join(data_dir, "images/HH"))
            self.images = [os.path.join(data_dir, "images/HH", image_id) for image_id in ids]
            self.imagesHH = [os.path.join(data_dir, "images/HH", image_id) for image_id in ids]
            self.imagesHV = [os.path.join(data_dir, "images/HV", image_id[: image_id.rfind('HH')]+'HV.tiff') for image_id in ids]
            self.imagesVH = [os.path.join(data_dir, "images/VH", image_id[: image_id.rfind('HH')]+'VH.tiff') for image_id in ids]
            self.imagesVV = [os.path.join(data_dir, "images/VV", image_id[: image_id.rfind('HH')]+'VV.tiff') for image_id in ids]
            self.masks = [os.path.join(data_dir, "labels", image_id[: image_id.rfind('HH')]+'gt.png') for image_id in ids]

    def data_distribution_table(self):
        label_dist = {key: 0 for key in self.labels}
        for image, mask in self.dataloader():
            for index, label in enumerate(self.labels):
                label_dist[self.labels[index]] += mask[:, :, :, index].sum()
        label_count = pd.DataFrame.from_dict(label_dist, orient='index')
        label_count.columns = ["Number of pixels"]
        label_count = label_count.astype(float)
        return label_count