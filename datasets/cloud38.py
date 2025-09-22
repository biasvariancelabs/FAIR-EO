import numpy as np
import os
import pandas as pd

from .semantic_segmentation import SemanticSegmentationDataset
from ..utils import image_loader
from .schemas import FAIREOSchema

"""
This dataset contains 38 Landsat 8 scene images and their manually extracted pixel-level ground truths 
for cloud detection. The entire images of these scenes are cropped into multiple 384x384 patches to be 
proper for deep learning-based semantic segmentation algorithms. There are 8400 patches for training 
and 9201 patches for testing.
"""

class Cloud38Dataset(SemanticSegmentationDataset):
    url = "https://www.kaggle.com/datasets/sorour/38cloud-cloud-segmentation-in-satellite-images"

    labels = ["no-cloud","cloud"]
    color_mapping = [[0,0,0],[255,255,255]] 
    name = "Cloud-38"
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
                if mask[i,j] == 0:
                    single_band_mask[i,j] = 0
                if mask[i,j] == 255:
                    single_band_mask[i,j] = 1
        
        masks = [(single_band_mask == v) for v, label in enumerate(self.labels)]
        mask = np.stack(masks, axis=-1).astype("float32")

        if self.selection == "rgb":
            imageB01 = image_loader(self.imagesB01[index])
            imageB02 = image_loader(self.imagesB02[index])
            imageB03 = image_loader(self.imagesB03[index])

            image = np.dstack((imageB01,imageB02,imageB03))/25000 

            if self.transform:
                image = self.transform(image)
            if self.target_transform:
                mask = self.target_transform(mask)
            return image, mask

        elif self.selection == "all":
            imageB01 = image_loader(self.imagesB01[index])
            imageB02 = image_loader(self.imagesB02[index])
            imageB03 = image_loader(self.imagesB03[index])
            imageB04 = image_loader(self.imagesB04[index])
            
            image = np.dstack((imageB01,imageB02,imageB03,imageB04))

            if self.transform:
                image = self.transform(image)
            if self.target_transform:
               mask = self.target_transform(mask)

            return image, mask

    def load_dataset(self, data_dir, csv_file):
        if not self.labels:
            raise ValueError("You need to provide the list of labels for the dataset")

        ids = os.listdir(os.path.join(data_dir, "images/red"))

        self.imagesB01 = [os.path.join(data_dir, "images/red", image_id) for image_id in ids]
        self.imagesB02 = [os.path.join(data_dir, "images/green", image_id[: image_id.rfind('red')] + 'green'+image_id[image_id.rfind('_patch') :]) for image_id in ids]
        self.imagesB03 = [os.path.join(data_dir, "images/blue", image_id[: image_id.rfind('red')] + 'blue'+image_id[image_id.rfind('_patch') :]) for image_id in ids]
        self.imagesB04 = [os.path.join(data_dir, "images/nir", image_id[: image_id.rfind('red')] + 'nir'+image_id[image_id.rfind('_patch') :]) for image_id in ids]
        self.masks = [os.path.join(data_dir, "gt", image_id[: image_id.rfind('red')] + 'gt'+image_id[image_id.rfind('_patch') :]) for image_id in ids]

    def __len__(self):
        return len(self.masks)
    
    def data_distribution_table(self):
        label_dist = {key: 0 for key in self.labels}
        for image, mask in self.dataloader():
            for index, label in enumerate(self.labels):
                label_dist[self.labels[index]] += mask[:, :, :, index].sum()
        label_count = pd.DataFrame.from_dict(label_dist, orient='index')
        label_count.columns = ["Number of pixels"]
        label_count = label_count.astype(float)
        return label_count