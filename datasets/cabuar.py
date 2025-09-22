import numpy as np
import pandas as pd
import h5py

from .semantic_segmentation import SemanticSegmentationDataset
from .schemas import FAIREOSchema

"""
The CaBuAr (California Burned Areas) dataset contains pre- and post-fire images from 
Sentinel-2 satellites (L2A, 12 bands), acquired over burned regions in California. The 
ground truth vmasks are provided by the California Department of Forestry and Fire 
Protection and are aligned with the imagery.
"""

class CaBuArDataset(SemanticSegmentationDataset):

    url = ["https://huggingface.co/datasets/DarthReca/california_burned_areas"]
    labels = ["Undamaged area","Burned area"]
    color_mapping = [[128, 0, 128], [255, 255, 0]]
    name = "CaBuAr"
    schema = FAIREOSchema


    def __init__(self, config):
        # now call the constructor to validate the schema and split the data
        super().__init__(config)
        self.data_dir = self.config.data_dir
        self.mode = self.config.mode
        self.load_dataset(self.data_dir, self.mode)

    def __getitem__(self, index):
        with h5py.File(self.data_dir, 'r') as h5_file:
            key = self.data[index]
            integer_mask = h5_file[key]["mask"][:]
            image = h5_file[key]["post_fire"][:,:,[1,2,3]]
            image = image[:,:,[2,1,0]]

            image = image.astype(np.float32)
            if image.max() > 1.0:
                image = (image - image.min()) / (image.max() - image.min())

            if len(integer_mask.shape) == 3 and integer_mask.shape[0] == 1:
                integer_mask = integer_mask.squeeze(0)

            h, w = integer_mask.shape[:2]
            masks = [(integer_mask == v) for v in range(len(self.labels))]
            mask = np.stack(masks, axis=-1).astype("float32")

        return self.apply_transformations(image, mask)

    def load_dataset(self, data_dir, mode, csv_file=None):
        with h5py.File(data_dir, 'r') as h5_file:
            self.data = list(h5_file.keys())
            if mode == 'train':
                self.data = self.data[: 582]
            if mode == 'val':
                self.data = self.data[582 : 655]
            if mode == 'test':
                self.data = self.data[655 :]

    def __len__(self):
        return len(self.data)

    def data_distribution_table(self):
        label_pixel_counts = {label: 0 for label in self.labels}

        with h5py.File(self.data_dir, 'r') as h5_file:
            for key in self.data:
                mask = h5_file[key]["mask"][:]

                if len(mask.shape) == 3 and mask.shape[2] == 1:
                    mask = mask.squeeze(2)

                unique_values, counts = np.unique(mask, return_counts=True)
                for value, count in zip(unique_values, counts):
                    if value < len(self.labels):
                        label_pixel_counts[self.labels[value]] += count

        result_df = pd.DataFrame([
            {"Label": label, "Number of pixels": pixel_count}
            for label, pixel_count in label_pixel_counts.items()
            if pixel_count > 0
        ])

        return result_df