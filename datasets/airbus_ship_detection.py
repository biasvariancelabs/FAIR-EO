import os
import numpy as np
import pandas as pd

from ..utils import image_loader
from .schemas import SegmentationDatasetSchema
from .semantic_segmentation import SemanticSegmentationDataset

"""
The Airbus Ship Detection Challenge dataset consists of high-resolution satellite imagery with 
annotated ship locations. Released as part of a Kaggle competition, it contains over 190,000 
satellite images (768×768 pixels) captured by Airbus from Planet and DigitalGlobe satellites.
"""


class AirbusShipDataset(SemanticSegmentationDataset):
    url = ["https://www.kaggle.com/competitions/airbus-ship-detection/overview"]
    name = "Airbus Ship Detection Challenge"
    schemas = SegmentationDatasetSchema
    labels = ["no-ship", "ship"]
    color_mapping = [[0, 0, 0], [255, 255, 255]]

    def __init__(self, config):
        # now call the constructor to validate the schema and split the data
        super().__init__(config)
        self.data_dir = self.config.data_dir
        self.csv_file = self.config.csv_file

    def __getitem__(self, index):
        img_name = self.data[index]
        image = image_loader(os.path.join(self.data_dir, img_name))
        img_h, img_w = image.shape[:2]
        mask = np.zeros((img_h * img_w), dtype=np.uint8)

        for i, id in enumerate(list(self._csv["ImageId"].values)):
            if id == img_name:
                tokens = str(self._csv["EncodedPixels"][i]).split()
                if len(tokens) > 1:
                    for i in range(0, len(tokens), 2):
                        start = int(tokens[i]) - 1
                        length = int(tokens[i + 1])
                        mask[start : start + length] = 1

        mask = mask.reshape((img_h, img_w), order="F")
        num_classes = 2
        mask = np.eye(num_classes)[mask]
        return self.apply_transformations(image, mask)

    def load_dataset(self, data_dir, csv_file=None):
        self.data = []
        if csv_file is not None:
            self._csv = pd.read_csv(csv_file)
            ids = list(self._csv["ImageId"].values)
            for i in ids:
                if os.path.exists(os.path.join(data_dir, i)):
                    self.data.append(i)
            self.data = list(set(self.data))

    def __len__(self):
        return len(self.data)

    def data_distribution_table(self):
        label_dist = {key: 0 for key in self.labels}
        img_w, img_h = 768, 768

        for img in list(self._csv["EncodedPixels"].values):
            x = 0
            if pd.notna(img):
                tokens = str(img).split()
                for i in range(1, len(tokens), 2):
                    x += int(tokens[i])
            label_dist["ship"] += x
            label_dist["no-ship"] += (img_h * img_w) - x

        label_count = pd.DataFrame.from_dict(label_dist, orient="index")
        label_count.columns = ["Number of pixels"]
        return label_count