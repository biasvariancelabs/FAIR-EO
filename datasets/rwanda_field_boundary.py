import numpy as np
import os
import pandas as pd

from .semantic_segmentation import SemanticSegmentationDataset
from ..utils import image_loader
from .schemas import FAIREOSchema

"""
This dataset contains field boundaries for smallholder farms in eastern Rwanda. The dataset was created 
for the purpose of conducting the Rwanda Field boundary detection Challenge. The dataset includes rasterized 
labeled field boundaries and time series satellite imagery from Planet’s NICFI program. The paired dataset 
is provided in 256x256 chips for a total of 70 tiles covering 1532 individual fields.
"""


class RwandaFieldBoundaryDataset(SemanticSegmentationDataset):
    url = "https://source.coop/repositories/nasa/rwanda-field-boundary-competition/access"

    labels = ["no field boundary","field boundary"]
    color_mapping = [[0,0,0],[255,255,255]] 
    name = "Rwanda Field Boundary"
    schema = FAIREOSchema

    def __init__(self, config):
        # now call the constructor to validate the schema and split the data
        super().__init__(config)
        self.data_dir = self.config.data_dir
        self.selection = self.config.selection
        self.csv_file = self.config.csv_file


    def __getitem__(self, index):
        if self.selection == "rgb":
            imageB02 = image_loader(self.imagesB02[index])
            imageB03 = image_loader(self.imagesB03[index])
            imageB04 = image_loader(self.imagesB04[index])
            image = np.dstack((imageB04,imageB03,imageB02)) /1500

        elif self.selection == "all":
            imageB01 = image_loader(self.imagesB01[index])
            imageB02 = image_loader(self.imagesB02[index])
            imageB03 = image_loader(self.imagesB03[index])
            imageB04 = image_loader(self.imagesB04[index])
            image = np.dstack((imageB01,imageB02,imageB03,imageB04)) /1500

        mask = image_loader(self.masks[index])
        masks = [(mask == v) for v, label in enumerate(self.labels)]
        mask = np.stack(masks, axis=-1).astype("float32")

        return self.apply_transformations(image, mask)

    def load_dataset(self, data_dir, csv_file):
        if not self.labels:
            raise ValueError("You need to provide the list of labels for the dataset")

        ids = os.listdir(os.path.join(data_dir, "source/B01"))
        self.imagesB01 = [os.path.join(data_dir, "source/B01", image_id) for image_id in ids]
        self.imagesB02 = [os.path.join(data_dir, "source/B02", image_id[: image_id.rfind('B01')]+'B02.tif') for image_id in ids]
        self.imagesB03 = [os.path.join(data_dir, "source/B03", image_id[: image_id.rfind('B01')]+'B03.tif') for image_id in ids]
        self.imagesB04 = [os.path.join(data_dir, "source/B04", image_id[: image_id.rfind('B01')]+'B04.tif') for image_id in ids]

        self.images = [os.path.join(data_dir, "source", image_id) for image_id in ids]
        self.masks = [os.path.join(data_dir, "labels", image_id[8:10]+'.tif') for image_id in ids]

    def data_distribution_table(self):
        label_dist = {key: 0 for key in self.labels}
        for image, mask in self.dataloader():
            for index, label in enumerate(self.labels):
                label_dist[self.labels[index]] += mask[:, :, :, index].sum()
        label_count = pd.DataFrame.from_dict(label_dist, orient='index')
        label_count.columns = ["Number of pixels"]
        label_count = label_count.astype(float)
        return label_count