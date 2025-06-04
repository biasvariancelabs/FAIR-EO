import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt

from .semantic_segmentation import SemanticSegmentationDataset
from ..utils import image_loader
from .schemas import FAIREOSchema

"""
This dataset contains crop types of agricultural fields in four states of Uttar Pradesh, Rajasthan, Odisha 
and Bihar in northern India. There are 13 different classes in the dataset. The dataset is split to train 
and test collections as part of the AgriFieldNet India Competition.
"""


class AgriFieldNetDataset(SemanticSegmentationDataset):
    url = "https://source.coop/repositories/radiantearth/agrifieldnet-competition/access"

    labels = ["no-data","wheat","mustard","lentil","no crop/fallow","green pea","sugarcane","garlic","maize","gram","coriandre","potato","bersem","rice"]
    color_mapping = [[0,0,0],[192,50,192],[255,0,0],[255,255,0],[192,192,0],[0,255,0],[125,125,125],[0,0,255],[255,255,255],[0,125,255],[125,0,125],[125,125,0],[125,255,0],[125,0,255]] 
    name = "AgriFieldNet"
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
            image = np.dstack((imageB04,imageB03,imageB02))

        elif self.selection == "all":
            imageB01 = image_loader(self.imagesB01[index])
            imageB02 = image_loader(self.imagesB02[index])
            imageB03 = image_loader(self.imagesB03[index])
            imageB04 = image_loader(self.imagesB04[index])
            imageB05 = image_loader(self.imagesB05[index])
            imageB06 = image_loader(self.imagesB06[index])
            imageB07 = image_loader(self.imagesB07[index])
            imageB08 = image_loader(self.imagesB08[index])
            imageB8A = image_loader(self.imagesB8A[index])
            imageB09 = image_loader(self.imagesB09[index])
            imageB10 = image_loader(self.imagesB10[index])
            imageB11 = image_loader(self.imagesB11[index])
            imageB12 = image_loader(self.imagesB12[index])
            
            image = np.dstack((imageB01,imageB02,imageB03,imageB04, imageB05,imageB06,imageB07,imageB08,imageB8A,imageB09,imageB10,imageB11,imageB12))

        mask = image_loader(self.masks[index])
        masks = [(mask == v) for v, label in enumerate(self.labels)]
        mask = np.stack(masks, axis=-1).astype("float32")

        return self.apply_transformations(image, mask)

    def load_dataset(self, data_dir, csv_file):
        if not self.labels:
            raise ValueError("You need to provide the list of labels for the dataset")

        ids = os.listdir(os.path.join(data_dir, "source/B01"))
        self.imagesB01 = [os.path.join(data_dir, "source/B01", image_id) for image_id in ids]
        self.imagesB02 = [os.path.join(data_dir, "source/B02", image_id[: image_id.rfind('B01')]+'B02_10m.tif') for image_id in ids]
        self.imagesB03 = [os.path.join(data_dir, "source/B03", image_id[: image_id.rfind('B01')]+'B03_10m.tif') for image_id in ids]
        self.imagesB04 = [os.path.join(data_dir, "source/B04", image_id[: image_id.rfind('B01')]+'B04_10m.tif') for image_id in ids]
        self.imagesB05 = [os.path.join(data_dir, "source/B05", image_id[: image_id.rfind('B01')]+'B05_10m.tif') for image_id in ids]
        self.imagesB06 = [os.path.join(data_dir, "source/B06", image_id[: image_id.rfind('B01')]+'B06_10m.tif') for image_id in ids]
        self.imagesB07 = [os.path.join(data_dir, "source/B07", image_id[: image_id.rfind('B01')]+'B07_10m.tif') for image_id in ids]
        self.imagesB08 = [os.path.join(data_dir, "source/B08", image_id[: image_id.rfind('B01')]+'B08_10m.tif') for image_id in ids]
        self.imagesB8A = [os.path.join(data_dir, "source/B8A", image_id[: image_id.rfind('B01')]+'B8A_10m.tif') for image_id in ids]
        self.imagesB09 = [os.path.join(data_dir, "source/B09", image_id[: image_id.rfind('B01')]+'B09_10m.tif') for image_id in ids]
        self.imagesB10 = [os.path.join(data_dir, "source/B10", image_id[: image_id.rfind('B01')]+'B10_10m.tif') for image_id in ids]
        self.imagesB11 = [os.path.join(data_dir, "source/B11", image_id[: image_id.rfind('B01')]+'B11_10m.tif') for image_id in ids]
        self.imagesB12 = [os.path.join(data_dir, "source/B12", image_id[: image_id.rfind('B01')]+'B12_10m.tif') for image_id in ids]

        self.images = [os.path.join(data_dir, "source", image_id) for image_id in ids]
        self.masks = [os.path.join(data_dir, "labels", image_id[: image_id.rfind('_source_')]+'_labels'+image_id[38 : 44]+'.tif') for image_id in ids]

    def data_distribution_table(self):
        label_dist = {key: 0 for key in self.labels}
        for image, mask in self.dataloader():
            for index, label in enumerate(self.labels):
                label_dist[self.labels[index]] += mask[:, :, :, index].sum()
        label_count = pd.DataFrame.from_dict(label_dist, orient='index')
        label_count.columns = ["Number of pixels"]
        label_count = label_count.astype(float)
        return label_count