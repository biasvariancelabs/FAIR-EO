import numpy as np
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import csv

from aitlas.datasets.semantic_segmentation import SemanticSegmentationDataset
from aitlas.utils import image_loader
from .schemas import FAIREOSchema

"""
This dataset contains 3392 satellite images of Hudson Bay sea ice in the Canadian Arctic. Images are 
from Sentinel-2 satellites and are made up of bands 3, 4, and 8 (false color). Each image has an 
associated mask showing the concentration of sea ice throughout the image.
"""


class ArcticSeaIceImageMaskingDataset(SemanticSegmentationDataset):
    url = "https://www.kaggle.com/datasets/alexandersylvester/arctic-sea-ice-image-masking"

    labels = ["Undertermined/Unknown","Less than 1/10 (open water)","Bergy water","1/10","2/10","3/10","4/10","5/10",
              "6/10","7/10","8/10","9/10","10/10","9/10-10/10"]
    color_mapping = [[0,0,0],[200,0,0],[0,200,0],[150,250,0],[150,200,150],[200,0,200],[150,0,250],[150,150,250],
                     [200,150,200],[250,200,0],[200,200,0],[0,0,200],[250,0,150],[0,150,200]] 
    name = "Arctic Sea Ice Image Masking Dataset"
    schema = FAIREOSchema

    def __init__(self, config):
        # now call the constructor to validate the schema and split the data
        super().__init__(config)
        self.data_dir = self.config.data_dir
        self.selection = self.config.selection
        self.csv_file = self.config.csv_file


    def __getitem__(self, index):
        image = image_loader(self.images[index])
        mask = image_loader(self.masks[index])
        single_band_mask = np.zeros([mask.shape[0], mask.shape[1]], np.uint8)
        
        for i in np.arange(mask.shape[0]):
            for j in  np.arange(mask.shape[1]):
                if mask[i,j] == 100:
                    single_band_mask[i,j] = 0
                if mask[i,j] == 1:
                    single_band_mask[i,j] = 1
                if mask[i,j] == 2:
                    single_band_mask[i,j] = 2
                if mask[i,j] == 10:
                    single_band_mask[i,j] = 3
                if mask[i,j] == 20:
                    single_band_mask[i,j] = 4
                if mask[i,j] == 30:
                    single_band_mask[i,j] = 5
                if mask[i,j] == 40:
                    single_band_mask[i,j] = 6
                if mask[i,j] == 50:
                    single_band_mask[i,j] = 7
                if mask[i,j] == 60:
                    single_band_mask[i,j] = 8
                if mask[i,j] == 70:
                    single_band_mask[i,j] = 9
                if mask[i,j] == 80:
                    single_band_mask[i,j] = 10
                if mask[i,j] == 90:
                    single_band_mask[i,j] = 11
                if mask[i,j] == 92:
                    single_band_mask[i,j] = 12
                if mask[i,j] == 91:
                    single_band_mask[i,j] = 13

        masks = [(single_band_mask == v) for v, label in enumerate(self.labels)]
        mask = np.stack(masks, axis=-1).astype("float32")

        return self.apply_transformations(image, mask)

    def load_dataset(self, data_dir, csv_file):
        if not self.labels:
            raise ValueError("You need to provide the list of labels for the dataset")

        ids = os.listdir(os.path.join(data_dir, "Images"))
        
        split_list = []
        ids_final = []
        if csv_file: 
            with open(csv_file, mode ='r')as file:
                csvFile = csv.reader(file)
                for lines in csvFile:
                    split_list.append(lines[0])
            for i in ids:
                i_csv = 'Images/'+ i
                if i_csv in split_list:
                    ids_final.append(i)
            ids = ids_final

        self.images = [os.path.join(data_dir, "Images", image_id) for image_id in ids]
        self.masks = [os.path.join(data_dir, "Masks", image_id[: image_id.rfind('.jpg')]+'-mask.png') for image_id in ids]

    def data_distribution_table(self):
        label_dist = {key: 0 for key in self.labels}
        for image, mask in self.dataloader():
            for index, label in enumerate(self.labels):
                label_dist[self.labels[index]] += mask[:, :, :, index].sum()
        label_count = pd.DataFrame.from_dict(label_dist, orient='index')
        label_count.columns = ["Number of pixels"]
        label_count = label_count.astype(float)
        return label_count
    
    def data_distribution_barchart(self, show_title=True):
        label_count = self.data_distribution_table()
        fig, ax = plt.subplots(figsize=(12, 12))
        sns.barplot(data=label_count, x=label_count.index, y='Number of pixels', ax=ax)
        fig.autofmt_xdate()
        if show_title:
            ax.set_title(
                "Labels distribution for {}".format(self.get_name()), pad=20, fontsize=18
            )
        return fig