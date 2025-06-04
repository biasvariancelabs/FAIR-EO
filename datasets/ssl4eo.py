import numpy as np
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import csv
from matplotlib.patches import Patch
import math

from .semantic_segmentation import SemanticSegmentationDataset
from ..utils import image_loader
from .schemas import FAIREOSchema

"""
The dataset is intended to be used for evaluation of SSL techniques. It consists of 25,000 images 
with corresponding land cover classification masks.
"""


class SSL4EODataset(SemanticSegmentationDataset):
    url = "https://huggingface.co/datasets/torchgeo/ssl4eo_l_benchmark/tree/main"

    labels = ["Open water","Perennial ice/snow","Developed, open space","Developed, low intensity",
              "Developed, medium intensity","Developed, high intensity","Barren land","Deciduous forest",
              "Evergreen forest","Mixed forest","Dwarf scrub","Shrub/scrub","Grassland/herbaceous","Sedge/herbaceous",
              "Lichens","Moss","Pasture/hay","Cultivated crops","Woody wetlands","Emergent herbaceous wetlands"]
    color_mapping = [[72,109,162],[231,239,252],[225,205,206],[220,152,129],[241,1,0],[170,2,1],[179,175,164],
                     [108,169,102],[29,101,51],[189,204,147],[175,150,60],[209,187,130],[237,236,205],[208,209,129],
                     [164,204,80],[129,186,157],[221,216,62],[174,114,41],[187,215,237],[113,164,193]] 
    name = "SSL4EO-L-Benchmark"
    schema = FAIREOSchema

    def __init__(self, config):
        # now call the constructor to validate the schema and split the data
        super().__init__(config)
        self.data_dir = self.config.data_dir
        self.selection = self.config.selection
        self.csv_file = self.config.csv_file


    def __getitem__(self, index):
        image = image_loader(self.images[index])
        image = image[:,:,0:3]
        mask = image_loader(self.masks[index])
        single_band_mask = np.zeros([mask.shape[0], mask.shape[1]], np.uint8)
        
        for i in np.arange(mask.shape[0]):
            for j in  np.arange(mask.shape[1]):
                if mask[i,j] == 11:
                    single_band_mask[i,j] = 0
                if mask[i,j] == 12:
                    single_band_mask[i,j] = 1
                if mask[i,j] == 21:
                    single_band_mask[i,j] = 2
                if mask[i,j] == 22:
                    single_band_mask[i,j] = 3
                if mask[i,j] == 23:
                    single_band_mask[i,j] = 4
                if mask[i,j] == 24:
                    single_band_mask[i,j] = 5
                if mask[i,j] == 31:
                    single_band_mask[i,j] = 6
                if mask[i,j] == 41:
                    single_band_mask[i,j] = 7
                if mask[i,j] == 42:
                    single_band_mask[i,j] = 8
                if mask[i,j] == 43:
                    single_band_mask[i,j] = 9
                if mask[i,j] == 51:
                    single_band_mask[i,j] = 10
                if mask[i,j] == 52:
                    single_band_mask[i,j] = 11
                if mask[i,j] == 71:
                    single_band_mask[i,j] = 12
                if mask[i,j] == 72:
                    single_band_mask[i,j] = 13
                if mask[i,j] == 73:
                    single_band_mask[i,j] = 14
                if mask[i,j] == 74:
                    single_band_mask[i,j] = 15
                if mask[i,j] == 81:
                    single_band_mask[i,j] = 16
                if mask[i,j] == 82:
                    single_band_mask[i,j] = 17
                if mask[i,j] == 90:
                    single_band_mask[i,j] = 18
                if mask[i,j] == 95:
                    single_band_mask[i,j] = 19

        masks = [(single_band_mask == v) for v, label in enumerate(self.labels)]
        mask = np.stack(masks, axis=-1).astype("float32")

        return self.apply_transformations(image, mask)

    def load_dataset(self, data_dir, csv_file):
        if not self.labels:
            raise ValueError("You need to provide the list of labels for the dataset")

        ids = os.listdir(os.path.join(data_dir, "images"))
        
        split_list = []
        ids_final = []
        if csv_file: 
            with open(csv_file, mode ='r')as file:
                csvFile = csv.reader(file)
                for lines in csvFile:
                    split_list.append(lines[0])
            for i in ids:
                i_csv = 'images/'+ i
                if i_csv in split_list:
                    ids_final.append(i)
            ids = ids_final

        self.images = [os.path.join(data_dir, "images", image_id) for image_id in ids]
        self.masks = [os.path.join(data_dir, "labels", image_id[: image_id.rfind('all_bands')]+'nlcd_2019.tif') for image_id in ids]

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
    
    def show_image(self, index, show_title=False):
        img, mask = self[index]
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
        height_factor = math.ceil(len(self.labels)/3)
        if height_factor == 4:
            height_factor = 0.73
        elif height_factor == 2:
            height_factor = 0.80
        else:
            height_factor = 0.81
        fig.legend(handles=legend_elements, bbox_to_anchor=(0.05, height_factor, 0.9, 0.2), ncol=3, mode='expand',
                   loc='lower left', prop={'size': 12})
        plt.subplot(1, 2, 1)
        plt.imshow(img)
        plt.axis("off")
        plt.subplot(1, 2, 2)
        plt.imshow(img_mask)
        plt.axis("off")
        fig.tight_layout()
        plt.show()
        return fig