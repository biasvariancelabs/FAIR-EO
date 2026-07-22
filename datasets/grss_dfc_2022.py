import numpy as np
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import math
from matplotlib.patches import Patch

from aitlas.datasets.semantic_segmentation import SemanticSegmentationDataset
from aitlas.utils import image_loader
from .schemas import FAIREOSchema


"""
The 2022 IEEE GRSS Data Fusion Contest, organized by the Image Analysis and Data Fusion Technical Committee, aims to promote research on semi-supervised learning. The overall objective is to build models 
that are able to leverage a large amount of unlabelled data while only requiring a small number of annotated training samples.
"""


class DFC2022Dataset(SemanticSegmentationDataset):
    url = "https://ieee-dataport.org/competitions/data-fusion-contest-2022-dfc2022"

    labels = ["No information","Urban fabric","Industrial, commercial, public, military, private and transport units","Mine, dump and construction sites","Artificial non-agricultural vegetated areas",
                "Arable land (annual crops)","Permanent crops","Pastures","Complex and mixed cultivation patterns","Orchards at the fringe of urban classes","Forests",
                "Herbaceous vegetation associations","Open spaces with little or no vegetation","Wetlands","Water","Clouds, shadows or no data"]
    color_mapping = [[0,0,0],[255,0,0],[255,255,0],[192,192,0],[0,255,0],[125,125,125],[0,0,255],[192,0,0],[0,0,192],[255,255,192],[0,255,255],[255,0,255],[0,192,0],[0,192,192],[255,192,192],[192,255,192]] 
    name = "GRSS DFC2022"
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
        masks_stacked = [(mask == v) for v, label in enumerate(self.labels)]
        mask = np.stack(masks_stacked, axis=-1).astype("float32")

        return self.apply_transformations(image, mask)

    def load_dataset(self, data_dir, csv_file):
        if not self.labels:
            raise ValueError("You need to provide the list of labels for the dataset")
        
        data = []

        for aoi in os.listdir(data_dir):
            for ids in os.listdir(os.path.join(data_dir, aoi, "BDORTHO")):
                data.append((aoi, ids))

        self.images = [os.path.join(data_dir, image_id[0], "BDORTHO", image_id[1]) for image_id in data]
        self.masks = [os.path.join(data_dir, image_id[0], "UrbanAtlas", image_id[1][: -4] + '_UA2012.tif') for image_id in data]

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
        fig.legend(handles=legend_elements, bbox_to_anchor=(0.2, height_factor, 0.7, 0.2), ncol=1, mode='expand',
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

    def data_distribution_table(self):
        label_dist = {key: 0 for key in self.labels}
        for i in np.arange(len(self.images)):
            image, mask = self.__getitem__(i)
            for index, label in enumerate(self.labels):
                label_dist[self.labels[index]] += mask[:, :, index].sum()
        label_count = pd.DataFrame.from_dict(label_dist, orient='index')
        label_count.columns = ["Number of pixels"]
        label_count = label_count.astype(float)
        return label_count

    def data_distribution_barchart(self, show_title=True):
        label_count = self.data_distribution_table()
        fig, ax = plt.subplots(figsize=(12, 10))
        sns.barplot(data=label_count, x=label_count.index, y='Number of pixels', ax=ax)
        fig.autofmt_xdate()
        if show_title:
            ax.set_title(
                "Labels distribution for {}".format(self.get_name()), pad=20, fontsize=18
            )
        return fig