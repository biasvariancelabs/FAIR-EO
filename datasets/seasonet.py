import numpy as np
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import csv
import math
from matplotlib.patches import Patch

from aitlas.datasets.semantic_segmentation import SemanticSegmentationDataset
from aitlas.utils import image_loader
from .schemas import FAIREOSchema

"""
This dataset consists of 1759830 multi-spectral image patches from the Sentinel-2 mission, annotated with image- 
and pixel-level land cover and land usage labels from the German land cover model LBM-DE2018 with land cover classes 
based on the CORINE Land Cover database 2018. It includes pixel synchronous examples from each of the four seasons, 
plus an additional snowy set, spanning the time from April 2018 to February 2019.
"""


class SeasonetDataset(SemanticSegmentationDataset):
    url = "https://zenodo.org/records/6979994"

    labels = ["no-data","continuous urban fabric","discontinuous urban fabric","industrial or commercial units",
              "road and rail networks and associated land","port areas","airports","mineral extraction sites",
              "dump sites","construction sites","green urban areas","sport and leisure facilities",
              "non-irrigated arable land","vineyards","fruit trees and berry plantations","pastures","broad-leaved forest",
              "coniferous forest","mixed forest","natural grasslands","moors and heathland","transitional woodland/shrub",
              "beaches, dunes, sands","bare rock","sparsely vegetated areas","inland marshes","peat bogs","salt marshes",
              "intertidal flats","water courses","water bodies","coastal lagoons","estuaries","sea and ocean"]
    color_mapping = [[0,0,0],[129,101,255],[255,0,2],[204,77,242],[204,0,0],[129,196,255],[255,176,255],[97,0,255],
                     [97,0,72],[255,77,255],[104,103,65],[172,150,82],[255,255,175],[211,101,83],[129,129,129],
                     [230,230,77],[128,255,6],[0,166,0],[77,255,0],[159,205,88],[159,205,255],[255,144,34],[255,200,54],
                     [125,200,54],[0,103,54],[0,103,129],[0,35,63],[255,35,63],[255,173,63],[255,255,220],[129,243,239],
                     [255,173,222],[104,173,0],[104,103,0]] 
    name = "Seasonet"
    schema = FAIREOSchema

    def __init__(self, config):
        # now call the constructor to validate the schema and split the data
        super().__init__(config)
        self.data_dir = self.config.data_dir
        self.selection = self.config.selection
        self.csv_file = self.config.csv_file


    def __getitem__(self, index):
        image = image_loader(self.images[index])
        # normalize the image
        image = (image - image.min()) / (image.max() - image.min())
        mask = image_loader(self.masks[index])
        masks = [(mask == v) for v, label in enumerate(self.labels)]
        mask = np.stack(masks, axis=-1).astype("float32")

        return self.apply_transformations(image, mask)

    def load_dataset(self, data_dir, csv_file):
        if not self.labels:
            raise ValueError("You need to provide the list of labels for the dataset")

        ids = []
        split_set = []
        if csv_file: # train/val or test split
            with open(csv_file, mode='r') as file:
                reader = csv.reader(file)
                split_set = [row[0] for row in reader]

        images_dir = os.path.join(data_dir, "images")
        for season in os.listdir(images_dir):
            for img_entry in os.listdir(os.path.join(data_dir, "images", season)):
                id = f"{season}/{img_entry}"

                # Only append if in CSV or no CSV (all images)
                if not csv_file or id in split_set:
                    ids.append(id)

        self.images = [os.path.join(data_dir, "images", image_id) for image_id in ids]
        self.masks = [os.path.join(data_dir, "labels", image_id[: image_id.rfind('_10m_RGB')]+'_labels.tif') for image_id in ids]

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
        fig.legend(handles=legend_elements, bbox_to_anchor=(0, height_factor, 1, 0.2), ncol=4, mode='expand',
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