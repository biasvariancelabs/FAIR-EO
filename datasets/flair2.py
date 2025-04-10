import numpy as np
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from .semantic_segmentation import SemanticSegmentationDataset
from ..utils import image_loader
from .schemas import FAIREOSchema

"""
FLAIR stands for French Land cover from Aerospace ImageRy. It is composed of over 20 billion 
annotated pixels of very high resolution aerial imagery. Furthermore, to integrate broader spatial 
ontext and temporal information, high resolution Sentinel-2 1-year time series with 10 spectral 
band are also provided.
"""


class FLAIR2Dataset(SemanticSegmentationDataset):
    url = "https://ignf.github.io/FLAIR/#FLAIR2"

    labels = ["no-data","building","pervious surface","impervious surface","bare soil","water","coniferous","deciduous","brushwood","vineyard","herbaceous vegetation","agricultural land","plowed land","swimming pool","snow","clear cut","mixed","ligneous","greenhouse","other"]
    color_mapping = [[5,0,0],[218,16,152],[146,142,122],[248,13,0],[169,114,1],[21,82,174],[25,74,38],[70,228,131],[243,166,13],[102,0,130],[85,255,0],[255,243,13],[228,223,124],[61,230,235],[255,255,255],[138,179,160],[107,113,79],[197,220,66],[153,153,255],[0,0,0]] 
    name = "FLAIR2"
    schema = FAIREOSchema

    def __init__(self, config):
        # now call the constructor to validate the schema and split the data
        super().__init__(config)
        self.data_dir = self.config.data_dir
        self.selection = self.config.selection
        self.csv_file = self.config.csv_file


    def __getitem__(self, index):
        image = image_loader(self.images[index])
        image=image[:,:,0:3]
        mask = image_loader(self.masks[index])
        masks = [(mask == v) for v, label in enumerate(self.labels)]
        mask = np.stack(masks, axis=-1).astype("float32")

        return self.apply_transformations(image, mask)

    def load_dataset(self, data_dir, csv_file):
        if not self.labels:
            raise ValueError("You need to provide the list of labels for the dataset")

        ids = os.listdir(os.path.join(data_dir, "images"))

        self.images = [os.path.join(data_dir, "images", image_id) for image_id in ids]
        self.masks = [os.path.join(data_dir, "labels", 'MSK'+image_id[image_id.rfind('_') :]) for image_id in ids]

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