import numpy as np
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import csv

from .semantic_segmentation import SemanticSegmentationDataset
from ..utils import image_loader
from .schemas import FAIREOSchema

"""
SkyScapes is an aerial image dataset with highly-accurate, fine-grained annotations for 
pixel-level semantic labeling. It was created to capture the complexity of aerial scenes 
at the level of granularity required by real-world applications.
"""


class SkyScapesDataset(SemanticSegmentationDataset):
    url = "https://www.dlr.de/en/eoc/about-us/remote-sensing-technology-institute/photogrammetry-and-image-analysis/public-datasets/dlr-skyscapes"

    labels = ["no_data","low vegetation","paved road","non-paved road","paved parking place","non-paved parking place","bike-way","sidewalk","entrance/exit",
              "danger area","lane-markings","building","car","trailer","van","truck","large truck","bus","clutter","impervious surface","tree"]
    color_mapping = [[0,0,0],[244,35,232],[102,102,156],[190,153,153],[153,153,153],[250,170,30],[152,251,152],[70,130,180],[107,142,35],[220,220,0],[255,0,0],
                     [220,20,60],[125,0,142],[170,200,40],[200,60,100],[150,18,80],[81,180,81],[190,241,21],[11,119,32],[120,240,120],[70,70,70]] 
    name = "SkyScapes"
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
        masks = [(mask == v) for v, label in enumerate(self.labels)]
        mask = np.stack(masks, axis=-1).astype("float32")

        return self.apply_transformations(image, mask)

    def load_dataset(self, data_dir, csv_file):
        if not self.labels:
            raise ValueError("You need to provide the list of labels for the dataset")

        ids = os.listdir(os.path.join(data_dir, "images"))

        self.images = [os.path.join(data_dir, "images", image_id) for image_id in ids]
        self.masks = [os.path.join(data_dir, "labels/grayscale", image_id[: image_id.rfind('.jpg')]+'.png') for image_id in ids]

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