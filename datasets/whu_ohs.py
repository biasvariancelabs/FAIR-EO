import numpy as np
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from .semantic_segmentation import SemanticSegmentationDataset
from ..utils import image_loader
from .schemas import FAIREOSchema

"""
The dataset consists of about 90 million manually labeled samples of 7795 Orbita hyperspectral 
satellite (OHS) image patches from 40 Chinese locations.
"""


class WHUOHSDataset(SemanticSegmentationDataset):
    url = "http://irsip.whu.edu.cn/resources/WHU_OHS_show.php"

    labels = ["no data","paddy field","dry farm","woodland","shrubbery","sparse woodland","other forest land","high-covered grrassland","medium-covered grassland","low-covered grassland","river canal","lake","reservoir pond","beach land","shoal","urban built-up","rural settlement","other constructionh land","sand","gobi","saline-alkali soil","marshland","bare land","bare rock","ocean"]
    color_mapping = [[0,0,0],[193,208,255],[0,255,195],[39,115,0],[160,255,113],[74,230,2],[84,255,1],[118,116,4],[167,169,0],[252,255,1],[116,178,251],[1,91,233],[0,39,115],[123,142,245],[0,167,134],[110,1,0],[254,127,129],[253,193,191],[253,191,232],[254,1,195],[228,2,169],[166,1,127],[110,2,77],[253,116,219],[162,160,159]] 
    name = "WHU-OHS"
    schema = FAIREOSchema

    def __init__(self, config):
        # now call the constructor to validate the schema and split the data
        super().__init__(config)
        self.data_dir = self.config.data_dir
        self.selection = self.config.selection
        self.csv_file = self.config.csv_file


    def __getitem__(self, index):
        image = image_loader(self.images[index])
        Red = image[:,:,13]
        Green = image[:,:,6]
        Blue = image[:,:,1]
        image = np.dstack([Red,Green,Blue])/10000 #pixel values were scaled by 10000
        mask = image_loader(self.masks[index])
        masks = [(mask == v) for v, label in enumerate(self.labels)]
        mask = np.stack(masks, axis=-1).astype("float32")

        return self.apply_transformations(image, mask)

    def load_dataset(self, data_dir, csv_file):
        if not self.labels:
            raise ValueError("You need to provide the list of labels for the dataset")

        ids = os.listdir(os.path.join(data_dir, "images"))

        self.images = [os.path.join(data_dir, "images", image_id) for image_id in ids]
        self.masks = [os.path.join(data_dir, "labels", image_id) for image_id in ids]

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