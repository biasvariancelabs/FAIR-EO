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
Five Billion Pixels is a large-scake land cover classification dataset. 
It covers areas of more than 50,000 square kilometers in China, and contains 
more than 5 billion labeled pixels. It possesses the advantage of rich 
categories, large coverage, wide distribution, and high-spatial resolution, 
which well reflects the distributions of real-world ground objects and can 
benefit to different land cover related studies.
"""


class FiveBillionPixelsDataset(SemanticSegmentationDataset):
    url = "https://drive.google.com/drive/folders/1924VnO08Gqo3Nv7Y4KirgJ9kqqCup7f0"

    labels = ["unlabeled","industrial area","paddy field","irrigated field","dry cropland","garden land","arbor forest","shrub forest","park","natural meadow","artificial meadow","river","urban residential","lake","pond","fish pond","snow","bareland","rural residential","stadium","square","road","overpass","railway station","airport"]
    color_mapping = [[0,0,0],[200,0,0],[0,200,0],[150,250,0],[150,200,150],[200,0,200],[150,0,250],[150,150,250],[200,150,200],[250,200,0],[200,200,0],[0,0,200],[250,0,150],[0,150,200],[0,200,250],[150,200,250],[250,250,250],[200,200,200],[200,150,150],[250,200,150],[150,150,0],[250,150,150],[250,150,0],[250,200,250],[200,150,0]] 
    name = "Five Billion Pixels"
    schema = FAIREOSchema

    def __init__(self, config):
        # now call the constructor to validate the schema and split the data
        super().__init__(config)
        self.data_dir = self.config.data_dir
        self.selection = self.config.selection
        self.csv_file = self.config.csv_file


    def __getitem__(self, index):
        mask = image_loader(self.masks[index])
        masks = [(mask == v) for v, label in enumerate(self.labels)]
        mask = np.stack(masks, axis=-1).astype("float32")

        if self.selection == "rgb":
            image = image_loader(self.images[index])
            image = image[:,:,1 : 4]
            if self.transform:
                image = self.transform(image)
            if self.target_transform:
                mask = self.target_transform(mask)
            return image, mask

        if self.selection == "all":
            image = image_loader(self.images[index])

            if self.transform:
                image = self.transform(image)
            if self.target_transform:
               mask = self.target_transform(mask)

            return image, mask
    

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
        self.masks = [os.path.join(data_dir, "labels", image_id[: image_id.rfind('.tif')]+'_24label.png') for image_id in ids]

    def data_distribution_table(self):
        label_dist = {key: 0 for key in self.labels}
        for i in np.arange(len(self.images)):  #because images have different sizes
            mask = self[i][1]
            for index, label in enumerate(self.labels):
                label_dist[self.labels[index]] += mask[:, :, index].sum()
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