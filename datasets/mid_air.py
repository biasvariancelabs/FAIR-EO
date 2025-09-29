import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import csv
import seaborn as sns

from .semantic_segmentation import SemanticSegmentationDataset
from .schemas import FAIREOSchema
from ..utils import pil_loader, tiff_loader

"""
Mid-Air (Montefiore Institute Dataset of Aerial Images and Records) is a multi-purpose synthetic dataset 
for low altitude drone flights. It provides a large amount of synchronized data corresponding to flight 
records for multi-modal vision sensors and navigation sensors. More than 420k frames are simulated in 
various climate conditions.
"""

def image_loader(file_path, convert_to_grayscale=False):
    """
    Opens an image from disk

    :param file_path: path to the image
    :type file_path: str
    :param convert_to_grayscale: whether to convert the image to grayscale
    :type convert_to_grayscale: bool

    """
    filename, file_extension = os.path.splitext(file_path)
    if file_extension in [".jpg", ".png", ".bmp", ".jpeg",".JPG",".JPEG",".PNG"]:
        return pil_loader(file_path, convert_to_grayscale)
    elif file_extension in [".tif", ".tiff",".TIF"]:
        return tiff_loader(file_path)
    else:
        raise ValueError(
            "Invalid image. It should be `.jpg, .png, .bmp, .tif, .tiff, .jpeg`"
        )
    

class MidAirDataset(SemanticSegmentationDataset):
    
    #zip files need to be excrated beforehand
    url = ["https://midair.ulg.ac.be/index.html"]
    labels = [
        "background",
        "Animals",
        "Trees",
        "Dirt ground",
        "Ground vegetation",
        "Rocky ground",
        "Boulders",
        "empty",
        "Water plane",
        "Man-made construction",
        "Road", "Train track",
        "Road sign",
        "Other man-made objects"
    ]
    color_mapping = [
        [0, 0, 0],
        [255, 0, 0],
        [0, 255, 0],
        [139, 69, 19],
        [0, 128, 0],
        [128, 128, 128],
        [64, 64, 64],
        [255, 255, 255],
        [0, 0, 255],
        [255, 165, 0],
        [128, 0, 128],
        [255, 255, 0],
        [255, 0, 255],
        [0, 255, 255]
    ]
    name = "Mid-Air Dataset"
    schema = FAIREOSchema

    def __init__(self, config):
        # now call the constructor to validate the schema and split the data
        super().__init__(config)
        self.data_dir = self.config.data_dir
        self.csv_file = self.config.csv_file
        self.load_dataset(self.data_dir, self.csv_file)

    def __getitem__(self, index):

        image = image_loader(self.images[index])
        mask = image_loader(self.masks[index])

        # Convert 2D mask with class IDs to one-hot encoded 3D mask
        masks = [(mask == v) for v in range(len(self.labels))]
        mask = np.stack(masks, axis=-1).astype("float32")

        return self.apply_transformations(image, mask)

    def load_dataset(self, data_dir, csv_file):
        
        ids = []
        for i in os.listdir(data_dir):
            weather = os.path.join(data_dir, i)
            for j in os.listdir(weather):
                traj = os.path.join(weather, j)
                for k in os.listdir(traj):
                    ids.append(i+'/'+j+'/'+k)

        split_list = []
        if csv_file: 
            with open(csv_file, mode ='r')as file:
                csvFile = csv.reader(file)
                for lines in csvFile:
                    weather = lines[1]
                    split_list.append(weather+'/'+lines[0])     
            ids = split_list

        self.images = [os.path.join(data_dir, image_id) for image_id in ids]
        self.masks = [os.path.join(data_dir[: data_dir.rfind("images")], "annotations", image_id[: image_id.rfind('.JPEG')]+'.PNG') for image_id in ids]


    def __len__(self):
        return len(self.images)

    def data_distribution_table(self):
        label_dist = {key: 0 for key in self.labels}
        for image, mask in self.dataloader():
            for index, label in enumerate(self.labels):
                label_dist[self.labels[index]] += mask[:, :, :, index].sum().item()
        label_count = pd.DataFrame.from_dict(label_dist, orient='index')
        label_count.columns = ["Number of pixels"]
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