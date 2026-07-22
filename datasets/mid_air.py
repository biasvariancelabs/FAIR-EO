import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import csv
import seaborn as sns
from matplotlib.patches import Patch
import math

from aitlas.datasets.semantic_segmentation import SemanticSegmentationDataset
from .schemas import FAIREOSchema
from aitlas.utils import pil_loader, tiff_loader

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
    
    # zip files need to be extracted beforehand
    # this dataloader assumes a structure such as '/MidAir/Kite_training/images/cloudy/trajectory_3000/000000.JPEG for images 
    # and /MidAir/Kite_training/annotations/cloudy/trajectory_3000/000000.PNG for annotations
    url = ["https://midair.ulg.ac.be/index.html"]
    labels = [
        "background",
        "Animals",
        "Trees",
        "Dirt ground",
        "Ground vegetation",
        "Rocky ground",
        "Boulders",
        "Empty",
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
        
        data = []
        simulation = ['Kite_training','PLE_training']
        for sim in simulation:
            for weather in os.listdir(os.path.join(data_dir, sim, 'images')):
                for traj in os.listdir(os.path.join(data_dir, sim, 'images', weather)):
                    for file_name in os.listdir(os.path.join(data_dir, sim, 'images', weather, traj)):
                        data.append((sim, weather, traj+'/'+file_name))

        # csv_files created by BVLabs provide a 60/20/20 train/val/test split stratified based on simulation environment (Kite or PLE) and season/weather
        if csv_file: 
            data = []
            with open(csv_file, mode ='r')as file:
                csvFile = csv.reader(file)
                next(csvFile, None) # skip header
                for lines in csvFile:
                    file_name = lines[0]
                    weather = lines[1]
                    simulation = lines[2]
                    data.append((simulation, weather, file_name))     

        self.images = [os.path.join(data_dir, img_data[0], "images", img_data[1], img_data[2]) for img_data in data]
        self.masks = [os.path.join(data_dir, img_data[0], "annotations", img_data[1], img_data[2][: img_data[2].rfind('.JPEG')]+'.PNG') for img_data in data]


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
        fig.legend(handles=legend_elements, bbox_to_anchor=(0.1, height_factor, 0.8, 0.2), ncol=3, mode='expand',
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