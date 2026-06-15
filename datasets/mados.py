import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import fnmatch
import seaborn as sns

from .semantic_segmentation import SemanticSegmentationDataset
from ..utils import image_loader
from .schemas import FAIREOSchema

"""
Marine Debris and Oil Spill (MADOS) is a marine pollution dataset based on Sentinel-2 remote sensing data, 
focusing on marine litter and oil spills. MADOS formulates a challenging semantic segmentation task using 
sparse annotations.
"""


class MADOSDataset(SemanticSegmentationDataset):
    url = "https://zenodo.org/records/10664073"

    labels = [
        "no-data","marine debris","oil spill","dense sargassum","sparse floating algae",
        "natural organic material","ship","marine water","sediment-laden water",
        "foam","turbid water","shallow water","waves and wakes","oil platform",
        "jellyfish","sea snot"
    ]
    color_mapping = [
        [0,0,0],[60,255,128],[90,255,255],[255,0,0],[0,255,0],[0,0,255],
        [255,255,0],[0,255,255],[255,0,255],[128,128,128],[255,165,0],
        [75,0,130],[255,192,203],[139,69,19],[0,128,128],[220,20,60]
    ]
    name = "MADOS"
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
        
        # this dataloader considers the 10m resolution images
        
        scenes = []
        ids = []

        if csv_file == None:
            scenes = os.listdir(data_dir)
            for scene in scenes:
                if scene != "splits":
                    for img_id in os.listdir(os.path.join(data_dir, scene, "10")):
                        if fnmatch.fnmatch(img_id, "*_rgb_*"):
                            ids.append(os.path.join(data_dir, scene, "10", img_id))

        elif csv_file: # split txt files with a list of scenes
            with open(csv_file, "r",encoding="utf-8-sig") as f:
                for line in f:
                    line = line.strip()
                    scene = "Scene_" + line.split('_')[1]
                    img_id = line.split('_')[2]
                    scenes.append((scene, img_id))
            for scene in scenes:
                ids.append(os.path.join(data_dir, scene[0], "10", f"{scene[0]}_L2R_rgb_{scene[1]}.png"))

        self.images = [img_id for img_id in ids]
        self.masks = [img_id[: img_id.rfind('_rgb_')] + '_cl_' + img_id[img_id.rfind('_rgb_') :][5 : -4] + '.tif' for img_id in ids]


    def data_distribution_table(self):
        label_dist = {key: 0 for key in self.labels}
        for i in np.arange(len(self.images)): 
            mask = self[i][1]
            for index, label in enumerate(self.labels):
                label_dist[self.labels[index]] += mask[:, :, index].sum()
        label_count = pd.DataFrame.from_dict(label_dist, orient='index')
        label_count.columns = ["Number of pixels"]
        label_count = label_count.astype(float)
        return label_count

    def data_distribution_barchart(self, show_title=True):
        label_count = self.data_distribution_table()[1:] # discard the no-data for visualization
        fig, ax = plt.subplots(figsize=(12, 12))
        sns.barplot(data=label_count, x=label_count.index, y='Number of pixels', ax=ax)
        fig.autofmt_xdate()
        if show_title:
            ax.set_title(
                "Labels distribution for {}".format(self.get_name()), pad=20, fontsize=18
            )
        return fig
    
    def show_image(self, index, show_title=False):
        image, mask = self[index]
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
        fig.legend(handles=legend_elements, bbox_to_anchor=(0.1, 0.8, 0.8, 0.1), ncol=3, mode='expand',
                   loc='lower left', prop={'size': 12})
        plt.subplot(1, 2, 1)
        plt.imshow(image)
        plt.axis("off")
        plt.subplot(1, 2, 2)
        plt.imshow(img_mask)
        plt.axis("off")
        fig.tight_layout()
        plt.show()
        return fig