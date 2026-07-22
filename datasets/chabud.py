import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Patch
import fnmatch
from PIL import Image
import io

from aitlas.base import BaseDataset
from .schemas import FAIREOSchema
from aitlas.utils import image_loader


"""
ChaBuD is a dataset for Change detection for Burned area Delineation and is used for the 
ChaBuD ECML-PKDD 2023 Discovery Challenge.
"""

class ChaBuDDataset(BaseDataset):

    url = "https://huggingface.co/datasets/blanchon/ChaBuD"
    name = "ChaBuD"
    schema = FAIREOSchema
    labels = ["no change","burned area"]
    color_mapping =  [[0,0,0],[255, 255, 255]]


    def __init__(self, config):
        # now call the constructor to validate the schema and split the data
        super().__init__(config)
        self.data_dir = self.config.data_dir
        self.mode = self.config.mode
        self.selection = self.config.selection
        self.data = self.load_dataset(self.data_dir)

    def __getitem__(self, index):
        img1_bytes, img2_bytes, mask_bytes = self.data[index]

        # load image from bytes (files are in parquet format)
        pre_image = np.asarray(Image.open(io.BytesIO(img1_bytes)))
        post_image = np.asarray(Image.open(io.BytesIO(img2_bytes)))

        mask = np.asarray(Image.open(io.BytesIO(mask_bytes)))
        masks = [(mask == v) for v, label in enumerate(self.labels)]
        mask = np.stack(masks, axis=-1).astype("float32")

        if self.transform:
            pre_image = self.transform(pre_image)
            post_image = self.transform(post_image)
        if self.target_transform:
            mask = self.target_transform(mask)
        
        return (pre_image, post_image), mask


    def load_dataset(self, data_dir):
        files = []
        for f in os.listdir(self.data_dir):
            if fnmatch.fnmatch(f, f'{self.mode}*'): # self.mode being 'train' or 'validation'
                files.append(os.path.join(self.data_dir, f))

        # Load and combine all parquet files into a single DataFrame
        df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)

        data = []
        for _, row in df.iterrows():
            img1_bytes = row['image1']['bytes']  # raw image bytes
            img2_bytes = row['image2']['bytes']
            mask_bytes = row['mask']['bytes']
            data.append((img1_bytes, img2_bytes, mask_bytes))

        return data

    def __len__(self):
        return len(self.data)
    
    def get_labels(self):
        return self.labels

    def show_image(self, index, show_title=False):
        data = self[index]
        (pre_img, post_img), mask = data

        # Create mask visualization
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

        fig = plt.figure(figsize=(15, 6))

        # Create subplot with space for legend on top
        plt.subplots_adjust(top=0.85, bottom=0.1)

        # Add legend at the top
        fig.legend(handles=legend_elements, bbox_to_anchor=(0.1, 1.05, 0.8, 0.05), ncol=3,
                    loc='center', prop={'size': 12})

        plt.subplot(1, 3, 1)
        plt.imshow(pre_img)
        plt.title("Pre-event")
        plt.axis("off")

        plt.subplot(1, 3, 2)
        plt.imshow(post_img)
        plt.title("Post-event")
        plt.axis("off")

        plt.subplot(1, 3, 3)
        plt.imshow(img_mask)
        plt.title("Mask")
        plt.axis("off")

        fig.tight_layout()
        plt.show()
        return fig


    def data_distribution_table(self):
        label_dist = {key: 0 for key in self.labels}
        for i in np.arange(len(self.data)):
            image, mask = self.__getitem__(i)
            for index, label in enumerate(self.labels):
                label_dist[self.labels[index]] += mask[:, :, index].sum()
        label_count = pd.DataFrame.from_dict(label_dist, orient='index')
        label_count.columns = ["Number of pixels"]
        return label_count

    def data_distribution_barchart(self, show_title=True):
        label_count = self.data_distribution_table()
        fig, ax = plt.subplots(figsize=(12, 10))
        sns.barplot(data=label_count, x=label_count.index, y='Number of pixels', ax=ax)
        if show_title:
            ax.set_title(
                "Labels distribution for {}".format(self.get_name()), pad=20, fontsize=18
            )
        return fig