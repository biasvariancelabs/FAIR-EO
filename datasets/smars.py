import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Patch

from .semantic_segmentation import BaseDataset
from .schemas import SMARSSchema
from ..utils import image_loader


"""
SMARS is a large synthetic dataset containing pairs of scenes with simulated urban changes aimed at training and validating change detection 
applications, as well as urban segmentation and building extraction tasks.
"""

PATCH_H, PATCH_W = 256, 256

class SMARSDataset(BaseDataset):

    url = "https://www.dlr.de/de/eoc/ueber-uns/institut-fuer-methodik-der-fernerkundung/photogrammetrie-und-bildanalyse/offentliche-datensatze/smars"
    name = "SMARS"
    schema = SMARSSchema
    labels = ["no change","construction","demolition"]
    color_mapping = [[0,0,0],[0,255,0],[255,0,0]]


    def __init__(self, config):
        # now call the constructor to validate the schema and split the data
        super().__init__(config)
        self.data_dir = self.config.data_dir
        self.mode = self.config.mode
        self.selection = self.config.selection
        self.imagery = self.config.imagery
        self.task_type = self.config.task_type
        self.data = self.load_dataset(self.data_dir)

    def __getitem__(self, index):
        
        pre_image = image_loader(self.data[index][0][0])
        pre_image_dem = image_loader(self.data[index][0][1])

        post_image = image_loader(self.data[index][1][0])
        post_image_dem = image_loader(self.data[index][1][1])

        # multilabel mask
        mask = image_loader(self.data[index][2][1]) 
        masks = [(mask == v) for v, label in enumerate(self.labels)]
        mask = np.stack(masks, axis=-1).astype("float32")

        # binary mask
        if self.task_type == "binary": 
            self.labels = ["non-change","change"]
            self.color_mapping = [[0,0,0],[255,255,255]]
            mask = image_loader(self.data[index][2][0]) 
            masks = [(mask == v) for v, label in enumerate(self.labels)]
            mask = np.stack(masks, axis=-1).astype("float32")

        if self.imagery == 'all':
            image1, image2 = np.dstack((pre_image, pre_image_dem)), np.dstack((post_image, post_image_dem))
            #normalize image 
            image1 = (image1 - image1.min()) / (image1.max() - image1.min() + 1e-8)
            image2 = (image2 - image2.min()) / (image2.max() - image2.min() + 1e-8)
        elif self.imagery == 'rgb':
            image1, image2 = pre_image, post_image
        elif self.imagery == 'dem':
            image1, image2 = pre_image_dem, post_image_dem

        if self.transform:
            image1 = self.transform(image1)
            image2 = self.transform(image2)
        if self.target_transform:
            mask = self.target_transform(mask)
        
        return (image1, image2), mask


    def load_dataset(self, data_dir):
        self.data = []
        for aoi in os.listdir(data_dir):
            split = self.mode

            im1_dir = os.path.join(data_dir, aoi, '50cm', 'pre', 'splitting', split) # 30cm images are also available
            img1 = [os.path.join(im1_dir, img) for img in sorted(os.listdir(im1_dir))] # rgb + dem

            im2_dir = os.path.join(data_dir, aoi, '50cm', 'post', 'splitting', split)
            img2 = [os.path.join(im2_dir, img) for img in sorted(os.listdir(im2_dir))] # rgb + dem

            ann_dir = os.path.join(data_dir, aoi, '50cm', 'change_map', split)
            ann = [os.path.join(ann_dir, img) for img in sorted(os.listdir(ann_dir))] # 2 classes + 3 classes

            self.data.append((img1, img2, ann))

        return self.data

    def __len__(self):
        return len(self.data)
    
    def get_labels(self):
        return self.labels

    def show_image(self, index, show_title=False):
        data = self[index]
        (image1, image2), mask = data

        if self.imagery == 'all':
            pre_image = image1[:,:,:3]
            pre_image_dem = image1[:,:,3:]
            post_image = image2[:,:,:3]
            post_image_dem = image2[:,:,3:]

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
            axes = fig.subplots(1, 5)

            # Add legend at the top
            fig.legend(handles=legend_elements, bbox_to_anchor=(0.1, 1.05, 0.8, 0.05), ncol=3,
                        loc='center', prop={'size': 12})

            axes[0].imshow(pre_image)
            axes[0].set_title("RGB image t1")
            axes[0].axis("off")

            axes[1].imshow(pre_image_dem)
            axes[1].set_title("DEM image t1")
            axes[1].axis("off")

            axes[2].imshow(post_image)
            axes[2].set_title("RGB image t2")
            axes[2].axis("off")

            axes[3].imshow(post_image_dem)
            axes[3].set_title("DEM image t2")
            axes[3].axis("off")

            axes[4].imshow(img_mask)
            axes[4].set_title("Mask")
            axes[4].axis("off")

        elif self.imagery != 'all':
            pre_image = image1
            post_image = image2
            
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

            axes = fig.subplots(1, 3)

            # Add legend at the top
            fig.legend(handles=legend_elements, bbox_to_anchor=(0.1, 1.05, 0.8, 0.05), ncol=3,
                        loc='center', prop={'size': 12})

            axes[0].imshow(pre_image)
            axes[0].set_title("Image t1")
            axes[0].axis("off")

            axes[1].imshow(post_image)
            axes[1].set_title("Image t2")
            axes[1].axis("off")

            axes[2].imshow(img_mask)
            axes[2].set_title("Mask")
            axes[2].axis("off")

        fig.tight_layout()
        plt.show()
        return fig


    def data_distribution_table(self):
        label_dist = {key: 0 for key in self.labels}
        for i in np.arange(int(len(self.img2)/2)):
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