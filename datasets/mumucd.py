import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
import h5py
from matplotlib.patches import Patch

from ..base import BaseDataset
from .schemas import MSBCSchema
from ..utils import image_loader


"""
The MUlti-modal MUlti-class Change Detection (MUMUCD) dataset provides bitemporal acquisitions obtained by different satellites. 
MUMUCD includes hyperspectral (visible and shortwave infrared) by the PRISMA satellite, multispectral (visible and shortwave infrared) 
by Sentinel-2 satellites and synthetic aperture radar magnitude information by Sentinel-1 satellites. Additional layers include 
bitemporal land cover data from the Dynamic World project and static digital elevation model data retrieved via Google Earth Engine, 
and derived binary land cover change maps.
"""

class MUMUCDDataset(BaseDataset):

    url = "https://zenodo.org/records/10674011"
    name = "MUMUCD"
    schema = MSBCSchema
    labels = ["no change","change"]
    color_mapping =  [[0,0,0],[255, 255, 255]]


    def __init__(self, config):
        # now call the constructor to validate the schema and split the data
        super().__init__(config)
        self.data_dir = self.config.data_dir
        self.mode = self.config.mode
        self.selection = self.config.selection
        self.imagery = self.config.imagery
        self.data = self.load_dataset(self.data_dir)

    def __getitem__(self, index):
        img_id = self.data[index]

        s1_before = h5py.File(os.path.join(self.data_dir, img_id, f'{img_id}-before-s1.nc'),"r")['S1_data'][:] # (1536, 1536, 3)
        s1_after = h5py.File(os.path.join(self.data_dir, img_id, f'{img_id}-after-s1.nc'),"r")['S1_data'][:]

        s2_before = h5py.File(os.path.join(self.data_dir, img_id, f'{img_id}-before-s2.nc'),"r")['sr'][:] # (1536, 1536, 12)
        s2_after = h5py.File(os.path.join(self.data_dir, img_id, f'{img_id}-after-s2.nc'),"r")['sr'][:]

        prs_before = h5py.File(os.path.join(self.data_dir, img_id, f'{img_id}-before-prs.nc'),"r")['sr'][:] # (1536, 1536, 230)
        prs_after = h5py.File(os.path.join(self.data_dir, img_id, f'{img_id}-after-prs.nc'),"r")['sr'][:]

        # dynamic world land cover mask
        self.dw_labels = ["water","trees","grass","flooded vegetation","crops","shrub and scrub","built","bare","snow and ice","no data"]
        self.dw_color_mapping =  [[73,154,212],[61,129,75],[163,169,102],[124,136,196],[227,151,53],[223,200,106],[195,43,38],[166,153,139],[157,153,202],[22,89,97]]
        dw_before = h5py.File(os.path.join(self.data_dir, img_id, f'{img_id}-before-dw.nc'),"r")['lcc'][:] # (1536, 1536)
        dw_masks = [(dw_before == v) for v, label in enumerate(self.dw_labels)]
        dw_before = np.stack(dw_masks, axis=-1).astype("float32")
        dw_after = h5py.File(os.path.join(self.data_dir, img_id, f'{img_id}-after-dw.nc'),"r")['lcc'][:]
        dw_masks = [(dw_after == v) for v, label in enumerate(self.dw_labels)]
        dw_after = np.stack(dw_masks, axis=-1).astype("float32")

        dem = h5py.File(os.path.join(self.data_dir, img_id, f'{img_id}-static-dem.nc'),"r")['elev'][:]

        if self.imagery == 's2' and self.selection == 'rgb':
            image_before = s2_before[:,:,1:4]
            image_after = s2_after[:,:,1:4]
        elif self.imagery == 's2' and self.selection == 'all':
            image_before = s2_before
            image_after = s2_after
        elif self.imagery == 's1':
            image_before = s1_before
            image_after = s1_after
        elif self.imagery == 'all':
            image_before = (s1_before, s2_before, prs_before, dw_before, dem)
            image_after = (s1_after, s2_after, prs_after, dw_after)

        # binary change mask
        mask = h5py.File(os.path.join(self.data_dir, img_id, f'{img_id}-cd-binary.nc'),"r")['change'][:]
        masks = [(mask == v) for v, label in enumerate(self.labels)]
        mask = np.stack(masks, axis=-1).astype("float32")

        if self.transform:
            image_before = self.transform(image_before)
            image_after = self.transform(image_after)
        
        if self.target_transform:
            mask = self.target_transform(mask)
        
        return (image_before, image_after), mask


    def load_dataset(self, data_dir):
        self.data = []
        for aoi in os.listdir(os.path.join(data_dir)):
            if os.path.isdir(os.path.join(data_dir, aoi)):
                self.data.append(aoi)

        train = int((len(self.data)/100)*60)
        val = int((len(self.data)/100)*20)
        if self.mode == 'train':
            self.data = self.data[: train]
        elif self.mode == 'val':
            self.data = self.data[train : train+val]
        elif self.mode == 'test':
            self.data = self.data[train+val :]

        return self.data

    def __len__(self):
        return len(self.data)
    
    def get_labels(self):
        return self.labels

    def show_image(self, index, show_title=False):
        if self.imagery != 'all':
            print("self.imagery must be set to 'all' for image visualization")

        data = self[index]
        ((s1_before, s2_before, prs_before, dw_before, dem),(s1_after, s2_after, prs_after, dw_after)), mask = data

        # Create binary change mask visualization
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

        # Create dynamic world visualization for "before" image
        dw_before_mask = np.zeros([dw_before.shape[0], dw_before.shape[1], 3], np.uint8)
        legend_elements_2 = []
        for i, label in enumerate(self.dw_labels):
            legend_elements_2.append(
                Patch(
                    facecolor=tuple([x / 255 for x in self.dw_color_mapping[i]]),
                    label=self.dw_labels[i],
                )
            )
            dw_before_mask[np.where(dw_before[:, :, i] == 1)] = self.dw_color_mapping[i]

        # Create dynamic world visualization for "after" image
        dw_after_mask = np.zeros([dw_after.shape[0], dw_after.shape[1], 3], np.uint8)
        legend_elements_2 = []
        for i, label in enumerate(self.dw_labels):
            legend_elements_2.append(
                Patch(
                    facecolor=tuple([x / 255 for x in self.dw_color_mapping[i]]),
                    label=self.dw_labels[i],
                )
            )
            dw_after_mask[np.where(dw_after[:, :, i] == 1)] = self.dw_color_mapping[i]

        fig = plt.figure(figsize=(15, 6))
        axes = fig.subplots(2, 5)

        # Add legends at the top
        fig.legend(handles=legend_elements, bbox_to_anchor=(0.1, 1.2, 0.8, 0.05), ncol=4,
                    loc='center', prop={'size': 12})

        fig.legend(handles=legend_elements_2, bbox_to_anchor=(0.1, 1.05, 0.8, 0.05), ncol=4,
                    loc='center', prop={'size': 12})

        axes[0,0].imshow(s1_before)
        axes[0,0].set_title("S1 image t1")
        axes[0,0].axis("off")

        axes[1,0].imshow(s1_after)
        axes[1,0].set_title("S1 image t2")
        axes[1,0].axis("off")

        axes[0,1].imshow(s2_before[:,:,1:4]) # rgb
        axes[0,1].set_title("S2 image t1")
        axes[0,1].axis("off")

        axes[1,1].imshow(s2_after[:,:,1:4]) # rgb
        axes[1,1].set_title("S2 image t2")
        axes[1,1].axis("off")

        axes[0,2].imshow(prs_before[:,:,[35,24,16]]) # rgb
        axes[0,2].set_title("PRS image t1")
        axes[0,2].axis("off")

        axes[1,2].imshow(prs_after[:,:,[35,24,16]]) # rgb
        axes[1,2].set_title("PRS image t2")
        axes[1,2].axis("off")

        axes[0,3].imshow(dw_before_mask)
        axes[0,3].set_title("DW image t1")
        axes[0,3].axis("off")

        axes[1,3].imshow(dw_after_mask)
        axes[1,3].set_title("DW image t2")
        axes[1,3].axis("off")

        axes[0,4].imshow(dem)
        axes[0,4].set_title("DEM")
        axes[0,4].axis("off")

        axes[1,4].imshow(img_mask)
        axes[1,4].set_title("Mask")
        axes[1,4].axis("off")

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