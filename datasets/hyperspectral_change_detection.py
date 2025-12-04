import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Patch
import scipy.io

from .semantic_segmentation import BaseDataset
from .schemas import FAIREOSchema
from ..utils import image_loader


"""
This dataset can be used to perform change detection techniques in multitemporal hyperspectral images. 
It includes two different hyperspectral scenes from the AVIRIS sensor: Santa Barbara scene, whose spatial 
dimensions are 984 x 740 pixels, spatial resolution 15.2 meters per pixel and includes 224 spectral bands. 
Bay Area scene, whose spatial dimensions are 600 x 500 pixels, spatial resolution 16.9 meters per pixel and 
includes 224 spectral bands. It also includes a hyperspectral scene from the HYPERION sensor: Hermiston city 
scene, whose spatial dimensions are 390 x 200 pixels and includes 242 spectral bands. 5 types of changes 
related with crop transitions are identified in this scene.
"""

class HyperspectralChangeDetectionDataset(BaseDataset):

    url = "https://gitlab.citius.usc.es/hiperespectral/ChangeDetectionDataset"
    name = "Hyperspectral Change Detection"
    schema = FAIREOSchema
    labels = ["no change","type 1","type 2","type 3","type 4","type 5"]
    color_mapping =  [[0,0,0],[0,255,255],[255,0,255],[0,0,255],[255,255,0],[192,192,192]]


    def __init__(self, config):
        # now call the constructor to validate the schema and split the data
        super().__init__(config)
        self.data_dir = self.config.data_dir
        self.mode = self.config.mode
        self.selection = self.config.selection
        self.data = self.load_dataset(self.data_dir)

    def __getitem__(self, index):
        if self.selection == 'rgb': # choose 3 out of the 224 bands
            pre_image = self.data[index][0][0][:,:,[0,100,200]]
            post_image = self.data[index][0][1][:,:,[0,100,200]]
        elif self.selection == 'all':
            pre_image = self.data[index][0][0]
            post_image = self.data[index][0][1]

        # normalize images for better visualization
        pre_img_min, pre_img_max = pre_image.min(), pre_image.max()
        pre_image = (pre_image - pre_img_min) / (pre_img_max - pre_img_min + 1e-8)

        post_img_min, post_img_max = post_image.min(), post_image.max()
        post_image = (post_image - post_img_min) / (post_img_max - post_img_min + 1e-8)

        mask = self.data[index][1]
        masks = [(mask == v) for v, label in enumerate(self.labels)]
        mask = np.stack(masks, axis=-1).astype("float32")

        if self.transform:
            pre_image = self.transform(pre_image)
            post_image = self.transform(post_image)
        if self.target_transform:
            mask = self.target_transform(mask)
        
        return (pre_image, post_image), mask


    def load_dataset(self, data_dir):
        self.data = []

        # Aviris sensor AOIs 
        aviris_aois = ['bayArea','santaBarbara']
        for aoi in aviris_aois:
            im1_mat_file = os.path.join(data_dir, aoi, 'mat', f'{aoi}_2013.mat')
            mat_data = scipy.io.loadmat(im1_mat_file)
            im1 = mat_data['HypeRvieW'] # shape: (600, 500, 224) = (H, W, bands)

            if aoi == 'bayArea':
                im2_mat_file = os.path.join(data_dir, aoi, 'mat', f'{aoi}_2015.mat')
                gt_mat_file = os.path.join(data_dir, aoi, 'mat', f'{aoi}_gtChanges2.mat.mat')
            elif aoi == 'santaBarbara':
                im2_mat_file = os.path.join(data_dir, aoi, 'mat', f'{aoi}_2014.mat')
                gt_mat_file = os.path.join(data_dir, aoi, 'mat', f'{aoi}_gtChanges.mat')
            im2_mat_data = scipy.io.loadmat(im2_mat_file)
            im2 = im2_mat_data['HypeRvieW']

            gt_mat_data = scipy.io.loadmat(gt_mat_file)
            gt = gt_mat_data['HypeRvieW']

            self.data.append(((im1, im2), gt))

        # Hyperion sensor AOI
        im1_mat_file = os.path.join(data_dir, 'Hermiston', 'hermiston2004.mat')
        mat_data = scipy.io.loadmat(im1_mat_file)
        im1 = mat_data['HypeRvieW'] 

        im2_mat_file = os.path.join(data_dir, 'Hermiston', 'hermiston2007.mat')
        mat_data = scipy.io.loadmat(im2_mat_file)
        im2 = mat_data['HypeRvieW']

        gt_mat_file = os.path.join(data_dir, 'Hermiston', 'rdChangesHermiston_5classes.mat')
        mat_data = scipy.io.loadmat(gt_mat_file)
        gt = mat_data['gt5clasesHermiston']

        self.data.append(((im1, im2), gt))

        return self.data

    def __len__(self):
        return len(self.data)
    
    def get_labels(self):
        return self.labels

    def show_image(self, index, show_title=False):
        data = self[index]
        (pre_img, post_img), mask = data

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
        plt.subplots_adjust(top=0.85, bottom=0.1)

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