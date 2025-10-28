import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch
import math
import seaborn as sns

from .schemas import FAIREOSchema
from .semantic_segmentation import SemanticSegmentationDataset

'''
PASTIS is a benchmark dataset for panoptic and semantic segmentation of agricultural parcels from satellite 
time series. It contains 2,433 patches within the French metropolitan territory with panoptic annotations 
(instance index + semantic label for each pixel). Each patch is a Sentinel-2 multispectral image time series 
of variable lentgh.
'''

class PASTISDataset(SemanticSegmentationDataset):
    schema = FAIREOSchema
    name = "PASTIS"
    url = "https://github.com/VSainteuf/pastis-benchmark"
    labels = [
        "Background", "Meadow", "Soft winter wheat", "Corn", "Winter barley",
        "Winter rapeseed", "Spring barley", "Sunflower", "Grapevine", "Beet",
        "Winter triticale", "Winter durum wheat", "Fruits, vegetables, flowers",
        "Potatoes", "Leguminous fodder", "Soybeans", "Orchard", "Mixed cereal",
        "Sorghum", "Void label"
    ]
    color_mapping = [
        [0, 0, 0], [174, 199, 232], [255, 127, 14], [255, 187, 120], [44, 160, 44],
        [152, 223, 138], [214, 39, 40], [255, 152, 150], [148, 103, 189],
        [197, 176, 213], [140, 86, 75], [196, 156, 148], [227, 119, 194],
        [247, 182, 210], [127, 127, 127], [199, 199, 199], [188, 189, 34],
        [219, 219, 141], [23, 190, 207], [255, 255, 255]
    ]

    def __init__(self, config):
        super().__init__(config)
        self.data_dir = self.config.data_dir
        self.mode = self.config.mode
        self.selection = self.config.selection
        self.data = self.load_dataset(self.data_dir, self.mode)

    def __getitem__(self, index):
        id = self.data[index]

        image = np.load(os.path.join(self.data_dir, "DATA_S2", "S2_"+id+".npy"))
        mask = np.load(os.path.join(self.data_dir, "ANNOTATIONS", "TARGET_"+id+".npy"))

        image_list = []
        for t in range(image.shape[0]):
            if self.selection == "rgb":
                time_step_image = np.transpose(image[t, [2,1,0]], (1, 2, 0))
            else:
                time_step_image = np.transpose(image[t, 0:10], (1, 2, 0))

            if time_step_image.max() > time_step_image.min():
                time_step_image = (time_step_image - time_step_image.min()) / (time_step_image.max() - time_step_image.min())
            
            image_list.append(time_step_image)

        # S2 time series
        image = np.array(image_list)

        semantic_mask = mask[0] # the label values (0 to 19) are in the first band of the mask
        num_classes = len(self.labels)
        mask = np.eye(num_classes)[semantic_mask]

        return self.apply_transformations(image, mask)

    def load_dataset(self, data_dir, mode, csv_file=None):
        self.data = []
        ids = list(set(os.path.splitext(f)[0] for f in os.listdir(os.path.join(data_dir, "DATA_S2"))))
        ids = [id_name.split('_')[-1] for id_name in ids if '_' in id_name]
        for name in ids:
            # check if the images have a mask
            if os.path.exists(os.path.join(data_dir,'ANNOTATIONS',f'TARGET_{name}.npy')): 
                self.data.append(name)

        train = int((len(self.data)/100)*80)
        val = int((len(self.data)/100)*10)
        if mode == 'train':
            self.data = self.data[: train]
        if mode == 'val':
            self.data = self.data[train : train+val]
        if mode == 'test':
            self.data = self.data[train+val :]

        return self.data
    
    def __len__(self):
        return len(self.data)

    def show_image(self, index, show_title=False):
        # plot the first image of the time series (date 0)
        img, mask = self.__getitem__(index)

        img = img[0] # first image
        img = img[:, :, [2, 1, 0]] # RGB bands for visualization

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

        fig.legend(handles=legend_elements, bbox_to_anchor=(0.08, height_factor, 0.85, 0.2),
                  ncol=4, mode='expand', loc='lower left', prop={'size': 12})

        plt.subplot(1, 2, 1)
        plt.imshow(img)
        plt.axis("off")
        plt.subplot(1, 2, 2)
        plt.imshow(img_mask)
        plt.axis("off")
        fig.tight_layout()
        plt.show()
        return fig

    def data_distribution_table(self):
        label_dist = {key: 0 for key in self.labels}
        for i in range(len(self)):
            _, mask = self.__getitem__(i)
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

    def show_timeseries(self, index, longitude, latitude):
        self.selection = "all"
        image, label = self.__getitem__(index)

        location = []
        for i in range(len(image)):
            pixel_values = image[i][longitude, latitude]
            location.append(pixel_values)
        location = np.array(location)

        crop = label[longitude, latitude]
        crop_name = "Unknown"
        for j, k in enumerate(self.labels):
            if crop[j] == 1:
                crop_name = k
                break

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.set_title(
            f"Time series with index {index} from pixel coordinates ({longitude},{latitude}), with label {crop_name}\n",
            fontsize=14,
        )

        band_names = ['B01', 'B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B8A', 'B09']
        colors = ['purple', 'blue', 'green', 'red', 'orange', 'brown', 'pink', 'gray', 'olive', 'cyan']

        for i in range(10):
            ax.plot(location[:, i], label=band_names[i], color=colors[i])

        ax.legend(band_names, bbox_to_anchor=(1.2,1), loc='upper right')
        ax.set_xlabel("Time step")
        ax.set_ylabel("Normalized reflectance")

        return fig