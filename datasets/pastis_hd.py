import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch
import seaborn as sns
import csv

from ..utils import image_loader
from .schemas import PASTISHDSchema
from .semantic_segmentation import SemanticSegmentationDataset


'''
PASTIS-HD is a high-resolution multimodal benchmark dataset for semantic segmentation of agricultural parcels 
from satellite image time series. This implementation uses Sentinel-2 optical bands and semantic annotations 
for compatibility with the aitlas framework.
'''

class PASTISHDDataset(SemanticSegmentationDataset):
    schema = PASTISHDSchema
    name = "PASTIS HD"
    url = "https://huggingface.co/datasets/IGNF/PASTIS-HD"
    labels = [
        "Background",
        "Meadow",
        "Soft winter wheat",
        "Corn",
        "Winter barley",
        "Winter rapeseed",
        "Spring barley",
        "Sunflower",
        "Grapevine",
        "Beet",
        "Winter triticale",
        "Winter durum wheat",
        "Fruits vegetables flowers",
        "Potatoes",
        "Leguminous fodder",
        "Soybeans",
        "Orchard",
        "Mixed cereal",
        "Sorghum",
        "Void label"
    ]
    color_mapping = [
        [0, 0, 0],
        [174, 199, 232],
        [255, 127, 14],
        [255, 187, 120],
        [44, 160, 44],
        [152, 223, 138],
        [214, 39, 40],
        [255, 152, 150],
        [148, 103, 189],
        [197, 176, 213],
        [140, 86, 75],
        [196, 156, 148],
        [227, 119, 194],
        [247, 182, 210],
        [127, 127, 127],
        [199, 199, 199],
        [188, 189, 34],
        [219, 219, 141],
        [23, 190, 207],
        [255, 255, 255]
    ]

    def __init__(self, config):
        super().__init__(config)
        self.data_dir = self.config.data_dir
        self.csv_file = self.config.csv_file
        self.selection = self.config.selection
        self.imagery = self.config.imagery

    def __getitem__(self, index):
        id = self.data[index]
        # the data contains only one SPOT image per patch but time series for S2 and S1
        spot_image = image_loader(os.path.join(self.data_dir, "DATA_SPOT", "PASTIS_SPOT6_RVB_1M00_2019",f"SPOT6_RVB_1M00_2019_{id}.tif"))
        s2_image = np.load(os.path.join(self.data_dir, "DATA_S2", f"S2_{id}.npy"))
        s1a_image = np.load(os.path.join(self.data_dir, "DATA_S1A", f"S1A_{id}.npy"),allow_pickle=True)
        s1d_image = np.load(os.path.join(self.data_dir, "DATA_S1D", f"S1D_{id}.npy"))
        mask = np.load(os.path.join(self.data_dir, "ANNOTATIONS", f"TARGET_{id}.npy"))

        image_list = []
        image_list_s1a = []
        image_list_s1d = []
        for t in range(s2_image.shape[0]): # S2 has 10 bands
            if self.selection == "rgb":
                time_step_image = np.transpose(s2_image[t, [2,1,0]], (1, 2, 0))
            else:
                time_step_image = np.transpose(s2_image[t, :], (1, 2, 0))
            image_list.append(time_step_image)
        s2 = np.array(image_list)

        for t in range(s1a_image.shape[0]): # S1 has VV, VH and VV/VH bands
            time_step_image_s1a = np.transpose(s1a_image[t, 0:3], (1, 2, 0))
            image_list_s1a.append(time_step_image_s1a)
        s1a = np.array(image_list_s1a)

        for t in range(s1d_image.shape[0]): # S1 has VV, VH and VV/VH bands
            time_step_image_s1d = np.transpose(s1d_image[t, 0:3], (1, 2, 0))
            image_list_s1d.append(time_step_image_s1d)
        s1d = np.array(image_list_s1d)

        timeseries = {
                "sentinel1a": s1a,
                "sentinel1d": s1d,
                "sentinel2": s2
            }
    
        if self.imagery == 'all':
            image = np.array([spot_image, timeseries])
        elif self.imagery == 'SPOT':
            image = spot_image
            if self.transform:
                image = self.transform(image)

        if len(mask.shape) == 3:
            semantic_mask = mask[0]
        else:
            semantic_mask = mask

        num_classes = len(self.labels)
        mask = np.eye(num_classes)[semantic_mask]
        if self.target_transform:
            mask = self.target_transform(mask)

        return image, mask

    def load_dataset(self, data_dir, csv_file=None):
        # Find IDs which have both S2 data and annotations (not all images are annotated)
        s2_dir = os.path.join(data_dir, "DATA_S2")
        annotation_dir = os.path.join(data_dir, "ANNOTATIONS")

        s2_files = os.listdir(s2_dir)
        s2_ids = [os.path.splitext(f)[0] for f in s2_files if f.endswith('.npy')]
        s2_ids = [id_name.split('_')[-1] for id_name in s2_ids if '_' in id_name]

        annotation_files = os.listdir(annotation_dir)
        annotation_ids = [os.path.splitext(f)[0] for f in annotation_files if f.endswith('.npy')]
        annotation_ids = [id_name.split('_')[-1] for id_name in annotation_ids if '_' in id_name]

        common_ids = list(set(s2_ids).intersection(set(annotation_ids)))
        self.data = common_ids

        # csv_files with a 60/20/20 train/val/test split are provided by BVLabs
        # the csv_files contain the annotation file names
        if csv_file:
            self.data = []
            with open(csv_file, "r", encoding="utf-8-sig") as f:
                        reader = csv.reader(f)
                        next(reader, None) # skip header
                        for row in reader:
                            file_name = row[0][7 : -4]  
                            self.data.append(file_name)

    def __len__(self):
        return len(self.data)

    def show_image(self, index, show_title=False):
        # Shows the first image of the time series
        id = self.data[index]
        # Check which modalities are available for this ID
        available_modalities = []
        modality_paths = {
            'S2': os.path.join(self.data_dir, "DATA_S2", f"S2_{id}.npy"),
            'S1A': os.path.join(self.data_dir, "DATA_S1A", f"S1A_{id}.npy"),
            'S1D': os.path.join(self.data_dir, "DATA_S1D", f"S1D_{id}.npy"),
            'SPOT': os.path.join(self.data_dir, "DATA_SPOT", "PASTIS_SPOT6_RVB_1M00_2019", f"SPOT6_RVB_1M00_2019_{id}.tif")
        }

        for modality, path in modality_paths.items():
            if os.path.exists(path):
                available_modalities.append((modality, path))

        self.selection = "rgb"
        _, mask = self.__getitem__(index)

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

        n_modalities = len(available_modalities)
        fig_width = 4 * (n_modalities + 1)
        fig = plt.figure(figsize=(fig_width, 6))

        fig.legend(handles=legend_elements, bbox_to_anchor=(0.1, 0.9, 0.8, 0.2),
                  ncol=min(4, len(self.labels)), mode='expand', loc='lower left', prop={'size': 12})

        for i, (modality, path) in enumerate(available_modalities):
            plt.subplot(1, n_modalities + 1, i + 1)

            if modality == 'SPOT':
                img_data = image_loader(path)
                plt.imshow(img_data)

            elif modality == 'S2':
                img_data = np.load(path)[0]
                img_display = np.transpose(img_data, (1, 2, 0))
                img_display = img_display[:,:,[2,1,0]]

                if img_display.max() > img_display.min():
                    img_display = (img_display - img_display.min()) / (img_display.max() - img_display.min())
                plt.imshow(img_display)

            elif modality in ['S1A', 'S1D']:
                img_data = np.load(path)[0]
                img_display = img_data[0]
                plt.imshow(img_display, cmap='gray')

            plt.axis("off")
            plt.title(f"{modality}")

        plt.subplot(1, n_modalities + 1, n_modalities + 1)
        plt.imshow(img_mask)
        plt.axis("off")
        plt.title("Semantic Mask")
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
        self.imgery = "all"
        
        image, mask = self[index]
        timeseries = image[1]

        crop = mask[longitude, latitude]
        crop_name = "Unknown"
        for j, k in enumerate(self.labels):
            if crop[j] == 1:
                crop_name = k
                break

        s1a = timeseries["sentinel1a"] 
        s1d = timeseries["sentinel1d"] 
        s2 = timeseries["sentinel2"] 

        time_range_s1a = range(len(s1a))
        time_range_s1d = range(len(s1d))
        time_range_s2 = range(len(s2))

        # extract S1A bands for a specific pixel
        s1a_pixel = []
        for b in range(3):
            s1a_band_series = [s1a[i][longitude, latitude, b] for i in time_range_s1a]
            s1a_pixel.append(s1a_band_series)

        # extract S1D bands for a specific pixel
        s1d_pixel = []
        for b in range(3):
            s1d_band_series = [s1d[i][longitude, latitude, b] for i in time_range_s1d]
            s1d_pixel.append(s1d_band_series)

        # extract Sentinel-2 bands
        s2_pixel = []
        for b in range(10):
            s2_band_series = [s2[i][longitude, latitude, b] for i in time_range_s2]
            s2_pixel.append(s2_band_series)

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.set_title(f"Time series with index {index} from the region with pixel coordinates ({longitude},{latitude}), with label {crop_name}\n")

        for i, s1a_band_series in enumerate(s1a_pixel):
            if i == 0:
                band = 'VV'
            elif i == 1:
                band = 'VH'
            else: 
                band = 'VV/VH'
            ax.plot(time_range_s1a, s1a_band_series, label=f"S1A {band}")

        for i, s1d_band_series in enumerate(s1d_pixel):
            if i == 0:
                band = 'VV'
            elif i == 1:
                band = 'VH'
            else: 
                band = 'VV/VH'
            ax.plot(time_range_s1d, s1d_band_series, label=f"S1D {band}")

        for i, s2_band_series in enumerate(s2_pixel):
            ax.plot(time_range_s2, s2_band_series, label=f"S2-B{i+1}", alpha=0.5)

        ax.set_xlabel("Time step")
        ax.set_ylabel("Pixel value")
        ax.legend()
        plt.tight_layout()
        plt.show()
        return fig