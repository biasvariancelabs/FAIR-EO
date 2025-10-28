import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch
import seaborn as sns

from ..utils import image_loader
from .schemas import FAIREOSchema
from .semantic_segmentation import SemanticSegmentationDataset


'''
PASTIS-HD is a high-resolution multimodal benchmark dataset for semantic segmentation of agricultural parcels 
from satellite image time series. This implementation uses Sentinel-2 optical bands and semantic annotations 
for compatibility with the aitlas framework.
'''

class PASTISHDDataset(SemanticSegmentationDataset):
    schema = FAIREOSchema
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
        "Fruits, vegetables, flowers",
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
        self.selection = self.config.selection

    def __getitem__(self, index):
        id = self.data[index]
        # the data contains only one SPOT image but time series for S2 and S1
        spot_image = image_loader(os.path.join(self.data_dir, "DATA_SPOT", "PASTIS_SPOT6_RVB_1M00_2019",f"SPOT6_RVB_1M00_2019_{id}.tif"))
        s2_image = np.load(os.path.join(self.data_dir, "DATA_S2", f"S2_{id}.npy"))
        s1a_image = np.load(os.path.join(self.data_dir, "DATA_S1A", f"S1A_{id}.npy"),allow_pickle=True)
        s1d_image = np.load(os.path.join(self.data_dir, "DATA_S1D", f"S1D_{id}.npy"))
        mask = np.load(os.path.join(self.data_dir, "ANNOTATIONS", f"TARGET_{id}.npy"))

        image_list = []
        image_list_s1a = []
        image_list_s1d = []
        for t in range(s2_image.shape[0]):
            if self.selection == "rgb":
                time_step_image = np.transpose(s2_image[t, [2,1,0]], (1, 2, 0))
            else:
                time_step_image = np.transpose(s2_image[t, :], (1, 2, 0))
            image_list.append(time_step_image)
        s2_image = np.array(image_list)

        for t in range(s1a_image.shape[0]): # S1 has VV, VH and VV/VH bands
            time_step_image_s1a = np.transpose(s1a_image[t, 0:3], (1, 2, 0))
            image_list_s1a.append(time_step_image_s1a)
        s1a_image = np.array(image_list_s1a)

        for t in range(s1d_image.shape[0]): # S1 has VV, VH and VV/VH bands
            time_step_image_s1d = np.transpose(s1d_image[t, 0:3], (1, 2, 0))
            image_list_s1d.append(time_step_image_s1d)
        s1d_image = np.array(image_list_s1d)

        if len(mask.shape) == 3:
            semantic_mask = mask[0]
        else:
            semantic_mask = mask

        num_classes = len(self.labels)
        mask = np.eye(num_classes)[semantic_mask]

        return self.apply_transformations(s2_image, mask)

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
        image, label = self.__getitem__(index)

        if len(image.shape) < 4:
            print("Time series visualization requires temporal data")
            return None

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

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.set_title(
            f"S2 Time series - Index {index}, Pixel ({longitude},{latitude}), Label: {crop_name}",
            fontsize=14,
        )

        band_names = ['B01', 'B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B8A', 'B09']
        colors = ['purple', 'blue', 'green', 'red', 'orange', 'brown', 'pink', 'gray', 'olive', 'cyan']

        n_bands = min(len(band_names), location.shape[1])
        for i in range(n_bands):
            ax.plot(location[:, i], label=band_names[i], color=colors[i])

        ax.legend(band_names[:n_bands])
        ax.set_xlabel("Time step")
        ax.set_ylabel("Normalized reflectance")

        return fig