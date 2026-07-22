from aitlas.datasets.multiclass_classification import MultiClassClassificationDataset
from aitlas.utils import image_loader

import glob
import json
import h5py
import os
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
import random

'''
TreeSatAI-TS is a multimodal dataset for tree species identification,
containing 50,381 tiles of 60×60 m with multi-label annotations for 20 classes and
all taken in Germany.
'''

LABELS = ["Abies_alba","Acer_pseudoplatanus","Alnus_spec.","Betula_spec.","Cleared","Fagus_sylvatica",
          "Fraxinus_excelsior","Larix_decidua","Larix_kaempferi","Picea_abies","Pinus_nigra","Pinus_strobus",
          "Pinus_sylvestris","Populus_spec.","Prunus_spec.","Pseudotsuga_menziesii","Quercus_petraea",
          "Quercus_robur","Quercus_rubra","Tilia_spec."]


class TreeSatAI_TSDataset(MultiClassClassificationDataset):

    url = "https://huggingface.co/datasets/IGNF/TreeSatAI-Time-Series"
    labels = LABELS
    name = "TreeSatAI-TS dataset"

    def __init__(self, config, date=None):
        # now call the constructor to validate the schema and load the data
        super().__init__(config)
        # load the data
        self.data_dir = self.config.data_dir
        self.csv_file = self.config.csv_file
        self.data = self.load_dataset()


    def __getitem__(self, index):
        aerial_path, h5_path, label_index = self.data[index]
        # load aerial image
        aerial_img = image_loader(aerial_path)[:, :, :3]

        # load timeseries
        with h5py.File(h5_path, 'r') as f:
            # Sentinel-1 ascending and descending (each have VV and VH bands)
            s1_asc = f["sen-1-asc-data"][:]
            s1_des = f["sen-1-des-data"][:]  

            # Sentinel-2
            s2 = f["sen-2-data"][:]  
        
        timeseries = {
            "sentinel1_asc": s1_asc,
            "sentinel1_des": s1_des,
            "sentinel2": s2
        }
        
         # apply transforms to aerial image if any
        if self.transform:
            aerial_img = self.transform(aerial_img)

        target = label_index

        return aerial_img, timeseries, target


    def load_dataset(self):
        aerial_dir = os.path.join(self.data_dir, "aerial")
        ts_dir = os.path.join(self.data_dir, "sentinel-ts")
        labels_path = os.path.join(self.data_dir, "labels", "TreeSatBA_v9_60m_multi_labels.json")

        with open(labels_path, 'r') as f:
            labels_json = json.load(f)

        with open(self.csv_file, 'r') as f:
            split_filenames = set(line.strip() for line in f.readlines())

        data = []
        for fname in os.listdir(aerial_dir):
            if fname.endswith(".tif"):
                if fname not in split_filenames:
                    continue

                if fname not in labels_json:
                    continue

                label_str = "_".join(fname.split("_")[:2])
                if label_str not in self.labels:
                    continue
                label_index = self.labels.index(label_str)

                aerial_path = os.path.join(aerial_dir, fname)
                base_name = fname.replace(".tif", "")

                h5_files = glob.glob(os.path.join(ts_dir, base_name + "*.h5"))
                if h5_files:
                    h5_path = h5_files[0]
                    data.append((aerial_path, h5_path, label_index))

        return data

   
    def show_samples(self):
        trimmed_data = [
            (os.path.basename(aerial), os.path.basename(h5), self.labels[label])
            for aerial, h5, label in self.data
        ]
        df = pd.DataFrame(trimmed_data, columns=["Aerial Image", "TS File", "Label Index"])
        return df.head(20)
    

    def show_image(self, index):
        # show the first image of the time series
        aerial_img, timeseries, target = self[index]
        label_name = self.labels[target]

        fig, axs = plt.subplots(1, 4, figsize=(16, 6))

        # Aerial image
        axs[0].imshow(aerial_img)
        axs[0].set_title("Aerial Image", fontsize=16)
        axs[0].axis("off")

        # Sentinel-1 time slice or blank
        # S1 ascending (VV, VH bands)
        if "sentinel1_asc" not in timeseries:
            axs[1].axis("off")
            axs[1].set_title("No Sentinel-1 Ascending time slice", fontsize=16)
        else:
            img = timeseries["sentinel1_asc"][0] # first image of the time series
            vv_band = img[0]

            axs[1].imshow(vv_band, cmap='gray')
            axs[1].set_title(f"Sentinel-1 Ascending", fontsize=16)
            axs[1].axis("off")

        # S1 descending (VV, VH bands)
        if "sentinel1_des" not in timeseries:
            axs[2].axis("off")
            axs[2].set_title("No Sentinel-1 Descending time slice", fontsize=16)
        else:
            img = timeseries["sentinel1_des"][0]
            vv_band = img[0]

            axs[2].imshow(vv_band, cmap='gray')
            axs[2].set_title(f"Sentinel-1 Descending", fontsize=16)
            axs[2].axis("off")

        # Sentinel-2 time slice or blank
        if "sentinel2" not in timeseries:
            axs[3].axis("off")
            axs[3].set_title("No Sentinel-2 time slice", fontsize=16)
        else:
            img = timeseries["sentinel2"][0]
            try:
                rgb = np.stack([img[3], img[2], img[1]], axis=-1)  # Bands 4,3,2 for RGB
            except Exception:
                rgb = np.stack(img[:3], axis=-1)
            rgb_min, rgb_max = rgb.min(), rgb.max()
            rgb = (rgb - rgb_min) / (rgb_max - rgb_min + 1e-10)
            axs[3].imshow(rgb)
            axs[3].set_title(f"Sentinel-2", fontsize=16)
            axs[3].axis("off")

        # main title 
        fig.suptitle(
            f"Image with index {index} from the dataset {self.get_name()}, with label {label_name}\n",
            fontsize=20,
            y=1.05
        )
        plt.tight_layout()
        return fig


    def show_batch(self, size, show_title=True):
        if size % 5 != 0:
            raise ValueError("The provided size should be divisible by 5!")

        image_indices = random.sample(range(len(self.data)), size)
        rows = size // 5
        figure, ax = plt.subplots(rows, 5, figsize=(13.75, 2.8 * rows))

        if show_title:
            figure.suptitle(
                "Example aerial images with labels from {}".format(self.get_name()),
                fontsize=32,
                y=1.006,
            )

        for axes, idx in zip(ax.flatten(), image_indices):
            aerial_img, _, target = self[idx]  # unpack __getitem__ output
            axes.imshow(aerial_img)
            axes.set_title(self.labels[target], fontsize=18, pad=10)
            axes.axis('off')

        figure.tight_layout()
        return figure


    def show_timeseries(self, index, row=3, col=3):
        
        _, timeseries, label_index = self[index]
        label_name = self.labels[label_index]

        s1_asc = timeseries["sentinel1_asc"]  
        s1_des = timeseries["sentinel1_des"]  
        s2 = timeseries["sentinel2"]  

        T1_ASC = s1_asc.shape[0]
        T1_DES = s1_des.shape[0]
        T2 = s2.shape[0]
        max_T = max(T1_ASC, T1_DES, T2)
        x = range(max_T)

        # extract ascending and descending bands for a specific pixel
        s1_asc_bands = []
        s1_des_bands = []
        for b in range(2):
            s1_asc_band_series = [s1_asc[i][b, row, col] for i in range(T1_ASC)]
            if T1_ASC < max_T:
                s1_asc_band_series.extend([float('nan')] * (max_T - T1_ASC))
            s1_asc_bands.append(s1_asc_band_series)

            s1_des_band_series = [s1_des[i][b, row, col] for i in range(T1_DES)]
            if T1_DES < max_T:
                s1_des_band_series.extend([float('nan')] * (max_T - T1_DES))
            s1_des_bands.append(s1_des_band_series)

        # extract Sentinel-2 bands
        s2_bands = []
        for b in range(s2.shape[1]):
            band_series = [s2[i][b, row, col] for i in range(T2)]
            if T2 < max_T:
                band_series.extend([float('nan')] * (max_T - T2))
            s2_bands.append(band_series)

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.set_title(f"Time series with index {index} from the region with pixel coordinates ({row},{col}), with label {label_name}\n")

        for i, s1_asc_band_series in enumerate(s1_asc_bands):
            if i == 0:
                band = 'VV'
            else:
                band = 'VH'
            ax.plot(x, s1_asc_band_series, label=f"S1 ASC {band}")

        for i, s1_des_band_series in enumerate(s1_des_bands):
            if i == 0:
                band = 'VV'
            else:
                band = 'VH'
            ax.plot(x, s1_des_band_series, label=f"S1 DES {band}")

        for i, band_series in enumerate(s2_bands):
            ax.plot(x, band_series, label=f"S2-B{i+1}", alpha=0.5)

        ax.set_xlabel("Time step")
        ax.set_ylabel("Pixel value")
        ax.legend()
        plt.tight_layout()
        plt.show()
        return fig


    def data_distribution_table(self):
        with open(self.csv_file, 'r') as f:
            filenames = [line.strip() for line in f.readlines() if line.strip()]
        
        labels = []
        for fname in filenames:
            parts = fname.split('_')
            if parts[0] == 'Cleared':
                label = 'Cleared'
            else:
                label = parts[0] + '_' + parts[1]
            labels.append(label)

        df = pd.DataFrame(labels, columns=["Label"])
        label_count = df.value_counts().reset_index(name='Count')
        
        return label_count

    def data_distribution_barchart(self):
        label_count = self.data_distribution_table()
        fig, ax = plt.subplots(figsize=(12, 10))
        sns.barplot(y="Label", x="Count", data=label_count, ax=ax)
        ax.set_title(
            "Labels distribution for {}".format(self.get_name()), pad=20, fontsize=18
        )
        return fig