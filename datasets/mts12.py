import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
import torch
from glob import glob
from scipy.io import loadmat
from matplotlib.patches import Patch
import math
import seaborn as sns

from aitlas.base import BaseDataset
from .schemas import MTS12Schema


"""
The Multi-Temporal Sentinel-1/2 (MTS12) Dataset is composed of optical Sentinel-2 and SAR Sentinel-1 time series from January to December 
2019 covering the whole Slovenia, an area of 20,271 square kilometers, for land cover classification. The whole dataset included 936 
patches, each with the size of 500×500 pixels and the temporal dimension of 12 (one per month), consisting of eight land-cover classes.
"""

class MTS12Dataset(BaseDataset):
    url = "https://gpcv.whu.edu.cn/data/dataset12/dataset12.html"
    labels = ["no data", "cultivated land", "forest", "grassland", "shrubland", "water", "wetland", "artificial surface", "bare land"]
    color_mapping = [(0,0,0), (255,255,0), (0,128,0), (0,255,0), (128,64,0), (0,0,255), (0,255,255), (255,0,0), (192,192,192)]
    name = "MTS12"
    schema = MTS12Schema

    def __init__(self, config):
        super().__init__(config)   

        self.data_dir = self.config.data_dir
        self.selection = self.config.selection
        self.date = int(self.config.date) if self.config.date is not None else None
        self.imagery = self.config.imagery
        self.split = self.config.split
        self.load_dataset(self.data_dir)

    def dataloader(self):
        """Create and return a dataloader for the dataset"""
        return torch.utils.data.DataLoader(
            self,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=lambda x: x,
        )

    def __getitem__(self, index):
        image_path = self.images[index]
        mask_path = self.masks[index]

        mat_data = loadmat(image_path)

        if self.imagery == "s2":
            data = np.nan_to_num(mat_data['band'])  # shape (12, H, W, 6)

            band_name_to_index = {
                "B02": 0,
                "B03": 1,
                "B04": 2,
                "B08": 3,
                "B11": 4,
                "B12": 5,
            }

            if self.selection == "rgb":
                bands_indices = [band_name_to_index[b] for b in ["B04", "B03", "B02"]]
            elif self.selection == "all":
                bands_indices = list(range(data.shape[-1])) 
            else:
                raise ValueError(f"Invalid selection: {self.selection}")
    
            selected_data = data[:, :, :, bands_indices] 
            selected_data = np.transpose(selected_data, (3, 0, 1, 2))  # (bands, time step, H, W)

            if self.date is not None:
                image = selected_data[:, self.date, :, :]  
            else:
                image = selected_data  

            image = image.astype(np.float32) 
            image = np.clip(image, 0, 1)

        elif self.imagery == "s1":
            data = np.nan_to_num(mat_data["band"])

            if self.date is not None:
                vv = data[self.date, :, :, 0]  
                vh = data[self.date, :, :, 1]  
                ratio = vv / (vh + 1e-6)
                image = np.stack([vv, vh, ratio], axis=0)
            else:
                vv = data[:, :, :, 0]  
                vh = data[:, :, :, 1] 
                ratio = vv / (vh + 1e-6)
                image = np.stack([vv, vh, ratio], axis=0)
        else:
            raise ValueError(f"Invalid imagery type '{self.imagery}'")

        label_data = loadmat(mask_path)
        mask = label_data["label"]  

        masks = [(mask == v) for v in range(len(self.labels))]  
        mask = np.stack(masks, axis=-1).astype(np.float32)
        mask = np.transpose(mask, (2, 0, 1))

        image = torch.tensor(image, dtype=torch.float32)
        mask = torch.tensor(mask, dtype=torch.float32)

        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            mask = self.target_transform(mask)

        return image, mask
    
    
    def __len__(self):
        return len(self.images)
    

    def load_dataset(self, data_dir):        
        if self.imagery not in ['s1', 's2']:
            raise ValueError(f"Invalid imagery type '{self.imagery}'. Must be 's1' or 's2'.")

        if self.split not in ['train', 'val', 'test']:
            raise ValueError(f"Invalid split '{self.split}'. Must be 'train', 'val', or 'test'.")

        image_dir = os.path.join(data_dir, self.imagery, self.split)
        label_dir = os.path.join(data_dir, "label", self.split)

        self.images = sorted(glob(os.path.join(image_dir, "*.mat")))
        self.masks = sorted(glob(os.path.join(label_dir, "*.mat")))

        if len(self.images) != len(self.masks):
            raise ValueError(f"Found {len(self.images)} images but {len(self.masks)} labels — mismatch!")


    def show_image(self, index, show_title=True):
        img, mask = self.__getitem__(index)  

        if self.date is None:
            time_dim = 1
            mid_slice = img.shape[time_dim] // 2
        else:
            mid_slice = self.date

        if self.imagery == "s2":
            if img.ndim == 4:
                image_disp = img[:, mid_slice, :, :]  # if img shape is (C, T, H, W), select the middle slice for visualisation
                title_text = "S2 Image (RGB) with middle date slice for visualisation"
            else:
                image_disp = img  # already sliced by date
                title_text = "S2 Image (RGB)"

            image_disp = np.transpose(image_disp, (1, 2, 0))  
            # normalize the image
            image_disp = (image_disp - image_disp.min()) / (image_disp.max() - image_disp.min() + 1e-8)
            image_disp = np.clip(image_disp, 0, 1)
            fig, axs = plt.subplots(1, 2, figsize=(12, 6))

            axs[0].imshow(image_disp)
            axs[0].axis("off")
            if show_title:
                axs[0].set_title(title_text)

        elif self.imagery == "s1":
            if img.ndim == 4:
                image_disp = img[:, mid_slice, :, :]  
            else:
                image_disp = img

            vv = image_disp[0]  
            vh = image_disp[1]

            fig, axs = plt.subplots(1, 3, figsize=(18, 6))

            axs[0].imshow(vv, cmap='gray')
            axs[0].axis("off")
            if show_title:
                axs[0].set_title("S1 VV band")

            axs[1].imshow(vh, cmap='gray')
            axs[1].axis("off")
            if show_title:
                axs[1].set_title("S1 VH band")

            mask_ax = axs[2]  

        else:
            raise ValueError(f"Unknown imagery type {self.imagery}")

        img_mask = np.zeros([mask.shape[1], mask.shape[2], 3], dtype=np.uint8)
        legend_elements = []

        for i, label in enumerate(self.labels):
            legend_elements.append(
                Patch(
                    facecolor=tuple([c / 255 for c in self.color_mapping[i]]),
                    label=label,
                )
            )
            img_mask[np.where(mask[i] == 1)] = self.color_mapping[i]

        # Show mask on last subplot
        if self.imagery == "s2":
            axs[1].imshow(img_mask)
            axs[1].axis("off")
            if show_title:
                axs[1].set_title("Mask")
        else:  
            mask_ax.imshow(img_mask)
            mask_ax.axis("off")
            if show_title:
                mask_ax.set_title("Mask")

        # Add legend on the figure
        height_factor = math.ceil(len(self.labels) / 3)
        if height_factor == 4:
            height_factor = 0.73
        elif height_factor == 2:
            height_factor = 0.80
        else:
            height_factor = 0.81

        fig.legend(handles=legend_elements, bbox_to_anchor=(0.2, height_factor, 0.6, 0.2), 
                   ncol=3, mode='expand', loc='lower left', prop={'size': 12})

        fig.tight_layout()
        plt.show()
        return fig


    def show_timeseries(self, index, latitude, longitude):
        image, mask = self.__getitem__(index)

        if image.ndim == 3: 
            raise ValueError("No time dimension available; set 'date=None' to get timeseries data.")

        if self.imagery == "s2":
            if self.selection == "rgb":
                bands = ["B04", "B03", "B02"]
            elif self.selection == "all":
                bands = ["B02", "B03", "B04", "B08", "B11", "B12"]
            else:
                raise ValueError(f"Invalid selection: {self.selection}")
        elif self.imagery == "s1":
            bands = ["VV", "VH", "VV/VH"] 
        else:
            raise ValueError(f"Unknown imagery type: {self.imagery}")

        timeseries = []
        for band_idx in range(image.shape[0]): 
            ts = image[band_idx, :, latitude, longitude].cpu().numpy()  # time series for this pixel and band
            timeseries.append(ts)

        timeseries = np.array(timeseries)

        mask_np = mask.cpu().numpy()
        label_vector = mask_np[:, latitude, longitude]  
        label_name = "Unknown"
        if label_vector.sum() > 0:
            label_idx = label_vector.argmax()
            label_name = self.labels[label_idx]

        fig, ax = plt.subplots(figsize=(12, 6))
        ax.set_title(f"Timeseries with index {index} from the region with pixel coordinates ({latitude}, {longitude}), with label {label_name}")

        for band_idx in range(timeseries.shape[0]):
            ax.plot(timeseries[band_idx], label=bands[band_idx])

        ax.set_ylabel("ρ")
        ax.legend(loc='upper right')
        ax.grid(True)
        plt.tight_layout()
        plt.show()

        return fig


    def data_distribution_table(self):
        label_dist = {key: 0 for key in self.labels}
        for i in range(len(self.images)):  
            _, mask = self[i]  # mask shape: [C, H, W]
            mask_np = mask.cpu().numpy() if torch.is_tensor(mask) else mask
            for idx, label in enumerate(self.labels):
                label_dist[label] += mask_np[idx].sum()
        label_count = pd.DataFrame.from_dict(label_dist, orient='index', columns=["Number of pixels"])
        label_count = label_count.astype(float)
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