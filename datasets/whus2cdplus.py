import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import tifffile
from matplotlib.patches import Patch
import math

from aitlas.utils import image_loader
from aitlas.datasets.semantic_segmentation import SemanticSegmentationDataset
from .schemas import FAIREOSchema


'''
WHUS2-CD+ is a cloud validation detection dataset for Sentinel-2A images. WHUS2-CD+ contains 36 manually labeled
cloud masks at 10m resolution and corresponding Sentinel-2A images evenly distributed over China mainland.
'''
    
class WHUS2CDplusDataset(SemanticSegmentationDataset):

    url = "https://github.com/Neooolee/WHUS2-CD"

    labels = ["no data", "clear", "cloud"] 
    mask_values = [0, 128, 255]
    color_mapping = [[0, 0, 0],[0,0,255],[0,255,255]]

    name = "WHUS2CD+"
    schema = FAIREOSchema
    
    def __init__(self, config):
        # now call the constructor to validate the schema and split the data
        super().__init__(config)
        self.data_dir = self.config.data_dir
        self.selection = self.config.selection
        self.csv_file = self.config.csv_file

    def __getitem__(self, index):

        mask = image_loader(self.masks[index],False)
        masks = [(mask == v) for v in self.mask_values]
        mask = np.stack(masks, axis=-1).astype("float32")

        if self.selection == "rgb":
            b02_path, b03_path, b04_path = self.images[index]

            b02 = image_loader(b02_path)
            b03 = image_loader(b03_path)
            b04 = image_loader(b04_path)

            # Stack as RGB
            image = np.dstack((b02, b03, b04)).astype(np.float32)

            if self.transform:
                image = self.transform(image)
            if self.target_transform:
                mask = self.target_transform(mask)
            return image, mask

        elif self.selection == "all":
            bands = [
                self.imagesB01, self.imagesB02, self.imagesB03, self.imagesB04,
                self.imagesB05, self.imagesB06, self.imagesB07, self.imagesB08,
                self.imagesB8A, self.imagesB09, self.imagesB10, self.imagesB11,
                self.imagesB12
            ]
            band_arrays = []
            for band_list in bands:
                band = image_loader(band_list[index])
                band_arrays.append(band)

            image = np.dstack(band_arrays).astype(np.float32)
            image = torch.tensor(image.transpose(2, 0, 1), dtype=torch.float32) / 10000

            if self.transform:
                image = self.transform(image)
            if self.target_transform:
                mask = self.target_transform(mask)

            return image, mask
            

    def load_dataset(self, data_dir, csv_file=None):
        
        if not self.labels:
            raise ValueError("You need to provide the list of labels for the dataset")
        
        if csv_file is not None:
            df = pd.read_csv(csv_file, header=None)
            allowed_scenes = set(df[0].str.strip()) 
        else:
            allowed_scenes = None
        
        scene_dirs = [d for d in os.listdir(data_dir) if d.endswith(".SAFE")]
        
        self.images = []  # will hold (B02, B03, B04) paths as tuples
        self.imagesB02, self.imagesB03, self.imagesB04 = [], [], []
        self.imagesB01, self.imagesB05, self.imagesB06, self.imagesB07 = [], [], [], []
        self.imagesB08, self.imagesB8A, self.imagesB09, self.imagesB10 = [], [], [], []
        self.imagesB11, self.imagesB12 = [], []
        self.masks = []

        for scene in scene_dirs:
            scene_id = scene.replace(".SAFE", "")
            if allowed_scenes is not None and scene_id not in allowed_scenes:
                continue
            
            scene_path = os.path.join(data_dir, scene, scene)
            granule_path = os.path.join(scene_path, "GRANULE")
            if not os.path.exists(granule_path):
                continue

            granule_name = os.listdir(granule_path)[0]
            img_data_path = os.path.join(granule_path, granule_name, "IMG_DATA")

            def band_path(suffix):
                return next((os.path.join(img_data_path, f) for f in os.listdir(img_data_path) if f.endswith(f"_{suffix}.jp2")), None)

            b01 = band_path("B01")
            b02 = band_path("B02")
            b03 = band_path("B03")
            b04 = band_path("B04")
            b05 = band_path("B05")
            b06 = band_path("B06")
            b07 = band_path("B07")
            b08 = band_path("B08")
            b8A = band_path("B8A")
            b09 = band_path("B09")
            b10 = band_path("B10")
            b11 = band_path("B11")
            b12 = band_path("B12")

            self.imagesB01.append(b01)      
            self.imagesB02.append(b02)
            self.imagesB03.append(b03)
            self.imagesB04.append(b04)
            self.imagesB05.append(b05)
            self.imagesB06.append(b06)
            self.imagesB07.append(b07)
            self.imagesB08.append(b08)
            self.imagesB8A.append(b8A)
            self.imagesB09.append(b09)
            self.imagesB10.append(b10)
            self.imagesB11.append(b11)
            self.imagesB12.append(b12)

            # RGB image is a tuple of B02, B03, B04
            if b02 and b03 and b04:
                self.images.append((b02, b03, b04))

            mask_path = os.path.join(data_dir, "ReferenceMask", f"{scene_id}_Mask.tif")
            self.masks.append(mask_path)


    def data_distribution_table(self):
        label_dist = {key: 0 for key in self.labels}
        for image, mask in self.dataloader():
            for index, label in enumerate(self.labels):
                label_dist[self.labels[index]] += mask[:, :, :, index].sum()
        label_count = pd.DataFrame.from_dict(label_dist, orient='index')
        label_count.columns = ["Number of pixels"]
        label_count = label_count.astype(float)
        return label_count
    

    def show_image(self, index, show_title=False):
        img, mask = self[index]

        img_min, img_max = img.min(), img.max()
        img_norm = (img - img_min) / (img_max - img_min + 1e-8)

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
        fig.legend(handles=legend_elements, bbox_to_anchor=(0.2, height_factor, 0.6, 0.2), ncol=3, mode='expand',
                   loc='lower left', prop={'size': 12})
        plt.subplot(1, 2, 1)
        plt.imshow(img_norm)
        plt.axis("off")
        plt.subplot(1, 2, 2)
        plt.imshow(img_mask)
        plt.axis("off")
        fig.tight_layout()
        plt.show()
        return fig
