import numpy as np
import os
import pandas as pd
import glob
import math
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import imageio

from .semantic_segmentation import SemanticSegmentationDataset
from ..utils import image_loader
from .schemas import FAIREOSchema


"""
GeoNRW dataset consists of orthorectified aerial photographs, lidar derived DEMs, land cover maps 
with 10 classes and TerraSARX spotlight acquisitions over the German state North Rhine Westphalia.
"""


class GeoNRWDataset(SemanticSegmentationDataset):
    url = "https://ieee-dataport.org/open-access/geonrw"

    labels = ["no data", "forest", "water", "agricultural", "urban", "grassland",
            "railway", "highway", "airport, shipyard", "roads", "buildings"]
    color_mapping = [
        [0, 0, 0],    
        [0, 100, 0],   
        [0, 0, 255],       
        [218, 165, 32],    
        [139, 0, 0],       
        [144, 238, 144],   
        [112, 128, 144],   
        [255, 255, 0],    
        [138, 43, 226],    
        [0, 255, 255],     
        [255, 0, 0]       
    ]

    name = "GeoNRW"
    schema = FAIREOSchema

    def __init__(self, config):
        # now call the constructor to validate the schema and split the data
        super().__init__(config)
     
        self.data_dir = self.config.data_dir
        self.selection = self.config.selection
        self.csv_file = self.config.csv_file


    def __getitem__(self, index):
        rgb = imageio.imread(self.rgb_images[index])

        if rgb.shape[-1] == 4:
            rgb = rgb[:, :, :3]
        
        mask = image_loader(self.masks[index]).astype(np.uint8)
        masks = [(mask == v) for v in range(len(self.labels))]
        mask = np.stack(masks, axis=-1).astype("float32")

        return self.apply_transformations(rgb, mask)


    def load_dataset(self, data_dir, csv_file):
        if not self.labels:
            raise ValueError("You need to provide the list of labels for the dataset")

        self.rgb_images = []
        self.dem_images = []
        self.masks = []

        city_dirs = [os.path.join(data_dir, d) for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]

        if csv_file is not None:
            df = pd.read_csv(csv_file)
            
            required_columns = {'rgb_path', 'dem_path', 'mask_path'}
            if not required_columns.issubset(df.columns):
                raise ValueError(f"CSV file must contain columns: {required_columns}")
            
            for rgb in df['rgb_path']:
                if os.path.exists(os.path.join(data_dir, rgb)):
                    self.rgb_images.append(os.path.join(data_dir, rgb))

            self.dem_images = [os.path.join(data_dir, dem) for dem in df['dem_path']]
            for mask in df['mask_path']:
                if os.path.exists(os.path.join(data_dir, mask)):
                    self.masks.append(os.path.join(data_dir, mask))
            
        else:
            for city in city_dirs:
                rgb_files = glob.glob(os.path.join(city, "*_rgb.jp2"))
                for rgb_path in rgb_files:
                    base_id = os.path.basename(rgb_path).replace("_rgb.jp2", "")
                    dem_path = os.path.join(city, f"{base_id}_dem.tif")
                    seg_path = os.path.join(city, f"{base_id}_seg.tif")

                    if not (os.path.exists(dem_path) and os.path.exists(seg_path)):
                        #print(f"Warning: Missing DEM or SEG for {base_id} in {city}")
                        continue

                    self.rgb_images.append(rgb_path)
                    self.dem_images.append(dem_path)
                    self.masks.append(seg_path)

              
    def __len__(self):
        return len(self.rgb_images)

    def data_distribution_table(self):
        label_dist = {key: 0 for key in self.labels}

        for image, mask in self.dataloader():
            for index, label in enumerate(self.labels):
                label_dist[label] += mask[:, :, :, index].sum().item()

        label_count = pd.DataFrame.from_dict(label_dist, orient='index')
        label_count.columns = ["Number of pixels"]
        label_count = label_count.astype(float)
        return label_count
    

    def show_image(self, index, show_title=False):
        rgb, mask = self[index]
        dem = image_loader(self.dem_images[index])

        if hasattr(rgb, 'cpu'):
            rgb = rgb.cpu().numpy().transpose(1, 2, 0)
        if hasattr(mask, 'cpu'):
            mask = mask.cpu().numpy()
        if hasattr(dem, 'cpu'):
            dem = dem.cpu().numpy()

        if dem.ndim == 3:
            dem = dem.squeeze()

        img_mask = np.zeros([mask.shape[0], mask.shape[1], 3], np.uint8)
        legend_elements = []
        for i, label in enumerate(self.labels):
            legend_elements.append(
                Patch(
                    facecolor=tuple([x / 255 for x in self.color_mapping[i]]),
                    label=label,
                )
            )
            img_mask[np.where(mask[:, :, i] == 1)] = self.color_mapping[i]

        fig = plt.figure(figsize=(16, 6))

        height_factor = math.ceil(len(self.labels) / 3)
        height_factor = 0.73 if height_factor == 4 else 0.80 if height_factor == 2 else 0.81

        fig.legend(handles=legend_elements, bbox_to_anchor=(0.2, height_factor, 0.6, 0.2),
                ncol=3, mode='expand', loc='lower left', prop={'size': 12})

        # RGB Image
        plt.subplot(1, 3, 1)
        plt.imshow(rgb.astype(np.uint8))
        plt.title("RGB Image")
        plt.axis("off")

        # DEM Image
        plt.subplot(1, 3, 2)
        plt.imshow(dem, cmap="terrain")
        plt.title("DEM - Digital Elevation Model")
        plt.axis("off")

        # Segmentation Mask
        plt.subplot(1, 3, 3)
        plt.imshow(img_mask)
        plt.title("Segmentation Mask")
        plt.axis("off")

        fig.tight_layout()
        plt.show()
        return fig