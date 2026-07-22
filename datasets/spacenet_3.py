import numpy as np
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from skimage.draw import line
import math
from shapely import wkt

from aitlas.datasets.semantic_segmentation import SemanticSegmentationDataset
from aitlas.datasets.schemas import SegmentationDatasetSchema
from aitlas.utils import image_loader

"""
SpaceNet 3 is the third dataset released as part of the SpaceNet Challenge series. It focuses on road 
network detection from very high-resolution WorldView-3 satellite imagery. The dataset covers four 
cities (Las Vegas, Paris, Shanghai, Khartoum) with over 8000 kilometers of roads. Images are provided 
at ~0.3 m resolution, and the corresponding labels are vector building footprints.
"""


class SpaceNet3Dataset(SemanticSegmentationDataset):

    url = ["https://medium.com/the-downlinq/introducing-the-spacenet-road-detection-and-routing-challenge-and-dataset-7604de39b779"]
    name = "SpaceNet 3"
    schemas = SegmentationDatasetSchema
    labels = ["non-road","road"]
    color_mapping = [[0,0,0],[255,255,255]]

    def __init__(self, config):
        # now call the constructor to validate the schema and split the data
        super().__init__(config)
        self.data_dir = self.config.data_dir
        self.csv_file = self.config.csv_file

    def __getitem__(self, index):
        img_id = self.data[index]
        AOI = '_'.join(img_id.split('_')[0:3])
        image = image_loader(os.path.join(self.data_dir,AOI,'PS-RGB',f'SN3_roads_train_{img_id}.tif'))
        csv_file = os.path.join(self.data_dir,AOI,f'SN3_roads_train_{AOI}_road_labels.csv')

        image = image.astype(np.float32)

        # normalize uint16 range [0, 65535] to [0, 1]
        if image.max() > 1.0:
            image = image / 65535.0  

        img_h, img_w = image.shape[:2]
        mask_2d = np.zeros((img_h, img_w), dtype = np.int64)

        df = pd.read_csv(csv_file)
        img_id_csv = AOI + '_' + img_id.split('_')[4]
        image_rows = df[df['ImageId'].str.contains(f'^{img_id_csv}')]

        for index, row in image_rows.iterrows():
            wkt_geom = row['WKT_Pix']

            # ignore invalid geometries
            if pd.isna(wkt_geom) or 'EMPTY' in str(wkt_geom):
                continue

            geometry = wkt.loads(wkt_geom)
            coords = list(geometry.coords)
            mask_2d = self.add_road(mask_2d, coords)

        num_classes = len(self.labels)
        mask = np.eye(num_classes)[mask_2d]
        return self.apply_transformations(image, mask)

    def load_dataset(self, data_dir, csv_file=None):
        img_ids = []
        for AOI in os.listdir(data_dir):
            images_path = os.path.join(data_dir, AOI, 'PS-RGB')
            for filename in os.listdir(images_path):
                if filename.endswith('.tif'):
                    img_id = filename[16:-4]  # remove "SN3_roads_train_" and ".tif"
                    img_ids.append(img_id)
        self.data = img_ids

    def add_road(self, mask:np.ndarray, coords:list)->np.ndarray:
        pixel_coords = [(int(round(x)), int(round(y))) for x, y in coords]

        for i in range(len(pixel_coords) - 1):
            x0, y0 = pixel_coords[i]
            x1, y1 = pixel_coords[i + 1]

            # Get line coordinates between points
            rr, cc = line(y0, x0, y1, x1)

            # Ensure coordinates are within mask bounds
            valid_coords = (rr >= 0) & (rr < mask.shape[0]) & (cc >= 0) & (cc < mask.shape[1])
            rr, cc = rr[valid_coords], cc[valid_coords]

            # Set road pixels to appropriate class
            mask[rr, cc] = 1
        return mask
    
    def __len__(self):
        return len(self.data)

    def data_distribution_table(self):
        stats = {
            'non-road': 0,
            'road': 0
        }

        # Get image dimensions from first image
        sample_img_id = self.data[0]
        AOI = '_'.join(sample_img_id.split('_')[0:3])
        sample_image = image_loader(os.path.join(self.data_dir,AOI,'PS-RGB',f'SN3_roads_train_{sample_img_id}.tif'))
        img_h, img_w = sample_image.shape[:2]
        total_pixels_per_image = img_h * img_w
        total_pixels_all_images = total_pixels_per_image * len(self.data)

        # Read CSV and count road pixels
        for AOI in os.listdir(self.data_dir):
            csv_file = os.path.join(self.data_dir,AOI,f'SN3_roads_train_{AOI}_road_labels.csv')
            df = pd.read_csv(csv_file)
            for _, row in df.iterrows():
                wkt_geom = row['WKT_Pix']
                if pd.isna(wkt_geom) or 'EMPTY' in str(wkt_geom):
                    continue
                try:
                    geometry = wkt.loads(wkt_geom)
                    # For roads, calculate buffered line area (width ~4 pixels)
                    length = geometry.length
                    road_area = length * 4 
                    stats["road"] += int(road_area)
                except Exception as e:
                    continue

        # Calculate non-road pixels = total - road pixels
        stats['non-road'] = total_pixels_all_images - stats['road']

        # Convert to DataFrame
        label_count = pd.DataFrame.from_dict(stats, orient='index')
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

    def show_image(self, index, show_title=False):
        img, mask = self[index]
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

        img_normalized = (img - img.min()) / (img.max() - img.min())
        if img.max() == img.min():  # handle edge case
            img_normalized = np.zeros_like(img)

        plt.imshow(img_normalized)
        plt.axis("off")
        plt.subplot(1, 2, 2)
        plt.imshow(img_mask)
        plt.axis("off")
        fig.tight_layout()
        plt.show()
        return fig