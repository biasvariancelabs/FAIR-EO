import numpy as np
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import math
from shapely import wkt
from skimage.draw import polygon

from aitlas.datasets.semantic_segmentation import SemanticSegmentationDataset
from aitlas.datasets.schemas import SegmentationDatasetSchema
from aitlas.utils import image_loader

"""
SpaceNet 2 is the second dataset released as part of the SpaceNet Challenge series. It focuses on building 
footprint detection from very high-resolution WorldView-3 satellite imagery. The dataset covers four cities 
(Las Vegas, Paris, Shanghai, Khartoum) with over 300,000 building annotations. Images are provided at ~0.3 m 
resolution, and the corresponding labels are vector building footprints.
"""


class SpaceNet2Dataset(SemanticSegmentationDataset):

    url = [
        "https://www.kaggle.com/datasets/ugorjiir/spacenet-2-paris-buildings",
        "https://medium.com/the-downlinq/2nd-spacenet-competition-winners-code-release-c7473eea7c11"
    ]
    name = "SpaceNet 2"
    schemas = SegmentationDatasetSchema
    labels = [
        "non-building",
        "building"
    ]
    color_mapping = [
        [0,0,0],
        [255,255,255]
    ]

    def __init__(self, config):
        # now call the constructor to validate the schema and split the data
        super().__init__(config)
        self.data_dir = self.config.data_dir
        self.csv_file = self.config.csv_file

    def __getitem__(self, index):
        img_id = self.data[index]
        AOI = '_'.join(img_id.split('_')[0:3])
        image = image_loader(os.path.join(self.data_dir,AOI,'PS-RGB',f'SN2_buildings_train_{img_id}.tif'))
        csv_file = os.path.join(self.data_dir,AOI,f'SN2_buildings_train_{AOI}_solutions.csv')

        image = image.astype(np.float32)
        img_h, img_w = image.shape[:2]
        mask_2d = np.zeros((img_h, img_w), dtype = np.int64)

        df = pd.read_csv(csv_file)
        img_id_csv = AOI + '_' + img_id.split('_')[4]
        image_rows = df[df['ImageId'].str.contains(f'^{img_id_csv}')]
        for index, row in image_rows.iterrows():
            wkt_geom = row['PolygonWKT_Pix']

            # ignore invalid geometries
            if pd.isna(wkt_geom) or 'EMPTY' in str(wkt_geom):
                continue

            geometry = wkt.loads(wkt_geom)
            coords = list(geometry.exterior.coords)
            mask_2d = self.add_building(mask_2d, coords)

        num_classes = len(self.labels)
        mask = np.eye(num_classes)[mask_2d]
        return self.apply_transformations(image, mask)


    def load_dataset(self, data_dir, csv_file=None):
        img_ids = []

        for AOI in os.listdir(data_dir):
            images_path = os.path.join(data_dir, AOI, 'PS-RGB')
            for filename in os.listdir(images_path):
                if filename.endswith('.tif'):
                    img_id = filename[20:-4]
                    img_ids.append(img_id)

        self.data = img_ids

    
    def add_building(self, mask:np.ndarray, coords:list)->np.ndarray:
        # Handle 3D coordinates (x, y, z) by ignoring z
        pixel_coords = [(int(round(x)), int(round(y))) for x, y, *_ in coords]

        if len(pixel_coords) < 4:
            return mask

        # Extract x and y coordinates for polygon filling
        x_coords = [coord[0] for coord in pixel_coords[:-1]]
        y_coords = [coord[1] for coord in pixel_coords[:-1]]

        rr, cc = polygon(y_coords, x_coords, shape=mask.shape)
        mask[rr, cc] = 1
        return mask


    def data_distribution_table(self):
        stats = {
            'non-building': 0,
            'building': 0
        }

        # Get image dimensions from first image
        sample_img_id = self.data[0]
        AOI = '_'.join(sample_img_id.split('_')[0:3])
        sample_image = image_loader(os.path.join(self.data_dir,AOI,'PS-RGB',f'SN2_buildings_train_{sample_img_id}.tif'))
        img_h, img_w = sample_image.shape[:2]
        total_pixels_per_image = img_h * img_w
        total_pixels_all_images = total_pixels_per_image * len(self.data)

        # Read CSV and count building pixels
        for AOI in os.listdir(self.data_dir):
            csv_file = os.path.join(self.data_dir,AOI,f'SN2_buildings_train_{AOI}_solutions.csv')
            df = pd.read_csv(csv_file)
            for _, row in df.iterrows():
                wkt_geom = row['PolygonWKT_Pix']
                if pd.isna(wkt_geom) or 'EMPTY' in str(wkt_geom):
                    continue
                try:
                    geometry = wkt.loads(wkt_geom)
                    # calculate polygon area
                    area = geometry.area
                    stats['building'] += int(area)
                except Exception as e:
                    continue

        # Calculate non-building pixels = total - building pixels
        stats['non-building'] = total_pixels_all_images - stats['building']

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

    def __len__(self):
        return len(self.data)

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
        if img.max() == img.min():  # Handle edge case
            img_normalized = np.zeros_like(img)

        plt.imshow(img_normalized)
        plt.axis("off")
        plt.subplot(1, 2, 2)
        plt.imshow(img_mask)
        plt.axis("off")
        fig.tight_layout()
        plt.show()
        return fig