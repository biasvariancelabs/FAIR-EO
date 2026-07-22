from json.encoder import INFINITY
import os
import math
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from shapely import wkt
from skimage.draw import line
from matplotlib.patches import Patch

from aitlas.utils import image_loader
from .schemas import FAIREOSchema
from aitlas.datasets.semantic_segmentation import SemanticSegmentationDataset


"""
SpaceNet 5 focuses on the task of road network extraction in challenging geographic regions. The dataset 
provides satellite imagery (30 cm resolution) across multiple cities, along with labeled road networks 
in vector format. This dataloader considers 3 categories for route travel time/speed: slow, medium, fast.
"""


class SpaceNet5Dataset(SemanticSegmentationDataset):

    url = ["https://medium.com/the-downlinq/spacenet-5-dataset-release-80bd82d7c528"]
    name = "SpaceNet 5: Road Network and Route Travel Time"
    schema = FAIREOSchema
    labels = ["non-road","road"]
    color_mapping = [[0,0,0],[255,255,255]]
    max_speed = 0
    min_speed = INFINITY

    def __init__(self, config):
        # now call the constructor to validate the schema and split the data
        super().__init__(config)
        self.data_dir = self.config.data_dir
        self.csv_file = self.config.csv_file
        self.mode = self.config.mode

    def __getitem__(self, index):
        img_id = self.data[index]
        aoi = self.img_aoi[index]
        image = image_loader(os.path.join(self.data_dir,aoi,'PS-RGB',f'{img_id.replace("_chip", "_PS-RGB_chip")}.tif'))
        image = image.astype(np.float32)

        # normalize if values are large
        if image.max() > 1.0:
            image = image / 65535.0  # Normalize uint16 range [0, 65535] to [0, 1]

        img_h, img_w = image.shape[:2]
        mask_2d = np.zeros((img_h, img_w), dtype = np.int64)
        csv = os.path.join(self.data_dir, aoi, f'train_{aoi}_geojson_roads_speed_wkt_weighted_raw.csv')
        image_rows = pd.read_csv(csv)

        for index, row in image_rows.iterrows():
            if row['ImageId'] == img_id:
                wkt_geom = row['WKT_Pix']

                if float(row['travel_time_s']) == 0.0:
                    speed = 0.0
                else:
                    speed = float(row['length_m'])/float(row['travel_time_s'])
                if pd.isna(wkt_geom) or 'EMPTY' in str(wkt_geom):
                    continue

                geometry = wkt.loads(wkt_geom)
                coords = list(geometry.coords)
                mask_2d = self.add_road(mask_2d, coords,speed)

        num_classes = len(self.labels)
        mask = np.eye(num_classes)[mask_2d]
        return self.apply_transformations(image, mask)

    def load_dataset(self, data_dir, csv_file=None):
        img_ids = []
        img_aoi = []

        for aoi in ['AOI_7_Moscow','AOI_8_Mumbai']:
            csv_file = os.path.join(data_dir, aoi, f'train_{aoi}_geojson_roads_speed_wkt_weighted_raw.csv')
            csv = pd.read_csv(csv_file)

            for _, row in csv.iterrows():
                if float(row['travel_time_s']) == 0:
                    speed = 0
                else:
                    speed = float(row['length_m'])/float(row['travel_time_s'])
                self.max_speed = max(self.max_speed, speed)
                self.min_speed = min(self.min_speed, speed)

        self.max_speed = int(self.max_speed)
        self.min_speed = int(self.min_speed)
        self.interval = int((self.max_speed - self.min_speed)/3)

        if self.config.mode == "multi":
            # consider 3 categories for roads (slow, medium, fast)
            new_labels = ["no-road", "slow", "medium", "fast"]
            new_color_map = [[0,0,0], [245, 66, 66], [245, 179, 66], [66, 245, 66]]

            self.labels = new_labels
            self.color_mapping = new_color_map

        for aoi in os.listdir(data_dir):
            for filename in os.listdir(os.path.join(data_dir, aoi, 'PS-RGB')):
                if filename.endswith('.tif'):
                    temp = filename[:-4]
                    img_id = temp.replace("PS-RGB_", "")
                    img_ids.append(img_id)
                    img_aoi.append(aoi)
        self.data = img_ids
        self.img_aoi = img_aoi

    def add_road(self, mask:np.ndarray, coords:list, speed:float)->np.ndarray:
        pixel_coords = [(int(round(x)), int(round(y))) for x, y in coords]

        for i in range(len(pixel_coords) - 1):
            x0, y0 = pixel_coords[i]
            x1, y1 = pixel_coords[i + 1]

            # Get line coordinates between points
            rr, cc = line(y0, x0, y1, x1)

            # Ensure coordinates are within mask bounds
            valid_coords = (rr >= 0) & (rr < mask.shape[0]) & (cc >= 0) & (cc < mask.shape[1])
            rr, cc = rr[valid_coords], cc[valid_coords]

            if self.mode != "multi": # only "road" and "no-road" labels
                mask[rr, cc] = 1
            else:
                if hasattr(self, 'interval') and self.interval > 0:
                    # Calculate speed class: 0=background, 1-3=different speed ranges
                    speed_class = 1 + int((speed - self.min_speed) / self.interval)
                    # To valid class range
                    speed_class = max(1, min(speed_class, len(self.labels) - 1))
                    mask[rr, cc] = speed_class
                else:
                    mask[rr, cc] = 1
        return mask

    def __len__(self):
        return len(self.data)

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
        fig.tight_layout(rect=[0, 0, 1, 0.95])  
        plt.show()
        return fig