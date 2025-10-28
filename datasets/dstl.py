import os
import math
import json
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from skimage.draw import polygon
from matplotlib.patches import Patch

from ..utils import image_loader
from .schemas import SegmentationDatasetSchema
from .semantic_segmentation import SemanticSegmentationDataset

"""
The Dstl Satellite Imagery Feature Detection dataset was released as part of a Kaggle competition 
organized by the UK Defence Science and Technology Laboratory (Dstl). It consists of very high 
resolution WorldView-3 satellite images covering multiple geographic areas. The dataset includes 
detailed polygon annotations for a variety of features.
"""


class DstlDataset(SemanticSegmentationDataset):
    url = ["https://www.kaggle.com/competitions/dstl-satellite-imagery-feature-detection/overview"]
    name = "DSTL Satellite Imagery Feature Detection"
    schemas = SegmentationDatasetSchema
    labels = [
        "background",
        "Buildings",
        "Misc. Manmade structures",
        "Road",
        "Track",
        "Trees",
        "Crops",
        "Waterway",
        "Standing water",
        "Vehicle Large",
        "Vehicle Small",
    ]
    color_mapping = [
        [0, 0, 0],
        [0, 0, 255],
        [255, 255, 255],
        [255, 0, 0],
        [255, 192, 203],
        [128, 0, 128],
        [255, 20, 147],
        [255, 255, 0],
        [165, 42, 42],
        [210, 180, 140],
        [124, 252, 0],
    ]
    geojson_to_label = {
        # 1. Buildings
        "001_MM_L2_LARGE_BUILDING.geojson": "Buildings",
        "001_MM_L3_EXTRACTION_MINE.geojson": "Buildings",
        "001_MM_L3_NON_RESIDENTIAL_BUILDING.geojson": "Buildings",
        "001_MM_L3_RESIDENTIAL_BUILDING.geojson": "Buildings",
        "001_MM_L4_BRIDGE.geojson": "Buildings",
        "001_MM_L2_FUEL_STORAGE_FACILITY.geojson": "Buildings",
        "001_MM_L2_FORTIFIED_BUILDING.geojson": "Buildings",
        # 2. Roads
        "002_TR_L3_GOOD_ROADS.geojson": "Road",
        # 3. Tracks
        "002_TR_L4_POOR_DIRT_CART_TRACK.geojson": "Track",
        "002_TR_L6_FOOTPATH_TRAIL.geojson": "Track",
        # 4. Vehicles
        "003_VH_L4_LARGE_VEHICLE.geojson": "Vehicle Large",
        "003_VH_L5_SMALL_VEHICLE.geojson": "Vehicle Small",
        "003_VH_L6_MOTORBIKE.geojson": "Vehicle Small",
        "003_VH_L4_LOGISTICS_VEHICLE.geojson": "Vehicle Large",
        # 5. Utilities / Misc Manmade
        "001_MM_L5_MISC_SMALL_STRUCTURE.geojson": "Misc. Manmade structures",
        "004_UPI_L5_PYLONS.geojson": "Misc. Manmade structures",
        "004_UPI_L6_SATELLITE_DISHES_DISH_AERIAL.geojson": "Misc. Manmade structures",
        "005_VO_L6_MASTS_RADIO_TOWER.geojson": "Misc. Manmade structures",
        "005_VO_L7_FLAGPOLE.geojson": "Misc. Manmade structures",
        # 6. Vegetation / Trees
        "006_VEG_L2_SCRUBLAND.geojson": "Trees",
        "006_VEG_L2_WOODLAND.geojson": "Trees",
        "006_VEG_L3_HEDGEROWS.geojson": "Trees",
        "006_VEG_L5_GROUP_TREES.geojson": "Trees",
        "006_VEG_L5_STANDALONE_TREES.geojson": "Trees",
        # 7. Agriculture / Crops
        "007_AGR_L2_CONTOUR_PLOUGHING_CROPLAND.geojson": "Crops",
        "007_AGR_L2_DEMARCATED_NON_CROP_FIELD.geojson": "Crops",
        "007_AGR_L2_ORCHARD.geojson": "Crops",
        "007_AGR_L6_ROW_CROP.geojson": "Crops",
        "007_AGR_L7_FARM_ANIMALS_IN_FIELD.geojson": "Crops",
        # 8. Water
        "008_WTR_L2_STANDING_WATER.geojson": "Standing water",
        "008_WTR_L3_DRY_RIVERBED.geojson": "Waterway",
        "008_WTR_L3_WATERWAY.geojson": "Waterway",
    }

    def __init__(self, config):
        # now call the constructor to validate the schema and split the data
        super().__init__(config)
        self.data_dir = self.config.data_dir

    def __getitem__(self, index):
        img_id = self.data[index]
        image = image_loader(os.path.join(self.data_dir, f"three_band/{img_id}.tif"))
        image = image.astype(np.float32)
        image = image.transpose(1, 2, 0)
        image = np.flip(image, axis=0).copy()

        img_h, img_w = image.shape[:2]

        mask_2d = np.zeros((img_h, img_w), dtype=np.int64)

        geojsons_dir = os.path.join(self.data_dir, f"train_geojson_v3/{img_id}")
        for file in os.listdir(geojsons_dir):
            with open(os.path.join(geojsons_dir, file), "r") as f:
                geojson = json.load(f)

                if file not in self.geojson_to_label:
                    continue

                label = self.geojson_to_label[file]

                for feat in geojson["features"]:
                    geom_type = feat["geometry"]["type"]
                    geo_coords = feat["geometry"]["coordinates"]

                    # handle both Polygon and MultiPolygon
                    polygons = []
                    if geom_type == "Polygon":
                        polygons = [geo_coords[0]]
                    elif geom_type == "MultiPolygon":
                        polygons = [poly[0] for poly in geo_coords]

                    # transform from real-world coordinates to pixel coordinates
                    xmax, ymin = self.grid_sizes.loc[img_id, ["Xmax", "Ymin"]]
                    for polygon in polygons:
                        if polygon and len(polygon) > 0:
                            coords = [
                                (
                                    coord[0] / xmax * img_w,
                                    (coord[1] - ymin) / (0 - ymin) * img_h,
                                )
                                for coord in polygon
                            ]
                            mask_2d = self.add_building(mask_2d, coords, label)

        num_classes = len(self.labels)
        mask = np.eye(num_classes)[mask_2d]
        return self.apply_transformations(image, mask)

    def load_dataset(self, data_dir, csv_file=None):
        self.grid_sizes = pd.read_csv(os.path.join(data_dir, "grid_sizes.csv"), index_col=0)

        img_ids = []
        for filename in os.listdir(os.path.join(data_dir, "train_geojson_v3")):
            if filename[0] != "6":
                continue
            img_ids.append(filename)
        self.data = img_ids

    
    def add_building(self, mask: np.ndarray, coords: list, label) -> np.ndarray:
        # Handle 3D coordinates (x, y, z) by ignoring z
        pixel_coords = [(int(round(x)), int(round(y))) for x, y, *_ in coords]

        if len(pixel_coords) < 4:
            return mask

        x_coords = [coord[0] for coord in pixel_coords[:-1]]
        y_coords = [coord[1] for coord in pixel_coords[:-1]]

        rr, cc = polygon(y_coords, x_coords, shape=mask.shape)

        index = self.labels.index(label)
        mask[rr, cc] = index
        return mask
    

    def data_distribution_table(self):
        label_counts = {label: 0 for label in self.labels[1:]}

        for i in range(len(self.data)):
            _, mask = self[i]
            for j, label in enumerate(self.labels[1:], 1):
                label_counts[label] += np.sum(mask[:, :, j])

        return pd.DataFrame(list(label_counts.items()), columns=["Label", "Number of pixels"])
    
    def data_distribution_barchart(self, show_title=True):
        label_count = self.data_distribution_table()
        fig, ax = plt.subplots(figsize=(12, 10))
        sns.barplot(data=label_count, x='Label', y='Number of pixels', ax=ax)
        fig.autofmt_xdate()
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
        height_factor = math.ceil(len(self.labels) / 3)
        if height_factor == 4:
            height_factor = 0.73
        elif height_factor == 2:
            height_factor = 0.80
        else:
            height_factor = 0.81
        fig.legend(
            handles=legend_elements,
            bbox_to_anchor=(0.2, height_factor, 0.6, 0.2),
            ncol=3,
            mode="expand",
            loc="lower left",
            prop={"size": 12},
        )
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
