import os
import math
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from skimage.draw import polygon
import geopandas as gpd
import tifffile
from shapely.geometry import Polygon, MultiPolygon

from .semantic_segmentation import SemanticSegmentationDataset
from .schemas import SegmentationDatasetSchema
from ..utils import image_loader

"""
SpaceNet4: Off-Nadir Building Detection dataset was specifically built to explore advanced algorithms 
capabilities to process high off-nadir imagery. The dataset includes 27 WorldView 2 Satellite images 
from 7 degrees to 54 degrees off-nadir all captured within 5 minutes of each other.
"""

import warnings
import logging
# Suppress noisy tifffile warnings about GDAL_NODATA casting
warnings.filterwarnings("ignore", message=".*GDAL_NODATA.*")
logging.getLogger('tifffile').setLevel(logging.ERROR)

def get_transform_info(tif_path):
    """Extracts transform and CRS info using tifffile."""
    with tifffile.TiffFile(tif_path) as tif:
        page = tif.pages[0]
        # Extract Transform
        try:
            scale = page.tags['ModelPixelScaleTag'].value
            tiepoint = page.tags['ModelTiepointTag'].value
            x_min = tiepoint[3]
            y_max = tiepoint[4]
            dx = scale[0]
            dy = -scale[1] 
        except KeyError:
            x_min, y_max, dx, dy = 0, 0, 1, -1
            
        # Extract CRS (EPSG)
        epsg = None
        try:
            # GeoKeyDirectoryTag (34735)
            geokeys = page.tags[34735].value
            num_keys = geokeys[3]
            for i in range(num_keys):
                key_id = geokeys[4 + i*4]
                key_val = geokeys[4 + i*4 + 3]
                # 3072: ProjectedCSTypeGeoKey, 2048: GeographicTypeGeoKey
                if key_id in [3072, 2048] and key_val > 0:
                    epsg = f"EPSG:{key_val}"
                    break
        except:
            pass

        return x_min, y_max, dx, dy, epsg

def world_to_pixel(x, y, x_min, y_max, dx, dy):
    """Converts world coordinates to pixel coordinates."""
    col = (x - x_min) / dx
    row = (y - y_max) / dy
    return col, row

class SpaceNet4Dataset(SemanticSegmentationDataset):

    url = ["https://registry.opendata.aws/spacenet/"]
    name = "SpaceNet4: Off-Nadir Building Detection"
    schemas = SegmentationDatasetSchema
    labels = ["non-building","building"]
    color_mapping = [[0,0,0],[255,255,255]]

    def __init__(self, config):
        # now call the constructor to validate the schema and split the data
        super().__init__(config)
        self.data_dir = self.config.data_dir
        self.csv_file = self.config.csv_file

    def __getitem__(self, index):
        for i in np.arange(len(self.data)): # some geojsons are empty
            img_id = self.data[index]
            id = img_id[-18: -4]
            tif_path = os.path.join(self.data_dir,img_id)
            image = image_loader(tif_path)
            if image.shape[0] == 4:
                image = image.transpose(1,2,0)[:,:,:3]
            else:
                image = image[:,:,:3]

            # normalize the image
            image = (image - image.min()) / (image.max() - image.min()) 
            img_h, img_w = image.shape[:2]

            mask_2d = np.zeros((img_h, img_w), dtype = np.int64)
            annotation = os.path.join(self.data_dir,'geojson_buildings',f'SN4_buildings_train_AOI_6_Atlanta_geojson_buildings_{id}.geojson')
            if os.path.getsize(annotation) > 0:
                gdf = gpd.read_file(annotation)
                
                x_min, y_max, dx, dy, epsg = get_transform_info(tif_path)

                if epsg:
                    gdf = gdf.to_crs(epsg)

                for index, row in gdf.iterrows():
                    geometry = row['geometry']
                    if isinstance(geometry, Polygon):
                        polygons = [geometry]
                    elif isinstance(geometry, MultiPolygon):
                        polygons = list(geometry.geoms)

                    for poly in polygons:
                        coords = list(poly.exterior.coords)
                        # transform to pixel coordinates using manual helper
                        pixel_coords = [world_to_pixel(x, y, x_min, y_max, dx, dy) for x, y, *_ in coords]
                        # In the original, it was [(col, row) for row, col in rowcol]
                        # Our world_to_pixel returns (col, row) directly.
                        # Original: pixel_coords = [rowcol(transform, x, y) for x, y in coords] -> returns (row, col)
                        # Then: pixel_coords = [(col, row) for row, col in pixel_coords] -> returns (col, row)
                        # So we keep (col, row) from world_to_pixel.
                        mask_2d = self.add_building(mask_2d, pixel_coords)

                num_classes = len(self.labels)
                mask = np.eye(num_classes)[mask_2d]
                break
            else:
                index = index + 1

        return self.apply_transformations(image, mask)

    def load_dataset(self, data_dir, csv_file=None):
        img_ids = []
        for aoi in os.listdir(data_dir):
            if os.path.isdir(os.path.join(data_dir, aoi, 'PS-RGBNIR')):
                for filename in os.listdir(os.path.join(data_dir, aoi, 'PS-RGBNIR')):
                    if filename.endswith('.tif'):
                        img_ids.append(os.path.join(aoi, 'PS-RGBNIR', filename))
        self.data = img_ids

    def add_building(self, mask:np.ndarray, coords:list)->np.ndarray:
        # Handle 3D coordinates (x, y, z) by ignoring z
        pixel_coords = [(int(round(x)), int(round(y))) for x, y in coords]

        if len(pixel_coords) < 4:
            return mask

        # Extract x and y coordinates for polygon filling
        x_coords = [coord[0] for coord in pixel_coords[:-1]]
        y_coords = [coord[1] for coord in pixel_coords[:-1]]

        rr, cc = polygon(y_coords, x_coords, shape=mask.shape)
        mask[rr, cc] = 1
        return mask

    def __len__(self):
        return len(self.data)

    def data_distribution_table(self):
        label_dist = {key: 0 for key in self.labels}
        annot_paths = [] # several off-nadir images are assigned to the same geojson label file
        for i in np.arange(len(self.data)):
            image, mask = self.__getitem__(i)
            img_id = self.data[i]
            id = img_id[-18: -4]
            annotation = os.path.join(self.data_dir,'geojson_buildings',f'SN4_buildings_train_AOI_6_Atlanta_geojson_buildings_{id}.geojson')
            if annotation not in annot_paths:
                for index, label in enumerate(self.labels):
                    label_dist[self.labels[index]] += mask[:, :, index].sum()
                annot_paths.append(annotation)

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
        fig.legend(
            handles=legend_elements,
            bbox_to_anchor=(0.2, height_factor, 0.6, 0.2),
            ncol=3,
            mode='expand',
            loc='lower left',
            prop={'size': 12}
        )
        plt.subplot(1, 2, 1)

        plt.imshow(img)
        plt.axis("off")
        plt.subplot(1, 2, 2)
        plt.imshow(img_mask)
        plt.axis("off")
        fig.tight_layout()
        plt.show()
        return fig
