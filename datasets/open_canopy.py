import numpy as np
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import geopandas as gpd
import tifffile
from shapely.geometry import box
from shapely.affinity import affine_transform
from skimage.draw import polygon as sk_polygon
import json

from aitlas.base import BaseDataset
from aitlas.utils import image_loader
from .schemas import FAIREOSchema

"""
Open-Canopy is an open-access and country-scale benchmark for very high resolution (1.5 m) 
canopy height estimation. It can be used for canopy height change detection between two images 
taken at different years.
"""

class OpenCanopyDataset(BaseDataset):
    url = "https://huggingface.co/datasets/AI4Forest/Open-Canopy"

    labels = ["background", "forest"]
    color_mapping = [[0,0,0],[0,255,0]] 
    name = "Open-Canopy"
    schema = FAIREOSchema


    def __init__(self, config):
        super().__init__(config)
        self.data_dir = self.config.data_dir
        self.selection = self.config.selection
        self.csv_file = self.config.csv_file
        self.mode = self.config.mode
        self.images = self.load_dataset(self.data_dir, self.csv_file)

    def __getitem__(self, index):
        image_path = self.images[index]
        image = image_loader(image_path)

        x_origin, y_origin, dx, dy, height, width = self._read_geotiff_tags(image_path)

        tile_geom = box(
            x_origin,
            y_origin + dy * height,  
            x_origin + dx * width, 
            y_origin                
        )

        # CRS coords to pixel coords
        inv = [1/dx, 0, 0, 1/dy, -x_origin/dx, -y_origin/dy]

        mask_2d = np.zeros((height, width), dtype=np.int64)
        candidates = self._get_candidates(tile_geom) # all masks are in the same file

        for geom in candidates.geometry:
            if geom is None or geom.is_empty:
                continue
            pixel_geom = affine_transform(geom, inv)
            mask_2d = self._draw_polygon(mask_2d, pixel_geom, height, width)

        # Stack per-class binary masks (H, W, num_classes)
        masks = [(mask_2d == v) for v, label in enumerate(self.labels)]
        mask = np.stack(masks, axis=-1).astype("float32")

        if self.selection == "rgb":
            image = image[:, :, 0:3]

        return self.apply_transformations(image, mask)


    def load_dataset(self, data_dir, csv_file):
        if not self.labels:
            raise ValueError("You need to provide the list of labels for the dataset")
        
        if csv_file: # train/val/test split
            with open(csv_file) as f:
                geojson = json.load(f)
            split_images = set(
                f["properties"]["image_name"]
                for f in geojson["features"]
                if f["properties"]["split"] == self.mode
            )

        self.images = []
        years = ["2021", "2022", "2023"]
        for year in years:
            spot_dir = os.path.join(data_dir, year, "spot")
            for img_id in os.listdir(spot_dir):
                if csv_file:
                    if img_id in split_images:
                        self.images.append(os.path.join(spot_dir, img_id))
                else:
                    self.images.append(os.path.join(spot_dir, img_id))

        self.gdf = gpd.read_parquet(os.path.join(data_dir, "forest_mask.parquet"))

        # Pre-compute bounds once as numpy array for fast filtering
        # columns: minx, miny, maxx, maxy
        self.gdf_bounds = self.gdf.geometry.bounds.values

        return self.images

    def _get_candidates(self, tile_geom):
        """Vectorized bounding box filter using pre-computed bounds array."""
        tx_min, ty_min, tx_max, ty_max = tile_geom.bounds
        b = self.gdf_bounds  # (N, 4): minx, miny, maxx, maxy
        overlap = (
            (b[:, 2] >= tx_min) & 
            (b[:, 0] <= tx_max) &  
            (b[:, 3] >= ty_min) &  
            (b[:, 1] <= ty_max)   
        )
        return self.gdf[overlap]

    def _read_geotiff_tags(self, image_path):
        """Extract spatial metadata from GeoTIFF tags using tifffile."""
        with tifffile.TiffFile(image_path) as tif:
            page = tif.pages[0]
            height, width = page.shape[0], page.shape[1]
            tags = {tag.name: tag.value for tag in page.tags.values()}

            if 'ModelPixelScaleTag' in tags and 'ModelTiepointTag' in tags:
                scale = tags['ModelPixelScaleTag']
                tiepoint = tags['ModelTiepointTag']
                dx = scale[0]
                dy = -scale[1]  # negative: Y decreases downward in pixel space
                x_origin = tiepoint[3]
                y_origin = tiepoint[4]
            elif 'ModelTransformationTag' in tags:
                t = tags['ModelTransformationTag']
                dx, dy = t[0], t[5]
                x_origin, y_origin = t[3], t[7]
            else:
                raise ValueError(f"No geospatial tags found in {image_path}")

        return x_origin, y_origin, dx, dy, height, width

    def _draw_polygon(self, mask, geom, height, width):
        """Rasterize a shapely geometry into mask_2d using skimage."""
        if geom.geom_type == 'MultiPolygon':
            for part in geom.geoms:
                mask = self._draw_polygon(mask, part, height, width)
            return mask
        if geom.geom_type != 'Polygon' or geom.is_empty:
            return mask

        # Fill exterior ring
        coords = np.array(geom.exterior.coords)
        rr, cc = sk_polygon(coords[:, 1], coords[:, 0], shape=(height, width))
        mask[rr, cc] = 1

        # Punch out interior holes
        for interior in geom.interiors:
            coords = np.array(interior.coords)
            rr, cc = sk_polygon(coords[:, 1], coords[:, 0], shape=(height, width))
            mask[rr, cc] = 0

        return mask
    
    def apply_transformations(self, image, mask):
        if self.joint_transform:
            image, mask = self.joint_transform((image, mask))
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            mask = self.target_transform(mask)
        return image, mask
    
    def __len__(self):
        return len(self.images)
    
    def get_labels(self):
        return self.labels

    def data_distribution_table(self):
        label_dist = {key: 0 for key in self.labels}
        for i in np.arange(len(self.images)):
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