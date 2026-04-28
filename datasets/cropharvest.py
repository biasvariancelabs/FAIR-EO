from ..base import BaseDataset
from ..utils import image_loader
from .schemas import FAIREOSchema

import json
import os
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
import geopandas as gpd
import tifffile
from shapely.geometry import box
from skimage.draw import polygon
from matplotlib.patches import Patch
import math
import warnings
import logging

# Suppress noisy tifffile warnings about GDAL_NODATA casting
warnings.filterwarnings("ignore", message=".*GDAL_NODATA.*")
logging.getLogger('tifffile').setLevel(logging.ERROR)

'''
The CropHarvest dataset is a crop dataset of geo-referenced labels with satellite data inputs, 
each consisting of latitude, longitude, the associated agricultural label, and a satellite pixel 
time series.
'''

LABELS = ["no-data","cereals","vegetables_melons","fruits_nuts","oilseeds","root_tuber","beverage_spice","leguminous","sugar","other","non_crop"]

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

class CropHarvestDataset(BaseDataset):

    url = "https://zenodo.org/records/10251170"
    labels = LABELS
    color_mapping = [[0,0,0],[100,100,100],[255,0,0],[255,255,0],[0,0,255],[159,129,183],[0,255,0],[255,195,128],[200,200,90],[200,80,80],[50,50,0]] 
    name = "Crop Harvest dataset"
    schema = FAIREOSchema

    def __init__(self, config, date=None):
        # now call the constructor to validate the schema and load the data
        super().__init__(config)
        # load the data
        self.data_dir = self.config.data_dir
        self.csv_file = self.config.csv_file
        self.mode = self.config.mode
        self.images, self.data = self.load_dataset()


    def __getitem__(self, index):
        for i in np.arange(len(self.images)): # loop ignoring the images which do not have any polygons intersecting
            # The images in the eo_data folder contain 206 bands. 
            # These bands encompass 12 time steps. 
            # The first 2 bands are VV and VH, followed by the 13 S2 bands, a temperature band and a total precipitation band. 
            # This structure is repeated 12 times for the 12 time steps. The last 2 bands are the elevation and the slope.
            tif_path = os.path.join(self.data_dir, "eo_data", self.images[index])
            image = image_loader(tif_path) # 206 bands
            img_w, img_h = image.shape[:2]
            red = image[:,:,5]
            green = image[:,:,4]
            blue = image[:,:,3]
            rgb_image = np.dstack((red,green,blue))
            # normalize image for better visualization
            img_min, img_max = rgb_image.min(), rgb_image.max()
            rgb_image = (rgb_image - img_min) / (img_max - img_min + 1e-8)

            # apply transforms to RGB image if any
            if self.transform:
                rgb_image = self.transform(rgb_image)

            # load timeseries
            # Sentinel-1
            s1_series = []
            for t in range(12):
                vv_idx = t * 17
                vh_idx = vv_idx + 1
                vv = image[:, :, vv_idx]
                vh = image[:, :, vh_idx]
                s1_series.append(np.stack([vv, vh], axis=-1))  # (H, W, 2)

            s1 = np.stack(s1_series, axis=0)  # shape: (12, H, W, 2)

            # Sentinel-2
            s2_series = []
            for t in range(12):
                start_idx = t * 17 + 2     # S2 starts after VV, VH
                end_idx = start_idx + 13     # 13 S2 bands
                s2 = image[:, :, start_idx:end_idx]  # shape (H, W, 13)
                s2_series.append(s2)

            s2 = np.stack(s2_series, axis=0)  # shape: (12, H, W, 13)
            
            timeseries = {
                "sentinel1": s1,
                "sentinel2": s2
            }


            # Find the polygons which intersect the image
            # Load the GeoJSON with standard geometry
            mask = np.zeros((img_w,img_h),dtype=np.int64)
            cleaned_labels_path = os.path.join(self.data_dir, "labels_cleaned.geojson")
            if not os.path.exists(cleaned_labels_path):
                 # Just in case load_dataset hasn't run or file is missing
                 gdf = gpd.read_file(os.path.join(self.data_dir, "labels.geojson"))
            else:
                 gdf = gpd.read_file(cleaned_labels_path)
            
            x_min, y_max, dx, dy, epsg = get_transform_info(tif_path)
            x_max = x_min + img_w * dx
            y_min = y_max + img_h * dy
            
            if epsg:
                gdf = gdf.to_crs(epsg)

            image_bbox = box(x_min, y_min, x_max, y_max)
            intersecting_polygons = gdf[gdf.intersects(image_bbox)]
            if not intersecting_polygons.empty:
            
                for idx, geom in intersecting_polygons.iterrows():
                    label = geom["classification_label"]
                    
                    if label is None:
                        continue
                    geometry = geom["geometry"]
                    if geometry.geom_type == "Polygon":
                        coords = list(geometry.exterior.coords)

                        pixel_coords = [world_to_pixel(x, y, x_min, y_max, dx, dy) for x, y in coords]
                        label = geom["classification_label"]
                        mask = self.add_polygon(mask, pixel_coords, label)
                
                num_classes = len(self.labels)
                mask = np.eye(num_classes)[mask]
                if self.target_transform:
                    mask = self.target_transform(mask)

                break

            else: # skip to next image if there are no intersecting polygons
                index = index + 1
            
        return rgb_image, mask, timeseries


    def load_dataset(self):
        images = os.listdir(os.path.join(self.data_dir, "eo_data"))
        annotations = []
        data = []
        labels_path = os.path.join(self.data_dir, "labels.geojson")
        cleaned_labels_path = os.path.join(self.data_dir, "labels_cleaned.geojson") # without the "null" lines

        # Load the GeoJSON 
        gdf = gpd.read_file(labels_path)

        # Filter out rows where classification_label is "null"
        gdf_clean = gdf[gdf["classification_label"].notnull()]

        # Save cleaned GeoJSON to new file
        gdf_clean.to_file(cleaned_labels_path, driver="GeoJSON")

        print(f"Cleaned GeoJSON saved with {len(gdf_clean)} features (from {len(gdf)}).")

        with open(cleaned_labels_path, "r", encoding="utf-8") as f:
            self._jsonfile = json.load(f)

        for feat in self._jsonfile["features"]:
            test = int(feat["properties"]["is_test"])
            lon = feat["properties"]["lon"]
            lat = feat["properties"]["lat"]
            dataset = feat["properties"]["dataset"]
            classification_label = feat["properties"]["classification_label"]

            if self.mode == None: # consider all images
                data.append((lon,lat,dataset,classification_label))
                id = self.labels.index(classification_label)
                annotations.append({"label": id})

            elif self.mode == 'train':
                if test == 'false':
                    data.append((lon,lat,dataset,classification_label))
                    id = self.labels.index(classification_label)
                    annotations.append({"label": id})

        return images, data
    
    def __len__(self):
        return len(self.images)

    def get_labels(self):
        return self.labels
        
    def apply_transformations(self, image, mask):
        if self.joint_transform:
            image, mask = self.joint_transform((image, mask))
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            mask = self.target_transform(mask)
        return image, mask

    def add_polygon(self, mask:np.ndarray, coords:list, label:str)->np.ndarray:
        pixel_coords = [(int(round(x)), int(round(y))) for x, y in coords]

        # Check if we have enough coordinates for a valid polygon
        if len(pixel_coords) < 4:
            return mask

        # Determine class label
        label_value = self.labels.index(label)

        # Extract x and y coordinates for polygon filling
        x_coords = [coord[0] for coord in pixel_coords[:-1]]
        y_coords = [coord[1] for coord in pixel_coords[:-1]]

        rr, cc = polygon(y_coords, x_coords, shape=mask.shape)
        mask[rr, cc] = label_value
        return mask
    

    def show_image(self, index, show_title=False):
        # Show the rgb image
        img, mask, _ = self[index]
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
        plt.imshow(img)
        plt.axis("off")
        plt.subplot(1, 2, 2)
        plt.imshow(img_mask)
        plt.axis("off")
        fig.tight_layout()
        plt.show()
        return fig


    def show_timeseries(self, index, longitude, latitude):
        
        _, mask, timeseries = self[index]

        crop = mask[longitude, latitude]
        crop_name = "Unknown"
        for j, k in enumerate(self.labels):
            if crop[j] == 1:
                crop_name = k
                break

        s1 = timeseries["sentinel1"] 
        s2 = timeseries["sentinel2"] 

        time_range = np.arange(12)

        # extract S1 bands for a specific pixel
        s1_pixel = []
        for b in range(2):
            s1_band_series = [s1[i][longitude, latitude, b] for i in time_range]
            s1_pixel.append(s1_band_series)

        # extract Sentinel-2 bands
        s2_pixel = []
        for b in range(13):
            s2_band_series = [s2[i][longitude, latitude, b] for i in time_range]
            s2_pixel.append(s2_band_series)

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.set_title(f"Time series with index {index} from the region with pixel coordinates ({longitude},{latitude}), with label {crop_name}\n")

        for i, s1_band_series in enumerate(s1_pixel):
            if i == 0:
                band = 'VV'
            else:
                band = 'VH'
            ax.plot(time_range, s1_band_series, label=f"S1 {band}")

        for i, s2_band_series in enumerate(s2_pixel):
            ax.plot(time_range, s2_band_series, label=f"S2-B{i+1}", alpha=0.5)

        ax.set_xlabel("Time step")
        ax.set_ylabel("Pixel value")
        ax.legend()
        plt.tight_layout()
        plt.show()
        return fig
    

    def data_distribution_table(self):
        # Load cleaned GeoJSON
        gdf = gpd.read_file(os.path.join(self.data_dir, "labels_cleaned.geojson"))

        # Count the number of polygons per label
        label_dist = gdf["classification_label"].value_counts().reindex(self.labels, fill_value=0)

        label_count = pd.DataFrame({
            "Label": label_dist.index,
            "Number of polygons": label_dist.values
        })
        return label_count
    

    def data_distribution_barchart(self, show_title=True):
        label_count = self.data_distribution_table()
        fig, ax = plt.subplots(figsize=(12, 10))
        sns.barplot(data=label_count, x=label_count["Label"], y='Number of polygons', ax=ax)
        fig.autofmt_xdate()
        if show_title:
            ax.set_title(
                "Labels distribution for {}".format(self.get_name()), pad=20, fontsize=18
            )
        return fig
