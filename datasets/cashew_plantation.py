import json
import os
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
import geopandas as gpd
import tifffile
from skimage.draw import polygon
from matplotlib.patches import Patch
import math
import warnings
import logging

# Suppress noisy tifffile warnings about GDAL_NODATA casting
warnings.filterwarnings("ignore", message=".*GDAL_NODATA.*")
logging.getLogger('tifffile').setLevel(logging.ERROR)

from aitlas.base import BaseDataset
from aitlas.utils import image_loader
from .schemas import FAIREOSchema


'''
This dataset contains labels for cashew plantations in a 120 km^2 area in the center of Benin. Each pixel is classified 
for Well-managed plantation, Poorly-managed plantation, No plantation and other classes.
'''

LABELS = ["Well-managed plantation","Poorly-managed plantation","Non-plantation","Residential","Background","Uncertain"]

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

class CashewPlantationDataset(BaseDataset):

    url = "https://staging.source.coop/technoserve/cashews-benin"
    labels = LABELS
    color_mapping = [[0,0,0],[100,100,100],[255,0,0],[255,255,0],[0,0,255],[0,255,0]] 
    name = "Cashew Plantation dataset"
    schema = FAIREOSchema

    def __init__(self, config, date=None):
        # now call the constructor to validate the schema and load the data
        super().__init__(config)
        # load the data
        self.data_dir = self.config.data_dir
        self.csv_file = self.config.csv_file
        self.mode = self.config.mode
        self.dates, self.data = self.load_dataset()


    def __getitem__(self, index):
        # load Sentinel-2 timeseries
        s2_series = []
        for i in np.arange(len(self.dates)):
            date = self.dates[i]
            bands = ["B01","B02","B03","B04","B05","B06","B07","B08","B8A","B09","B11","B12"]
            images = [image_loader(os.path.join(self.data_dir, "imagery", "00", date, f"{date}_{band}_10m.tif")) for band in bands]
            image = np.dstack(images)
            img_w, img_h = image.shape[:2]

            #normalize image for better visualization
            img_min, img_max = image.min(), image.max()
            image = (image - img_min) / (img_max - img_min + 1e-8)

            s2_series.append(image)

        s2 = np.stack(s2_series, axis=0)  # shape: (70, H, W, 12)

        # Load the GeoJSON 
        mask = np.zeros((img_w,img_h),dtype=np.int64)
        gdf = gpd.read_file(os.path.join(self.data_dir, "labels","00.geojson"))
        
        # the first band is used to transform the labels from real-world coordinates to pixel coordinates
        image_B01_path = os.path.join(self.data_dir, "imagery", "00", self.dates[0], f"{self.dates[0]}_B01_10m.tif")
        
        x_min, y_max, dx, dy, epsg = get_transform_info(image_B01_path)

        if epsg:
            gdf = gdf.to_crs(epsg)
        
        for idx, geom in gdf.iterrows():
            label = geom["class_name"]
            
            geometry = geom["geometry"]
            if geometry.geom_type == "Polygon":
                coords = list(geometry.exterior.coords)
                pixel_coords = [world_to_pixel(x, y, x_min, y_max, dx, dy) for x, y in coords]
                label = geom["class_name"]
                mask = self.add_polygon(mask, pixel_coords, label)
            
        num_classes = len(self.labels)
        mask = np.eye(num_classes)[mask]
            
        return s2, mask


    def load_dataset(self):
        annotations = []
        data = []

        dates = os.listdir(os.path.join(self.data_dir, "imagery","00"))
        labels_path = os.path.join(self.data_dir, "labels","00.geojson")

        with open(labels_path, "r", encoding="utf-8") as f:
            self._jsonfile = json.load(f)

        for feat in self._jsonfile["features"]:
            classification_label = feat["properties"]["class_name"]
            data.append(classification_label)
            id = self.labels.index(classification_label)
            annotations.append({"label": id})

        return dates, data
    
    def __len__(self):
        return len(self.dates)

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
        img, mask = self[index] 
        img = img[10,:,:,1:4] # rgb image for visualization of the 10th date
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
        s2, mask = self[index]

        crop = mask[longitude, latitude]
        crop_name = "Unknown"
        for j, k in enumerate(self.labels):
            if crop[j] == 1:
                crop_name = k
                break

        time_range = np.arange(70)

        # extract Sentinel-2 bands
        s2_pixel = []
        for b in range(12):
            s2_band_series = [s2[i][longitude, latitude, b] for i in time_range]
            s2_pixel.append(s2_band_series)

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.set_title(f"Time series with index {index} from the region with pixel coordinates ({longitude},{latitude}), with label {crop_name}\n")

        for i, s2_band_series in enumerate(s2_pixel):
            ax.plot(time_range, s2_band_series, label=f"S2-B{i+1}", alpha=0.5)

        ax.set_xlabel("Time step")
        ax.set_ylabel("Pixel value")
        ax.legend()
        plt.tight_layout()
        plt.show()
        return fig


    def data_distribution_table(self):
        label_dist = {key: 0 for key in self.labels}
        image, mask = self.__getitem__(0)
        for index, label in enumerate(self.labels):
            label_dist[self.labels[index]] += mask[:, :, index].sum()
        label_count = pd.DataFrame.from_dict(label_dist, orient='index')
        label_count.columns = ["Number of pixels"]
        return label_count

    def data_distribution_barchart(self, show_title=True):
        label_count = self.data_distribution_table()
        fig, ax = plt.subplots(figsize=(12, 10))
        sns.barplot(data=label_count, x=label_count.index, y='Number of pixels', ax=ax)
        fig.autofmt_xdate()
        if show_title:
            ax.set_title(
                "Labels distribution for {}".format(self.get_name()), pad=20, fontsize=18
            )
        return fig
