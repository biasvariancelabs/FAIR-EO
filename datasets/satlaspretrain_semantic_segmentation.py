from ..base import BaseDataset
from ..utils import image_loader
from .schemas import SatlasPretrainSemanticSegmentationSchema

import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import fnmatch
import json
import math
from matplotlib.patches import Patch
from skimage.draw import polygon, line

'''
SatlasPretrain is a large-scale pre-training dataset for remote sensing image understanding. It consists of 302M labels 
under 137 categories and seven label types: points, polygons, polylines, properties, segmentation labels, regression labels, 
and classification labels. This dataloader considers the semantic segmentation task.
'''

LABELS = ["no-data","aerialway_pylon","airport","airport_apron","airport_hangar","airport_runway","airport_taxiway","airport_terminal",
           "airstrip","aquafarm","communications_tower","crop","dam","gas_station","helipad","landfill","lighthouse","lock",
           "mineshaft","park","parking_garage","parking_lot","petroleum_well","power_plant","power_substation","power_tower",
           "quarry","raceway","silo","ski_resort","solar_farm","stadium","storage_tank","theme_park","track","wastewater_plant",
           "water_park","water_tower","wind_turbine","offshore_wind_turbine","building","ms_building","chimney","flagpole",
           "fountain","pier","railway","river","road","street_lamp","toll_booth","traffic_signals","offshore_platform",
           "rooftop_solar_panel"]


def create_tile_list(self, data_dir, output_file):
    '''
    The dataset is too large to download all the images. Not all images exist in our subset. 
    This function creates a txt file with a list of the available tiles and their corresponding 
    s2a, s2b, naip images and label file.
    '''
    txt_path = os.path.join(data_dir, output_file)

    with open(txt_path, "w") as f:
        f.write("tile_name,s2a_path,s2b_path,naip_path,labels_path\n")  # header line

        for tile in os.listdir(os.path.join(data_dir, 'static')):
            labels = None
            s2a, s2b, naip = None, None, None

            # Collect all label JSONs
            tile_static_path = os.path.join(data_dir, 'static', tile)
            for label in os.listdir(tile_static_path):
                if fnmatch.fnmatch(label, '*.json'):
                    labels = os.path.join(tile_static_path, label)

            # Check Sentinel-2A folder 
            for image in os.listdir(os.path.join(data_dir, 'sentinel2a')):
                tci_path = os.path.join(data_dir, 'sentinel2a', image, 'tci', f"{tile}.png")
                if os.path.exists(tci_path):
                    s2a = tci_path
                    break  

            # Check Sentinel-2B folder 
            for image in os.listdir(os.path.join(data_dir, 'sentinel2b')):
                tci_path = os.path.join(data_dir, 'sentinel2b', image, 'tci', f"{tile}.png")
                if os.path.exists(tci_path):
                    s2b = tci_path
                    break

            # Check NAIP folder 
            for image in os.listdir(os.path.join(data_dir, 'naip')):
                tci_path = os.path.join(data_dir, 'naip', image, 'tci', f"{tile}.png")
                if os.path.exists(tci_path):
                    naip = tci_path
                    break

            # Only include tiles that exist in at least one data source
            if s2a or s2b or naip:
                f.write(f"{tile},{s2a or 'None'},{s2b or 'None'},{naip or 'None'},{labels}\n")


class SatlasPretrainSemanticSegmentationDataset(BaseDataset):

    url = "https://github.com/allenai/satlas/blob/main/SatlasPretrain.md"
    labels = LABELS
    color_mapping = [[0,0,0],[255,0,0],[0,255,0],[0,0,255],[255,255,0],[255,0,255],[0,255,255],[128,0,0],[0,128,0],[0,0,128],
                     [128,128,0],[128,0,128],[0,128,128],[192,192,192],[128,128,128],[255,128,128],[128,255,128],[128,128,255],
                     [255,255,128],[255,128,255],[128,255,255],[64,0,0],[0,64,0],[0,0,64],[64,64,0],[64,0,64],[0,64,64],[192,0,0],
                     [0,192,0],[0,0,192],[192,192,0],[192,0,192],[0,192,192],[255,64,0],[255,128,0],[255,192,0],[255,0,64],[255,0,128],
                     [255,0,192],[0,255,64],[0,255,128],[0,255,192],[64,255,0],[128,255,0],[192,255,0],[0,64,255],[0,128,255],
                     [0,192,255],[255,255,255],[25,25,25],[255,128,64],[0,0,100],[100,0,0],[50,50,50]] 
    name = "SatlasPretrain Semantic Segmentation dataset"
    schema = SatlasPretrainSemanticSegmentationSchema

    def __init__(self, config):
        # now call the constructor to validate the schema and load the data
        super().__init__(config)
        # load the data
        self.data_dir = self.config.data_dir
        self.csv_file = self.config.csv_file
        self.mode = self.config.mode
        self.tile_list = self.config.tile_list
        self.data = self.load_dataset()

    def __getitem__(self, index):
        # load image
        if self.data[index][1] != 'None':
            self.satellite = 'Sentinel-2A'
            image_path = self.data[index][1]
            image = image_loader(image_path)

        elif self.data[index][2] != 'None':
            self.satellite = 'Sentinel-2B'
            image_path = self.data[index][2]
            image = image_loader(image_path)
        elif self.data[index][3] != 'None':
            self.satellite = 'NAIP'
            image_path = self.data[index][3]
            image = image_loader(image_path)

        json_path = str(self.data[index][4])

        img_h, img_w = image.shape[:2] 

        # Initialize mask
        mask_2d = np.zeros((img_h, img_w), dtype=np.int64)

        with open(json_path, "r") as f:
            annot_json = json.load(f)
        # The vector.json labels contain geometries that describe coordinates within the tile. 
        # The authors of the dataset define an 8192x8192 coordinate system within each tile. 
        # These coordinates can be converted to longitude-latitude coordinates as follows:
        scale_x = img_w / 8192
        scale_y = img_h / 8192

        for label_name, geoms in annot_json.items():
            if not geoms or label_name == None:
                continue

            for geom_entry in geoms:
                geom = geom_entry.get("Geometry", {})
                geom_type = geom.get("Type", "").lower()

                if geom_type == "polygon" and geom.get("Polygon"):
                    # First ring = exterior
                    exterior = geom["Polygon"][0]
                    pixel_coords_exterior = [(int(round(x*scale_x)), int(round(y*scale_y))) for x, y in exterior]
                    mask_2d = self.add_shape(mask_2d, pixel_coords_exterior, label_name, "polygon")

                    # Handle holes
                    for hole in geom["Polygon"][1:]:
                        pixel_coords_hole = [(int(round(x*scale_x)), int(round(y*scale_y))) for x, y in hole]
                        mask_2d = self.add_shape(mask_2d, pixel_coords_hole, label_name, "hole")

                elif geom_type == "polyline" and geom.get("Polyline"):
                    coords = geom["Polyline"]
                    pixel_coords = [(int(round(x*scale_x)), int(round(y*scale_y))) for x, y in coords]
                    mask_2d = self.add_shape(mask_2d, pixel_coords, label_name, "polyline")

                elif geom_type == "point" and geom.get("Point"):
                    coords = geom["Point"]
                    if isinstance(coords[0], (int, float)):
                        coords = [coords]
                    pixel_coords = [(int(round(x*scale_x)), int(round(y*scale_y))) for x, y in coords]
                    mask_2d = self.add_shape(mask_2d, pixel_coords, label_name, "point")

        # One-hot encode the mask
        num_classes = len(self.labels)
        mask = np.eye(num_classes)[mask_2d]

        return self.apply_transformations(image, mask)

    def load_dataset(self):
        # the images in the splits are naip (high resolution) or s2 (low resolution)
        # json files list the tiles assigned to each split
        data = []
        split_tile_coords = []
        
        if self.mode:
            if self.mode == 'train': 
                csv_file = ["/home/dragik/data/FAIR_EO/satlaspretrain_image_classification/metadata/train_highres_train.json",
                            "/home/dragik/data/FAIR_EO/satlaspretrain_image_classification/metadata/train_lowres_train.json"]
            if self.mode == 'val':
                csv_file = ["/home/dragik/data/FAIR_EO/satlaspretrain_image_classification/metadata/train_highres_val.json",
                            "/home/dragik/data/FAIR_EO/satlaspretrain_image_classification/metadata/train_lowres_val.json"]
            if self.mode == 'test':
                csv_file = ["/home/dragik/data/FAIR_EO/satlaspretrain_image_classification/metadata/test_highres.json",
                            "/home/dragik/data/FAIR_EO/satlaspretrain_image_classification/metadata/test_lowres.json"]
                
            for csv in csv_file:
                with open(csv, "r") as f:
                    split = json.load(f)
                    split_tile_coords.extend(split)
            tile_coords = [f"{x}_{y}" for x, y in split_tile_coords]

        # Read the txt file with the tile list
        data = []
        with open(self.tile_list, "r") as f:
            header = f.readline()  # skip header
            for line in f:
                parts = [p.strip() for p in line.split(",")]
                tile, s2a, s2b, naip, labels = parts

                # Only include tiles in the split
                if self.mode is None or tile in tile_coords:
                    data.append((tile, s2a, s2b, naip, labels))

        return data


    def add_shape(self, mask: np.ndarray, coords: list, label: str, geometry_type: str) -> np.ndarray:
        """
        Adds a shape (polygon, line, or point) to the mask.
        
        mask: np.ndarray, the mask to update
        coords: list of points [[x1, y1], [x2, y2], ...]
        label: str, label name
        geometry_type: 'polygon', 'polyline', or 'point'
        """
        
        # Convert coordinates to integer pixels
        pixel_coords = [(int(round(x)), int(round(y))) for x, y in coords]

        # Determine label index
        if geometry_type != "hole":
            int_label = [i for i, x in enumerate(self.labels) if label == x]
            int_label = int_label[0]
        else:
            int_label = 0  # holes are background

        if geometry_type == 'polygon':
            if len(pixel_coords) < 3:
                return mask  # not enough points for a polygon
            x_coords = [coord[0] for coord in pixel_coords]
            y_coords = [coord[1] for coord in pixel_coords]
            rr, cc = polygon(y_coords, x_coords, shape=mask.shape)
            mask[rr, cc] = int_label

        elif geometry_type == 'polyline':
            for i in range(len(pixel_coords) - 1):
                x0, y0 = pixel_coords[i]
                x1, y1 = pixel_coords[i + 1]
                rr, cc = line(y0, x0, y1, x1)
                # Make sure we don't go out of bounds
                rr = np.clip(rr, 0, mask.shape[0] - 1)
                cc = np.clip(cc, 0, mask.shape[1] - 1)
                mask[rr, cc] = int_label

        elif geometry_type == 'point':
            x, y = pixel_coords[0]
            if 0 <= y < mask.shape[0] and 0 <= x < mask.shape[1]:
                mask[y, x] = int_label

        return mask
    
    def __len__(self):
        return len(self.data)

    def apply_transformations(self, image, mask):
        if self.joint_transform:
            image, mask = self.joint_transform((image, mask))
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            mask = self.target_transform(mask)
        return image, mask

    def get_labels(self):
        return self.labels
    
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
        label_count = label_count.iloc[1:]
        fig, ax = plt.subplots(figsize=(12, 12))
        sns.barplot(data=label_count, x=label_count.index, y='Number of pixels', ax=ax)
        fig.autofmt_xdate()
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
        fig.legend(handles=legend_elements, bbox_to_anchor=(0, 0.85, 1, 0.2), ncol=4, mode='expand',
                   loc='lower left', prop={'size': 11})
        plt.subplot(1, 2, 1)
        plt.imshow(img)
        plt.axis("off")
        plt.title(f"{self.satellite} image", fontsize=12)
        plt.subplot(1, 2, 2)
        plt.imshow(img_mask)
        plt.axis("off")
        plt.title("Mask", fontsize=12)
        fig.tight_layout()
        plt.show()
        return fig