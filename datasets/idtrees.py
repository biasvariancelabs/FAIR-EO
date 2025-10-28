import os
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import geopandas as gpd
import glob
import rasterio
from shapely.geometry import box

from ..utils import image_loader
from .schemas import ObjectDetectionYoloDatasetSchema
from .object_detection import BaseObjectDetectionDataset

'''
The IDTReeS (Integrating Data science with Trees and Remote Sensing) dataset was created for a competition held in 2020. 
The aim was to delineate and classify individual tree crowns in NEON data. The data contains individual tree crown (ITC) 
spatial extent to be used for trainning. The classes are taxonomic species.
'''

class IDTReeSDataset(BaseObjectDetectionDataset):
    schema = ObjectDetectionYoloDatasetSchema
    name = "IDTReeS"
    url = "https://zenodo.org/records/3934932"

    # labels: 0 index is reserved for background
    labels = [None]

    def __init__(self, config):
        # now call the constructor to validate the schema and split the data
        super().__init__(config)
        self.data_dir = self.config.data_dir
        self.annotation_dir = self.config.annotation_dir
        self.csv_file = self.config.csv_file

        self.labels, self.annotations, self.data = self.load_dataset(
            self.data_dir, self.annotation_dir, self.csv_file
        )

    def __getitem__(self, index):
        for i in np.arange(10): # skip images without labels
            img_name = self.data[index]
            image = image_loader(os.path.join(self.data_dir, f"{img_name}.tif")) / 255.0
            img_h, img_w = image.shape[:2]

            annot = self.annotations

            # all the labels are in the same file, one needs to clip it to the corresponding image
            with rasterio.open(os.path.join(self.data_dir, f"{img_name}.tif")) as src:
                bounds = src.bounds  
                tiff_crs = src.crs   
                transform = src.transform  # affine transform: real-world coordinates to pixel coordinates
            tiff_bounds_geom = box(*bounds)

            # select polygons that intersect the image (TIFF) bounds
            gdf_within_tiff = annot[annot.intersects(tiff_bounds_geom)]

            # extract bounds for each polygon: returns a dataframe with minx, miny, maxx, maxy
            bounds_df = gdf_within_tiff.geometry.bounds
            # add these as columns to the intersecting dataframe
            gdf_within_tiff = gdf_within_tiff.join(bounds_df)

            boxes = []
            labels = []
            for _, row in gdf_within_tiff.iterrows():
                # transform real-world coordinates to pixel coordinates
                x1, y1 = ~transform * (row["minx"], row["miny"]) 
                x2, y2 = ~transform * (row["maxx"], row["maxy"])

                xmin, xmax = sorted([x1, x2])
                ymin, ymax = sorted([y1, y2])

                # clip to image size
                xmin = int(np.clip(xmin, 0, img_w))
                xmax = int(np.clip(xmax, 0, img_w))
                ymin = int(np.clip(ymin, 0, img_h))
                ymax = int(np.clip(ymax, 0, img_h))

                if xmax <= xmin or ymax <= ymin:
                    continue  # skip invalid boxes

                label = row["taxonID"]
                if label not in self.labels: # some instances have a nan label
                    continue
                labels.append(self.labels.index(label))
                boxes.append([xmin, ymin, xmax, ymax])

            # convert boxes into a torch.Tensor
            boxes = torch.as_tensor(boxes, dtype=torch.float32)

            # suppose all instances are not crowd
            iscrowd = torch.zeros((boxes.shape[0],), dtype=torch.int64)

            labels = torch.as_tensor(labels, dtype=torch.int64)

            target = {"boxes": boxes, "labels": labels, "iscrowd": iscrowd} 

            image_id = torch.tensor([index])
            target["image_id"] = image_id

            image, target = self.apply_transformations(image, target)

            if len(target["boxes"]) == 0:
                index = index + 1
            else:
                break

        return image, target

    def load_dataset(self, data_dir, annotation_dir, csv_file):
        labels = []
        data = []
        label_polygons = []

        # the shapefile contains the polygon geometries
        shapefiles = glob.glob(os.path.join(annotation_dir, "*.shp"))
        for shp in shapefiles:
            annot_file = gpd.read_file(shp)
            label_polygons.append(annot_file)
        label_polygons = gpd.GeoDataFrame(pd.concat(label_polygons, ignore_index=True))

        # the csv file contains the polygon labels
        df = pd.read_csv(csv_file)
        label_values = df[['indvdID','taxonID']]

        # add the label values (strings) to the dataframe with the geometries
        annotations = label_polygons.merge(label_values, on='indvdID', how='left')

        ids = os.listdir(data_dir)
        for img in ids:
            name = img[: img.rfind('.tif')]
            data.append(name)

        labels = ["ACPE","ACRU","ACSA3","AMLA","BETUL","CAGL8","CATO6","FAGR","GOLA","LITU","LYLU3","MAGNO","NYBI","NYSY","OXYDE",
                "PEPA37","PIEL","PIPA2","PINUS","PITA","PRSE2","QUAL","QUCO2","QUGE2","QUHE2","QULA2","QULA3","QUMO4","QUNI","QURU",
                "QUERC","ROPS","TSCA"]

        return labels, annotations, data
    
    
    def show_image(self, index, show_title=False):
        # plot the image and bboxes
        # Bounding boxes are defined as follows: x-min y-min width height
        img, target = self[index]
        fig = plt.figure(figsize=(10, 8))
        plt.subplot(1, 2, 1)
        plt.imshow(img)
        plt.axis("off")

        ax = plt.subplot(1, 2, 2)
        plt.imshow(img)
        plt.axis("off")
        for box, label in zip(target["boxes"], target["labels"]):
            x, y, width, height = box[0], box[1], box[2] - box[0], box[3] - box[1]
            rect = patches.Rectangle(
                (x, y), width, height, linewidth=2, edgecolor="violet", facecolor="none"
            )
            # Draw the bounding box on top of the image
            ax.add_patch(rect)
            ax.annotate(
                self.labels[label],
                (box[0] + 15, box[1] - 20),
                color="violet",
                fontsize=12,
                ha="center",
                va="center",
            )
        plt.tight_layout()
        plt.show()
        return fig

    def data_distribution_table(self):
        df = pd.DataFrame(self.annotations)
        df = df.drop(['id','geometry','indvdID'], axis=1)
        df_count = df.groupby("taxonID").value_counts()
        df_count = pd.DataFrame(df_count).reset_index()
        df_count.columns = ["Label", "Count"]
        
        return df_count

    def data_distribution_barchart(self, show_title=True):
        objects_count = self.data_distribution_table()
        fig, ax = plt.subplots(figsize=(12, 10))
        sns.barplot(y="Label", x="Count", data=objects_count, ax=ax)
        ax.set_title(
            "Number of instances for {}".format(self.get_name()), pad=20, fontsize=18
        )
        return fig