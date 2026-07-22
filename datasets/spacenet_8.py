import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Patch
from shapely import wkt
from skimage.draw import line, polygon

from aitlas.base import BaseDataset
from .schemas import SpaceNet8DatasetSchema
from aitlas.utils import image_loader


"""
SpaceNet 8 is a multi-task remote sensing dataset combining building footprint detection, road network extraction, 
and flood attribution using pre- and post-event imagery covering three AOIs. The dataset contains tiled GeoTIFF 
imagery (pre- and post-event), GeoJSON vector annotations for buildings and roads, road speed attributes, and 
flood attributes derived from post-event imagery.
"""

class SpaceNet8Dataset(BaseDataset):

    url = ["https://medium.com/@SpaceNet_Project/announcing-spacenet-8-flood-detection-challenge-using-multiclass-segmentation-49bac6ae300"]
    name = "SpaceNet 8"
    schema = SpaceNet8DatasetSchema
    labels = [
        "Other",
        "non-flooded building",
        "flooded building",
        "non-flooded road",
        "flooded road"
    ]
    color_mapping =  [
        [255, 255, 255],
        [26, 169, 45],
        [44, 208, 240],
        [97, 249, 73],
        [44, 77, 240]
    ]


    def __init__(self, config):
        # now call the constructor to validate the schema and split the data
        super().__init__(config)
        self.data_dir = self.config.data_dir
        self.mode = self.config.mode
        self.load_dataset(self.data_dir)

    def __getitem__(self, index):
        # images are pre- and post-event pairs
        pre, post = self.data[index]
        img_id = pre.split('/')[2].split('.')[0]
        AOI = pre.split('/')[0]
        
        post_img = image_loader(os.path.join(self.data_dir, post))
        pre_img = image_loader(os.path.join(self.data_dir, pre))
        pre_img = pre_img.transpose(1, 2, 0)

        img_w, img_h = pre_img.shape[:2]

        mask = np.zeros((img_w,img_h),dtype=np.int64)

        # annotations are polygon pixel coordinates
        annotations_csv = os.path.join(self.data_dir, AOI, f'{AOI}_reference.csv')
        df = pd.read_csv(annotations_csv)
        image_rows = df[df['ImageId'].str.contains(f'^{img_id}')]

        for index, row in image_rows.iterrows():
            obj_type = row['Object']
            flooded = row['Flooded']
            flooded = True if flooded == "True" else False
            wkt_geom = row['Wkt_Pix']

            # Skip invalid or empty geometries
            if pd.isna(wkt_geom) or 'EMPTY' in str(wkt_geom):
                continue

            try:
                geometry = wkt.loads(wkt_geom)
            except Exception as e:
                print(f"Warning: Failed to parse WKT geometry: {wkt_geom}")
                continue

            if obj_type == "Road":
                coords = list(geometry.coords)
                mask = self.add_road(mask, coords, flooded)
            elif obj_type == "Building":
                coords = list(geometry.exterior.coords)
                mask = self.add_building(mask, coords, flooded)

        if self.mode == "post-event":
            num_classes = len(self.labels)
            mask = np.eye(num_classes)[mask]

            return self.apply_transformations(post_img,mask)
        
        return self.apply_transformations((post_img,pre_img),mask)


    def load_dataset(self, data_dir):
        pre_images = []
        post_images = []
        AOIs = ["Germany_Training_Public","Louisiana-East_Training_Public"]

        for AOI in AOIs:
            image_mapping_csv = os.path.join(data_dir, AOI,f'{AOI}_label_image_mapping.csv')
            df = pd.read_csv(image_mapping_csv)
            pre_images.append([f"{AOI}/PRE-event/{img}" for img in df['pre-event image']])
            post_images.append([f"{AOI}/POST-event/{img}" for img in df['post-event image 1']])

        pre_images = [path for AOI_list in pre_images for path in AOI_list]
        post_images = [path for AOI_list in post_images for path in AOI_list]

        self.data = []
        for pre, post in zip(pre_images, post_images):
            self.data.append((pre, post))


    def __len__(self):
        return len(self.data)
    
    def get_labels(self):
        return self.labels

    def add_road(self, mask:np.ndarray, coords:list, flooded:bool)->np.ndarray:
        """
        Creates a line from coordinates
        """
        pixel_coords = [(int(round(x)), int(round(y))) for x, y in coords]

        # Determine class label: 3 for non-flooded road, 4 for flooded road
        road_class = 4 if flooded else 3

        for i in range(len(pixel_coords) - 1):
            x0, y0 = pixel_coords[i]
            x1, y1 = pixel_coords[i + 1]

            # Get line coordinates between points
            rr, cc = line(y0, x0, y1, x1)

            # Ensure coordinates are within mask bounds
            valid_coords = (rr >= 0) & (rr < mask.shape[0]) & (cc >= 0) & (cc < mask.shape[1])
            rr, cc = rr[valid_coords], cc[valid_coords]

            # Set road pixels to appropriate class
            mask[rr, cc] = road_class

        return mask

    def add_building(self, mask:np.ndarray, coords:list, flooded:bool)->np.ndarray:
        """
        Fills a building polygon with the corresponding label pixels
        Uses a set of polygon coordinates as input
        """
        pixel_coords = [(int(round(x)), int(round(y))) for x, y in coords]

        # Check if we have enough coordinates for a valid polygon
        if len(pixel_coords) < 4:
            return mask

        # Determine class label: 1 for non-flooded building, 2 for flooded building
        building_class = 2 if flooded else 1

        # Extract x and y coordinates for polygon filling
        x_coords = [coord[0] for coord in pixel_coords[:-1]]
        y_coords = [coord[1] for coord in pixel_coords[:-1]]

        rr, cc = polygon(y_coords, x_coords, shape=mask.shape)
        mask[rr, cc] = building_class

        return mask

    def show_image(self, index, show_title=False):
        data = self[index]

        # Check if mode is pairs (both pre- and post-event images)
        if self.mode == "pairs":
            (post_img, pre_img), mask = data

            # Create mask visualization
            img_mask = np.zeros([mask.shape[0], mask.shape[1], 3], np.uint8)
            legend_elements = []
            for i, label in enumerate(self.labels):
                legend_elements.append(
                    Patch(
                        facecolor=tuple([x / 255 for x in self.color_mapping[i]]),
                        label=self.labels[i],
                    )
                )
                img_mask[np.where(mask == i)] = self.color_mapping[i]

            fig = plt.figure(figsize=(15, 6))

            # Create subplot with space for legend on top
            plt.subplots_adjust(top=0.85, bottom=0.1)

            # Add legend
            fig.legend(handles=legend_elements, bbox_to_anchor=(0.1, 0.95, 0.8, 0.05), ncol=3,
                        loc='center', prop={'size': 10})

            plt.subplot(1, 3, 1)
            plt.imshow(pre_img)
            plt.title("Pre-event")
            plt.axis("off")

            plt.subplot(1, 3, 2)
            plt.imshow(post_img)
            plt.title("Post-event")
            plt.axis("off")

            plt.subplot(1, 3, 3)
            plt.imshow(img_mask)
            plt.title("Mask")
            plt.axis("off")

        else:
            # Mode 'post-event': only post-event image
            img, mask = data
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

            fig = plt.figure(figsize=(10, 9))

            # Create subplot with space for legend on top
            plt.subplots_adjust(top=0.85, bottom=0.1)

            # Add legend
            fig.legend(handles=legend_elements, bbox_to_anchor=(0.1, 0.95, 0.8, 0.05), ncol=3,
                       loc='center', prop={'size': 12})
            plt.subplot(1, 2, 1)
            plt.imshow(img)
            plt.axis("off")
            plt.subplot(1, 2, 2)
            plt.imshow(img_mask)
            plt.axis("off")

        fig.tight_layout()
        plt.show()
        return fig


    def apply_transformations(self, image, mask):
        if self.joint_transform:
            image, mask = self.joint_transform((image, mask))
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            mask = self.target_transform(mask)
        return image, mask


    def data_distribution_table(self):
        # Initialize counters
        stats = {
            'non-flooded building': 0,
            'flooded building': 0,
            'non-flooded road': 0,
            'flooded road': 0
        }

        # Read annotations CSVs
        AOIs = ["Germany_Training_Public","Louisiana-East_Training_Public"]
        for AOI in AOIs:
            annotations_csv= os.path.join(self.data_dir, AOI, f'{AOI}_reference.csv')
            df = pd.read_csv(annotations_csv)

            for _, row in df.iterrows():
                obj_type = row['Object']
                flooded = row['Flooded']
                flooded = True if flooded == "True" else False
                wkt_geom = row['Wkt_Pix']

                # Skip empty geometries
                if pd.isna(wkt_geom) or 'EMPTY' in str(wkt_geom):
                    continue

                try:
                    geometry = wkt.loads(wkt_geom)

                    # Calculate approximate area/length in pixels
                    if obj_type == "Building":
                        # For buildings, calculate polygon area
                        area = geometry.area
                        label = 'flooded building' if flooded else 'non-flooded building'
                        stats[label] += int(area)

                    elif obj_type == "Road":
                        # For roads, calculate buffered line area (width ~4 pixels)
                        length = geometry.length
                        road_area = length * 4  # Approximate road width
                        label = 'flooded road' if flooded else 'non-flooded road'
                        stats[label] += int(road_area)

                except Exception as e:
                    continue

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