import json
import os
import torch
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from ..utils import image_loader
from .schemas import FAIREOObjectDetectionYoloSchema
from .object_detection import BaseObjectDetectionDataset


"""
The DIUx xView: Objects in Context in Overhead Imagery is a large-scale dataset designed to advance 
object detection techniques and research in overhead object detection. The dataset was collected using 
WorldView-3 satellites, providing high-resolution imagery.
"""


class xviewDataset(BaseObjectDetectionDataset):
    schema = FAIREOObjectDetectionYoloSchema
    name = "xView"
    url = "https://challenge.xviewdataset.org/challenge-description"

    labels = [
        "Fixed-wing Aircraft",
        "Small Aircraft",
        "Passenger/Cargo Plane",
        "Helicopter",
        "Passenger Vehicle",
        "Small Car",
        "Bus",
        "Pickup Truck",
        "Utility Truck",
        "Truck",
        "Cargo Truck",
        "Truck Tractor w/ Box Trailer",
        "Truck Tractor",
        "Trailer",
        "Truck Tractor w/ Flatbed Trailer",
        "Truck Tractor w/ Liquid Tank",
        "Crane Truck",
        "Railway Vehicle",
        "Passenger Car",
        "Cargo/Container Car",
        "Flat Car",
        "Tank Car",
        "Locomotive",
        "Maritime Vessel",
        "Motorboat",
        "Sailboat",
        "Tugboat",
        "Barge",
        "Fishing Vessel",
        "Ferry",
        "Yacht",
        "Container Ship",
        "Oil Tanker",
        "Engineering Vehicle",
        "Tower Crane",
        "Container Crane",
        "Reach Stacker",
        "Straddle Carrier",
        "Mobile Crane",
        "Dump Truck",
        "Haul Truck",
        "Tractor",
        "Front Loader/Bulldozer",
        "Excavator",
        "Cement Mixer",
        "Ground Grader",
        "Hut/Tent",
        "Shed",
        "Building",
        "Aircraft Hangar",
        "Damaged/Demolished Building",
        "Facility",
        "Construction Site",
        "Vehicle Lot",
        "Helipad",
        "Storage Tank",
        "Shipping Container Lot",
        "Shipping Container",
        "Pylon",
        "Tower",
        "Unknown",
        "Unknown",
    ]

    # List of label IDs (keys)
    label_ids = [
        11,
        12,
        13,
        15,
        17,
        18,
        19,
        20,
        21,
        23,
        24,
        25,
        26,
        27,
        28,
        29,
        32,
        33,
        34,
        35,
        36,
        37,
        38,
        40,
        41,
        42,
        44,
        45,
        47,
        49,
        50,
        51,
        52,
        53,
        54,
        55,
        56,
        57,
        59,
        60,
        61,
        62,
        63,
        64,
        65,
        66,
        71,
        72,
        73,
        74,
        76,
        77,
        79,
        83,
        84,
        86,
        89,
        91,
        93,
        94,
        82,
        75,
    ]

    def __init__(self, config):
        # now call the constructor to validate the schema and split the data
        super().__init__(config)
        self.data_dir = self.config.data_dir
        self.annotation_dir = self.config.annotation_dir
        self.annotations, self.data = self.load_dataset(
            self.data_dir, self.annotation_dir
        )

    def __getitem__(self, index):
        for i in np.arange(10): # ignore cases where the amount of boxes and labels differ
            img_name = self.data[index]
            image = image_loader(os.path.join(self.data_dir, f"{img_name}.tif"))/255
            image = image.transpose(1, 2, 0)
            height, width = image.shape[:2]

            boxes = []
            labels = []

            for feat in self._jsonfile["features"]:
                id = feat["properties"]["image_id"]

                if id != f"{img_name}.tif":
                    continue

                bbox = feat["properties"]["bounds_imcoords"]
                xmin, ymin, xmax, ymax = bbox.split(",")
                xmin, ymin = max(0, int(xmin)), max(0, int(ymin))
                xmax, ymax = min(width, int(xmax)), min(height, int(ymax))
                if xmax > xmin and ymax > ymin:
                    bbox = [xmin, ymin, xmax, ymax]
                    boxes.append(bbox)
                    label_key = int(feat["properties"]["type_id"])
                    labels.append(self.label_ids.index(label_key))

            # convert boxes into a torch.Tensor
            boxes = torch.as_tensor(boxes, dtype=torch.float32)

            # suppose all instances are not crowd
            iscrowd = torch.zeros((boxes.shape[0],), dtype=torch.int64)

            labels = torch.as_tensor(labels, dtype=torch.int64)

            target = {"boxes": boxes, "labels": labels, "iscrowd": iscrowd}

            image, target = self.apply_transformations(image, target)

            if len(target['boxes']) != len(target['labels']):
                index = index + 1 
            else:
                break

        return image, target

    def load_dataset(self, data_dir, annotation_dir):
        annotations = []
        ids = [img_name.split(".")[0] for img_name in os.listdir(data_dir)]

        with open(annotation_dir, "r", encoding="utf-8") as f:
            self._jsonfile = json.load(f)

        for feat in self._jsonfile["features"]:
            label_id = int(feat["properties"]["type_id"])
            id = self.label_ids.index(label_id)
            annotations.append({"label": id})

        return annotations, ids

    def data_distribution_table(self):
        df = pd.DataFrame(self.annotations)
        df_count = df.groupby("label").size().reset_index(name="Count")
        df_count["Label"] = [self.labels[i] for i in df_count["label"]]
        return df_count[["Label", "Count"]]

    def data_distribution_barchart(self, show_title=True):
        objects_count = self.data_distribution_table()
        fig, ax = plt.subplots(figsize=(12, 10))
        sns.barplot(y="Label", x="Count", data=objects_count, ax=ax)
        ax.set_title(
            "Number of instances for {}".format(self.get_name()), pad=20, fontsize=18
        )
        return fig