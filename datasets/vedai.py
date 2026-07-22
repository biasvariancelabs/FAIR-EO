import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pandas as pd
import seaborn as sns
import torch
import numpy as np
import csv

from aitlas.utils import image_loader
from .schemas import FAIREOObjectDetectionYoloSchema
from aitlas.datasets.object_detection import BaseObjectDetectionDataset

'''
VEDAI is a dataset for Vehicle Detection in Aerial Imagery, provided as a tool to benchmark automatic target
recognition algorithms in unconstrained environments. The vehicles contained in the database, in addition of 
being small, exhibit different variabilities such as multiple orientations, lighting/shadowing changes, 
specularities or occlusions. Images are available in two sizes: 1024x1024 and 512x512. Vehicle instances are
annotated with oriented bounding boxes. For ease of use, we translate them into axis-aligned bounding boxes.
'''

class VEDAIDataset(BaseObjectDetectionDataset):
    schema = FAIREOObjectDetectionYoloSchema
    name = "Vehicle Detection in Aerial Imagery (VEDAI)"
    url = "https://downloads.greyc.fr/vedai/"
    labels = ["no_data","Small Land Vehicles","Large Land Vehicles","Truck","Pickup","Vans","Plane","Others","Camping Car","Car","Tractor","Boat"]
    
    # the original label values are mapped to consecutive integers
    label_map = {
        '0': 0,
        '1': 1,
        '2': 2,
        '4': 3,
        '5': 4,
        '7': 5,
        '8': 6,
        '9': 7,
        '10': 8,
        '11': 9,
        '23': 10,
        '31': 11
        } 

    def __init__(self, config):
        # now call the constructor to validate the schema and split the data
        super().__init__(config)
        self.data_dir = self.config.data_dir
        self.annotation_dir = self.config.annotation_dir
        self.csv_file = self.config.csv_file
        self.annotations, self.data = self.load_dataset(
            self.data_dir, self.annotation_dir, self.csv_file
        )


    def __getitem__(self, index):
        for i in np.arange(10):
            img_name = self.data[index]
            image = image_loader(os.path.join(self.data_dir, f"{img_name}_co.png"))/255
            img_h, img_w = image.shape[:2]

            #annotation file
            annot_file_path = os.path.join(self.annotation_dir, f"{img_name}.txt")

            boxes = []
            labels = []
            with open(annot_file_path, "r") as f:
                for line in f:
                    parts = line.strip().split()

                    x_coords = list(map(float, parts[6:10]))  # x1,x2,x3,x4
                    y_coords = list(map(float, parts[10:14]))  # y1,y2,y3,y4

                    if len(x_coords)==0 and len(y_coords)==0:
                        break

                    # Combine into corner points
                    corners = np.array(list(zip(x_coords, y_coords)))

                    # Compute axis-aligned bounding box from corners
                    xmin, ymin = corners.min(axis=0)
                    xmax, ymax = corners.max(axis=0)

                    # Ensure the box is within image boundaries
                    xmin, xmax = max(0, xmin), min(img_w, xmax)
                    ymin, ymax = max(0, ymin), min(img_h, ymax)

                    label_id_initial = parts[3]
                    # attribute consecutive integer labels with label_map
                    label_id = self.label_map.get(label_id_initial, label_id_initial) 

                    if xmax > xmin and ymax > ymin:
                        labels.append(int(label_id))
                        boxes.append([xmin, ymin, xmax, ymax])

            # convert boxes into a torch.Tensor
            boxes = torch.as_tensor(boxes, dtype=torch.float32)

            # suppose all instances are not crowd
            iscrowd = torch.zeros((boxes.shape[0],), dtype=torch.int64)

            labels = torch.as_tensor(labels, dtype=torch.int64)

            target = {"boxes": boxes, "labels": labels, "iscrowd": iscrowd}

            if len(target["boxes"])==0:
                index = index +1
            else:
                break

        return self.apply_transformations(image, target)


    def load_dataset(self, data_dir, annotation_dir, csv_file):
        annotations = []
        data = []
        split_list =[]
        ids = list(os.path.splitext(f)[0] for f in os.listdir(annotation_dir) if f.endswith('.txt') and f.startswith('00'))

        if csv_file:
            with open(csv_file, mode ='r')as file:
                csvFile = csv.reader(file)
                for lines in csvFile:
                    split_list.append(lines[0][14 : 22])
        else:
            split_list = ids

        for annot in split_list:
            data.append(annot)
            with open(os.path.join(annotation_dir, f'{annot}.txt'), "r") as f:
                for line in f:
                    parts = line.strip().split()
                    label_id_initial = parts[3]
                    label_id = self.label_map.get(label_id_initial, label_id_initial) 
                    annotations.append({"label": int(label_id)})

        return annotations, data


    def data_distribution_table(self):
        label_to_name = {
            1:  "Small Land Vehicles",
            2:  "Large Land Vehicles",
            3:  "Truck",
            4:  "Pickup",
            5:  "Vans",
            6:  "Plane",
            7:  "Others",
            8: "Camping Car",
            9: "Car",
            10: "Tractor",
            11: "Boat"
        }
        df = pd.DataFrame(self.annotations)
        df_count = df['label'].value_counts().sort_index().reset_index()
        df_count.columns = ["Label", "Count"]
        df_count['Label'] = df_count['Label'].map(label_to_name)
        return df_count


    def data_distribution_barchart(self, show_title=True):
        objects_count = self.data_distribution_table()
        fig, ax = plt.subplots(figsize=(12, 10))
        sns.barplot(y="Label", x="Count", data=objects_count, ax=ax)
        ax.set_title(
            "Number of instances for {}".format(self.get_name()), pad=20, fontsize=18
        )
        return fig
    

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