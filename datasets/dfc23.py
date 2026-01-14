import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import os
import json
import torch
import matplotlib.patches as patches
from collections import defaultdict

from .object_detection import BaseObjectDetectionDataset
from .schemas import FAIREOObjectDetectionCocoSchema
from ..utils import image_loader


"""
DFC23 establishes a large-scale, fine-grained, and multi-modal benchmark for the classification of building roof types. It consists of two challenging competition 
tracks investigating the fusion of optical and SAR data, while focusing on roof type classification and building height estimation, respectively.
"""


class DFC23Dataset(BaseObjectDetectionDataset):

    schema = FAIREOObjectDetectionCocoSchema
    url = "https://ieee-dataport.org/competitions/2023-ieee-grss-data-fusion-contest-large-scale-fine-grained-building-classification"
    labels = ["no-data","flat","gable","gambrel","row","multiple eave","hipped v1","hipped v2","mansard","pyramid","arched","revolved","other"]
    color_mapping = [[0,0,0],[0,0,255],[255,0,0],[0,255,0],[255,127,80],[0,255,255],[153,0,0],[255,0,0],[255,255,0],[255,0,255],[120,120,120],[0,120,0],[0,0,120]]
    name = "DFC23"

    def __init__(self, config):
        # now call the constructor to validate the schema
        super().__init__(config)
        self.data_dir = self.config.data_dir
        self.annotation_dir = self.config.annotation_dir
        self.csv_file = self.config.csv_file
        self.selection = self.config.selection
        self.images, self.annotations = self.load_dataset(self.data_dir, self.annotation_dir, self.csv_file)

    def __getitem__(self, index):
        img_id = self.images[index][2]

        rgb_image = image_loader(self.images[index][0])/255
        sar_image = image_loader(self.images[index][1])/255
        if self.selection == 'rgb':
            image = rgb_image
        elif self.selection == 'sar':
            image = sar_image
        elif self.selection == 'rgb & sar':
            image = np.dstack((rgb_image, sar_image))
            #normalize image for better visualization
            img_min, img_max = image.min(), image.max()
            image = (image - img_min) / (img_max - img_min + 1e-8)
            
        boxes = []
        labels = []
        with open(self.annotation_dir, "r") as file:
            data = json.load(file)
            for annot in data['annotations']:
                if annot["image_id"] == img_id:
                    bbox = annot['bbox']
                    # bounding box
                    xmin = int(bbox[0])
                    ymin = int(bbox[1])

                    xmax = int(bbox[0] + bbox[2])
                    ymax = int(bbox[1] + bbox[3])

                    if xmax > xmin and ymax > ymin:
                        labels.append(int(annot['category_id']))
                        boxes.append([xmin, ymin, xmax, ymax])

        # convert boxes into a torch.Tensor
        boxes = torch.as_tensor(boxes, dtype=torch.float32)

        labels = torch.as_tensor(labels, dtype=torch.int64)

        target = {"boxes": boxes, "labels": labels}

        return self.apply_transformations(image, target)
    

    def load_dataset(self, data_dir, annotation_dir, csv_file):
        images = []
        annotations = []

        ids = os.listdir(os.path.join(data_dir, "rgb"))

        with open(annotation_dir, "r") as file:
            data = json.load(file)
        
        # index annotations by image_id to keep only images which contain at least one bbox
        annots_by_image = defaultdict(list)
        for annot in data["annotations"]:
            if annot["bbox"]:  # bbox exists and not empty
                annots_by_image[annot["image_id"]].append(annot)

        for image in data["images"]:
            filename = image["file_name"]
            img_id = image["id"]

            if filename in ids and img_id in annots_by_image:
                rgb_image = os.path.join(os.path.join(data_dir, "rgb"), filename)
                sar_image = os.path.join(os.path.join(data_dir, "sar"), filename)

                images.append((rgb_image, sar_image, img_id))

        for annot in data['annotations']:
            annotations.append({"label": int(annot['category_id'])})

        return images, annotations
    
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

    def show_image(self, index, show_title=False):
        # plot the image and bboxes
        # Bounding boxes are defined as follows: x-min y-min width height
        img, target = self[index]
        fig = plt.figure(figsize=(10, 8))
        if self.selection == 'rgb & sar':
            plt.subplot(1, 3, 1)
            plt.imshow(img[:,:,:3])
            plt.title('RGB image')
            plt.axis("off")

            ax = plt.subplot(1, 3, 2)
            plt.imshow(img[:,:,3])
            plt.title('SAR image')
            plt.axis("off")

            ax = plt.subplot(1, 3, 3)
            plt.imshow(img[:,:,:3])
            plt.axis("off")

        else:
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
        df_count = df.groupby("label").size().reset_index(name="Count")
        label_map = dict(enumerate(self.labels))
        df_count.columns = ["Label", "Count"]
        df_count["Label"] = df_count["Label"].map(label_map)
        return df_count
    
    def data_distribution_barchart(self, show_title=True):
        objects_count = self.data_distribution_table()
        fig, ax = plt.subplots(figsize=(12, 12))
        sns.barplot(data=objects_count, x="Label", y="Count", ax=ax)
        fig.autofmt_xdate()
        ax.set_title(
            "Number of instances for {} dataset".format(self.get_name()), pad=20, fontsize=18
        )
        return fig