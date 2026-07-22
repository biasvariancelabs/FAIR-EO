import os
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import csv 
import fnmatch
from PIL import Image
Image.MAX_IMAGE_PIXELS = None

from aitlas.utils import image_loader
from aitlas.datasets.object_detection import BaseObjectDetectionDataset
from .schemas import FAIREOObjectDetectionYoloSchema

'''
The Cars Overhead With Context (COWC) data set is a large set of annotated cars from overhead. 
All cars in the annotated images have a dot placed at their center. Negative samples are also 
included in the dataset. It can be used for image classification or object detection/counting 
tasks. This dataloader is written for the object detection subset of the dataset.
'''

# Define sliding window size 
PATCH_H, PATCH_W = 1024, 1024

class COWCObjectDetectionDataset(BaseObjectDetectionDataset):
    schema = FAIREOObjectDetectionYoloSchema
    name = "COWC for object detection dataset"
    url = "https://gdo152.llnl.gov/cowc/download/"

    # labels: 0 index is reserved for background
    labels = [None]

    def __init__(self, config):
        # now call the constructor to validate the schema and split the data
        super().__init__(config)
        self.data_dir = self.config.data_dir
        self.annotation_dir = self.config.annotation_dir
        self.csv_file = self.config.csv_file

        self.labels, self.data, self.image_patches, self.positives, self.negatives = self.load_dataset(
            self.data_dir, self.annotation_dir, self.csv_file
        )

    def __getitem__(self, index):
        for i in np.arange(100): # allows to skip images without labels
            image = self.image_patches[index]/ 255.0
            image = image[:,:,:3]
            img_h, img_w = image.shape[:2]

            # annotation files
            positives = self.positives[index]
            negatives = self.negatives[index]

            # extract the positive and negative pixels (red and blue respectively)
            positive_pixels = (positives[:, :, 0] == 255) & (positives[:, :, 1] == 0) & (positives[:, :, 2] == 0) & (positives[:, :, 3] == 255)
            negative_pixels = (negatives[:, :, 0] == 0) & (negatives[:, :, 1] == 0) & (negatives[:, :, 2] == 255) & (negatives[:, :, 3] == 255)
            
            boxes = []
            labels = []
            
            # Get the coordinates of positive pixels
            ys_pos, xs_pos = np.where(positive_pixels)

            for x, y in zip(xs_pos, ys_pos):
                # annotations are points centered on the cars, we create a small bounding box around them
                xmin = int(x) - 10
                xmax = int(x) + 10
                
                ymin = int(y) - 10
                ymax = int(y) + 10

                if xmax<=img_w and ymax<=img_h and xmin>=0 and ymin >=0:
                    label_value = 1
                    labels.append(int(label_value))
                    boxes.append([xmin, ymin, xmax, ymax])

            # Get the coordinates of negative pixels
            ys_neg, xs_neg = np.where(negative_pixels)

            for x, y in zip(xs_neg, ys_neg):
                # annotations are points centered on the negatives, create a small bounding box
                xmin = int(x) - 10
                xmax = int(x) + 10
                
                ymin = int(y) - 10
                ymax = int(y) + 10

                if xmax<=img_w and ymax<=img_h and xmin>=0 and ymin >=0:
                    label_value = 0
                    labels.append(int(label_value))
                    boxes.append([xmin, ymin, xmax, ymax])

            # convert boxes into a torch.Tensor
            boxes = torch.as_tensor(boxes, dtype=torch.float32)

            # suppose all instances are not crowd
            iscrowd = torch.zeros((boxes.shape[0],), dtype=torch.int64)

            labels = torch.as_tensor(labels, dtype=torch.int64)

            target = {"boxes": boxes, "labels": labels, "iscrowd": iscrowd} 
            
            image_id = torch.tensor([index])
            target["image_id"] = image_id

            if len(target['boxes']) == 0:
                index = index + 1
            elif len(target['boxes']) != len(target['labels']):
                index = index + 1
            else:
                break

        return self.apply_transformations(image, target)

    def load_dataset(self, data_dir, annotation_dir, csv_file):
        labels = []
        data = []
        positive_paths =[]
        negative_paths =[]
        images = []
        labels = ["negative","car"]
        stride = PATCH_H
        self.image_patches = []
        self.positives = []
        self.negatives = []
        
        for aoi in os.listdir(data_dir):
            if os.path.isdir(os.path.join(data_dir, aoi)):
                for file in os.listdir(os.path.join(data_dir, aoi)):
                    # ignore txt and xcf files
                    if fnmatch.fnmatch(file, '*.txt'):
                        continue
                    elif fnmatch.fnmatch(file, '*.xcf'):
                        continue
                    elif fnmatch.fnmatch(file, '*_Annotated_Cars.png'):
                        positive_paths.append(os.path.join(data_dir, aoi, file))
                    elif fnmatch.fnmatch(file, '*_Annotated_Negatives.png'):
                        negative_paths.append(os.path.join(data_dir, aoi, file))
                    else:
                        images.append(os.path.join(data_dir, aoi, file))
                        data.append(os.path.join(aoi,file))
     
        # create patches of the original 32 images to increase the amount of training data and decrease their large size
        for img_name in data:
            image = image_loader(os.path.join(self.data_dir, img_name))
            positives = image_loader(os.path.join(self.data_dir, img_name[:-4]+'_Annotated_Cars.png'))
            negatives = image_loader(os.path.join(self.data_dir, img_name[:-4]+'_Annotated_Negatives.png'))
                        
            img_h, img_w = image.shape[:2]
            for y in range(0, img_h - PATCH_H + 1, stride):
                for x in range(0, img_w - PATCH_W + 1, stride):
                    if y + PATCH_H <= img_h and x + PATCH_W <= img_w:
                        self.image_patches.append(image[y:y+PATCH_H, x:x+PATCH_W])
                        self.positives.append(positives[y:y+PATCH_H, x:x+PATCH_W])
                        self.negatives.append(negatives[y:y+PATCH_H, x:x+PATCH_W])

        return labels, data, self.image_patches, self.positives, self.negatives
    
    def __len__(self):
        return len(self.image_patches)
    
    def apply_transformations(self, image, target):
        if self.joint_transform:
            image, target = self.joint_transform((image, target))
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            target = self.target_transform(target)
        return image, target

    def get_labels(self):
        return self.labels

    def data_distribution_table(self):
        label_dist = {key: 0 for key in self.labels}
        for i in np.arange(len(self.data)):
            img, target = self.__getitem__(i)
            for label_value in target["labels"]:
                label_value = int(label_value)
                for index, label in enumerate(self.labels):
                    if label_value == index:
                        label_dist[self.labels[index]] += 1

        label_count = pd.DataFrame.from_dict(label_dist, orient='index')
        label_count.columns = ["Count"]
        label_count = label_count.astype(float)
        
        return label_count

    def data_distribution_barchart(self, show_title=True):
        label_count = self.data_distribution_table()
        fig, ax = plt.subplots(figsize=(12, 10))
        sns.barplot(data=label_count, x=label_count.index, y='Count', ax=ax)
        if show_title:
            ax.set_title(
                "Labels distribution for {}".format(self.get_name()), pad=20, fontsize=18
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
            if label == 0:
                rect = patches.Rectangle(
                    (x, y), width, height, linewidth=2, edgecolor="blue", facecolor="none"
                )
                # Draw the negative bounding box on top of the image
                ax.add_patch(rect)
            if label == 1:
                rect = patches.Rectangle(
                    (x, y), width, height, linewidth=2, edgecolor="red", facecolor="none"
                )
                # Draw the positive bounding box on top of the image
                ax.add_patch(rect)
        plt.tight_layout()
        plt.show()
        return fig