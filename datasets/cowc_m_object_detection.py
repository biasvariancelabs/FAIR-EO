import os
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch
import fnmatch
from PIL import Image
Image.MAX_IMAGE_PIXELS = None

from ..utils import image_loader
from .object_detection import BaseObjectDetectionDataset
from .schemas import FAIREOObjectDetectionYoloSchema

'''
The Cars Overhead With Context (COWC) data set is a large set of annotated cars from overhead. The COWC-M 
dataset extends the original COWC dataset by labeling the cars with types. Each car is now labeled as either 
Sedan, Pickup, Other or Unknown. It can be used for image classification or object detection tasks. This 
schema describes the object detection subset of the dataset.
'''

# Define sliding window size 
PATCH_H, PATCH_W = 1024, 1024

class COWCMObjectDetectionDataset(BaseObjectDetectionDataset):
    schema = FAIREOObjectDetectionYoloSchema
    name = "COWC-M for object detection dataset"
    url = "https://gdo152.llnl.gov/cowc/download/"

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
        img_id = self.data[index]
        image = image_loader(os.path.join(self.data_dir, img_id + '.jpg'))/ 255.0
        image = image[:,:,:3] # rgb image
        img_h, img_w = image.shape[:2]

        # annotation file
        annot_file_path = os.path.join(self.data_dir, img_id + '.txt')
        
        annot = open(annot_file_path, "r")
        annot = annot.readlines()
        boxes = []
        labels = []

        # box coordinates for txt files are extracted
        for annotation in annot:
            lines = annotation 
            elements = lines.split()
            # bounding box
            xmin = round((float(elements[1])-(float(elements[3])/2))*img_w)
            xmax = round((float(elements[1])+(float(elements[3])/2))*img_w)

            ymin = round((float(elements[2])-(float(elements[4])/2))*img_h)
            ymax = round((float(elements[2])+(float(elements[4])/2))*img_h)
           
            xmin = max(0, xmin)
            ymin = max(0, ymin)
            xmax = min(img_w, xmax)
            ymax = min(img_h, ymax)
      
            if xmax > xmin and ymax > ymin:
                labels.append(int(elements[0]))
                boxes.append([xmin, ymin, xmax, ymax])

        # convert boxes into a torch.Tensor
        boxes = torch.as_tensor(boxes, dtype=torch.float32)

        # suppose all instances are not crowd
        iscrowd = torch.zeros((boxes.shape[0],), dtype=torch.int64)

        labels = torch.as_tensor(labels, dtype=torch.int64)

        target = {"boxes": boxes, "labels": labels, "iscrowd": iscrowd} 
        
        # image_id
        image_id = torch.tensor([index])
        target["image_id"] = image_id

        return self.apply_transformations(image, target)

    def load_dataset(self, data_dir, annotation_dir, csv_file):
        annotations = []
        data = []
        labels = ["no-data","other","pickup","sedan","unknown"]
        split_list = []

        if self.csv_file: # train/val or test split
            with open(self.csv_file, "r",encoding="utf-8-sig") as f:
                for line in f:
                    split_list.append(line.split(',')[0])
        
        for aoi in os.listdir(data_dir):
            if os.path.isdir(os.path.join(data_dir, aoi)):
                for file in os.listdir(os.path.join(data_dir, aoi)):
                    if fnmatch.fnmatch(file, '*.txt'):
                        annot = open(os.path.join(data_dir, aoi, file), "r")
                        annot = annot.readlines()
                        for annotation in annot:
                            lines = annotation 
                            elements = lines.split()
                            annotations.append({"label": int(elements[0])})

                    elif fnmatch.fnmatch(file, '*check.jpg'):
                        continue
                    else:
                        if self.csv_file: # train/val or test split
                            if os.path.join(aoi,file) in split_list:
                                name = file[: file.rfind('.jpg')]
                                data.append(os.path.join(aoi,name))
                        else: # all images
                            name = file[: file.rfind('.jpg')]
                            data.append(os.path.join(aoi,name))

        return labels, annotations, data
    
    def __len__(self):
        return len(self.data)
    
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
        df = pd.DataFrame(self.annotations)
        df_count = df.groupby("label").value_counts()
        df_count = pd.DataFrame(df_count).reset_index()
        df_count = df_count.drop(['label'], axis=1)
        df_count.insert(0, "Label", ["other","pickup","sedan","unknown"], True)
        df_count.columns = ["Label", "Count"]
        
        return df_count

    def data_distribution_barchart(self):
        df_count = self.data_distribution_table()
        fig, ax = plt.subplots(figsize=(12, 10))
        sns.barplot(y="Label", x="Count", data=df_count, ax=ax)
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