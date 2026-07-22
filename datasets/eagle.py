import os
from xml.etree import ElementTree as et
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch

from aitlas.utils import image_loader
from .schemas import FAIREOObjectDetectionPascalSchema
from aitlas.datasets.object_detection import BaseObjectDetectionDataset

'''
EAGLE stands for oriEnted vehicle detection using Aerial imaGery in real-worLd scEnarios. 
The EAGLE dataset is used to detect vehicles of different classes including vehicle orientation 
based on aerial images.
'''


class EAGLEDataset(BaseObjectDetectionDataset):
    schema = FAIREOObjectDetectionPascalSchema
    name = "EAGLE"
    url = "https://www.dlr.de/en/eoc/about-us/remote-sensing-technology-institute/photogrammetry-and-image-analysis/public-datasets/eagle"

    # labels: 0 index is reserved for background
    labels = [None]

    def __init__(self, config):
        # now call the constructor to validate the schema and split the data
        super().__init__(config)
        self.image_dir = self.config.image_dir
        self.annotations_dir = self.config.annotations_dir

        self.labels, self.annotations, self.data = self.load_dataset(
            self.annotations_dir
        )

    def __getitem__(self, index):
        img_name = self.data[index]
        try:
            image = image_loader(os.path.join(self.image_dir, f"{img_name}.jpg")) / 255.0
        except:
            image = image_loader(os.path.join(self.image_dir, f"{img_name}.JPG")) / 255.0

        img_h, img_w = image.shape[:2]

        # annotation file
        annot_file_path = os.path.join(self.annotations_dir, f"{img_name}.xml")
        boxes = []
        labels = []
        tree = et.parse(annot_file_path)
        root = tree.getroot()

        # box coordinates for xml files are extracted
        for member in root.findall("object"):
            # bounding box
            try:
                xmin = round( (float(member.find("bndbox").find("cx").text)) - (float(member.find("bndbox").find("h").text)) )
                xmax = round( (float(member.find("bndbox").find("cx").text)) + (float(member.find("bndbox").find("h").text)) )

                ymin = round( (float(member.find("bndbox").find("cy").text)) - (float(member.find("bndbox").find("w").text)) )
                ymax = round( (float(member.find("bndbox").find("cy").text)) + (float(member.find("bndbox").find("w").text)) )

                if xmin < 0:
                    xmin = 0
                if ymin < 0:
                    ymin = 0
                if xmax > img_w:
                    xmax = img_w
                if ymax > img_h:
                    ymax = img_h

                if xmax > xmin and ymax > ymin:
                    labels.append(self.labels.index(member.find("type").text))
                    boxes.append([xmin, ymin, xmax, ymax])
            except:
                None #The object doesn't contain a bounding box

        # convert boxes into a torch.Tensor
        boxes = torch.as_tensor(boxes, dtype=torch.float32)

        # suppose all instances are not crowd
        iscrowd = torch.zeros((boxes.shape[0],), dtype=torch.int64)

        labels = torch.as_tensor(labels, dtype=torch.int64)

        target = {"boxes": boxes, "labels": labels, "iscrowd": iscrowd}
    
        image_id = torch.tensor([index])
        target["image_id"] = image_id

        return self.apply_transformations(image, target)

    def load_dataset(self, annotations_dir):
        labels = []
        annotations = []
        data = []
        for img in os.listdir(annotations_dir):
            annot_file_path = os.path.join(annotations_dir, img)

            tree = et.parse(annot_file_path)
            root = tree.getroot()

            # box coordinates for xml files are extracted
            has_box = False
            for member in root.findall("object"):
                try:
                    label = member.find("type").text.strip()
                    labels.append(label)

                    xmin = round( (float(member.find("bndbox").find("cx").text)) - (float(member.find("bndbox").find("h").text)) )
                    xmax = round( (float(member.find("bndbox").find("cx").text)) + (float(member.find("bndbox").find("h").text)) )

                    ymin = round( (float(member.find("bndbox").find("cy").text)) - (float(member.find("bndbox").find("w").text)) )
                    ymax = round( (float(member.find("bndbox").find("cy").text)) + (float(member.find("bndbox").find("w").text)) )

                    if xmax > xmin and ymax > ymin:
                        has_box = True
                        annotations.append({"image_id": img, "label": label})
                except:
                    None #The object doesn't contain a bounding box

            name = img[: img.rfind('.xml')]
            data.append(name)

        labels = ["no-data","vehicle"]

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
                (x, y), width, height, linewidth=2, edgecolor="white", facecolor="none"
            )

            # Draw the bounding box on top of the image
            ax.add_patch(rect)
            ax.annotate(
                self.labels[label],
                (box[0] + 15, box[1] - 20),
                color="white",
                fontsize=12,
                ha="center",
                va="center",
            )
        plt.tight_layout()
        plt.show()
        return fig

    def data_distribution_table(self):
        df = pd.DataFrame(self.annotations)
        df_count = df.groupby("label").count()
        df_count = df_count.iloc[::-1,:].reset_index()
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
