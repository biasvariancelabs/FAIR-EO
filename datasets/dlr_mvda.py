import os
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch
import math
import numpy as np
import matplotlib.patches as patches
from PIL import Image

from aitlas.utils import image_loader
from .schemas import FAIREOObjectDetectionYoloSchema
from aitlas.datasets.object_detection import BaseObjectDetectionDataset

'''
DLR-MVDA is an open aerial image dataset with annotated vehicles captured from an airplane over the city of Munich, Germany.
In this dataloader images from the dataset (which have a size larger than 5000 x 3000) are croped down to 256 x 256 to increase 
the amount of training data. The oriented bounding boxes are translated to axis-aligned for ease of use.
'''

IMG_W, IMG_H = 256, 256

class DLRMVDADAtaset(BaseObjectDetectionDataset):
    schema = FAIREOObjectDetectionYoloSchema
    name = "DLR-MVDA"
    url = "https://www.dlr.de/en/eoc/about-us/remote-sensing-technology-institute/photogrammetry-and-image-analysis/public-datasets/dlr-mvda"

    labels = ["no-data","Car","Car with a Trailer","Truck","Truck with a Trailer","Van","Van with a Trailer","Long truck","Bus"]

    # the label values of the original data are mapped to consecutive integers
    label_map = {
        '0': 0,
        '10': 1,
        '11': 2,
        '22': 3,
        '23': 4,
        '16': 5,
        '17': 6,
        '20': 7,
        '30': 8
        } 

    def __init__(self, config):
        # now call the constructor to validate the schema and split the data
        super().__init__(config)
        self.data_dir = self.config.data_dir
        self.annotation_dir = self.config.annotation_dir
        self.annotations, self.data = self.load_dataset(
            self.data_dir, self.annotation_dir
        )


    def __getitem__(self, index):
        img_name, x, y, bboxs = self.data[index]
        image = image_loader(os.path.join(self.data_dir, f"{img_name}.JPG"))/255

        # Crop the original image to 256x256 patch
        cropped_image = image[y:y+IMG_H, x:x+IMG_W]

        boxes = []
        labels = []

        for bbox, label_id in bboxs:
            xmin, ymin, xmax, ymax = bbox
            xmin -= x
            ymin -= y
            xmax -= x
            ymax -= y
            xmin, xmax = max(0, int(xmin)), min(int(IMG_W), int(xmax))
            ymin, ymax = max(0, int(ymin)), min(int(IMG_H), int(ymax))

            if xmax > xmin and ymax > ymin:
                boxes.append([xmin, ymin, xmax, ymax])
                labels.append(int(label_id))

        # convert boxes into a torch.Tensor
        boxes = torch.as_tensor(boxes, dtype=torch.float32)

        # suppose all instances are not crowd
        iscrowd = torch.zeros((boxes.shape[0],), dtype=torch.int64)

        labels = torch.as_tensor(labels, dtype=torch.int64)

        target = {"boxes": boxes, "labels": labels, "iscrowd": iscrowd}

        return self.apply_transformations(cropped_image, target)


    def load_dataset(self, data_dir, annotation_dir):
        annotations = []
        data = []

        image_names = list(set(os.path.splitext(f)[0] for f in os.listdir(data_dir) if f.endswith('.JPG')))
        stride = IMG_H

        for img in image_names:
            image = Image.open(os.path.join(self.data_dir, f"{img}.JPG"))
            width, height = image.size

            annot_names = list(os.path.splitext(f)[0] for f in os.listdir(data_dir) if f.endswith('.samp') and f.startswith(img))

            bboxs = []
            for annot in annot_names:
                samp_file_path = os.path.join(data_dir, f'{annot}.samp')
                with open(samp_file_path, 'r') as f:
                    for line in f:
                        if line.startswith('#') or line.startswith('@'):
                            # ignore comment lines in the annotation file
                            continue
                        parts = line.strip().split()
                        label_id_initial, centerx, centery, sizewidth, sizeheight, angle  = parts[1:]
                        # assign new label value with label_map
                        label_id = self.label_map.get(label_id_initial, label_id_initial) 

                        bboxs.append((self.get_bbox(centerx, centery, sizewidth, sizeheight, angle),label_id))
                        annotations.append({"label": int(label_id)})

            # redefine the bbox for 256 x 256 patches
            for y in range(0, height - IMG_H + 1, stride):
                for x in range(0, width - IMG_W + 1, stride):
                    if y + IMG_H <= height and x + IMG_W <= width:
                        newbbox = []
                        for bbox, label_id in bboxs:
                            if self.bbox_overlap_ratio(bbox, x, y):
                                newbbox.append((bbox, label_id))
                        if len(newbbox) > 0:
                            data.append((img, x, y, newbbox))

        return annotations, data
    
    
    def get_bbox(self, centerx, centery, sizew, sizeh, angle):
        """
        Translates the rotated bounding boxes into axis-aligned bounding boxes
        """

        centerx, centery, sizeh, sizew, angle = int(centerx), int(centery), int(sizeh), int(sizew), float(angle)
        # Convert angle from degrees to radians
        angle_rad = math.radians(angle)

        # Define the four corners of the box (relative to center of bbox)
        corners = np.array([
            [-sizew, -sizeh],  # top-left
            [sizew, -sizeh],   # top-right
            [sizew, sizeh],    # bottom-right
            [-sizew, sizeh]    # bottom-left
        ])

        # Rotation matrix
        cos_a, sin_a = math.cos(angle_rad), math.sin(angle_rad)
        rotation_matrix = np.array([
            [cos_a, -sin_a],
            [sin_a, cos_a]
        ])

        # Rotate corners and add center offset
        rotated_corners = np.dot(corners, rotation_matrix.T)
        rotated_corners[:, 0] += centerx
        rotated_corners[:, 1] += centery

        # Get min and max for bbox
        xmin, ymin = np.min(rotated_corners, axis=0)
        xmax, ymax = np.max(rotated_corners, axis=0)

        return [xmin, ymin, xmax, ymax]
    
    
    def bbox_overlap_ratio(self, bbox, crop_x, crop_y):
        """
        Check if more than half of the bounding box is within the cropped area.
        Returns True if more than 50% of bbox area is inside the cropped patch.
        """
        xmin, ymin, xmax, ymax = bbox

        # Calculate intersection area
        intersect_xmin = max(xmin, crop_x)
        intersect_ymin = max(ymin, crop_y)
        intersect_xmax = min(xmax, crop_x + IMG_W)
        intersect_ymax = min(ymax, crop_y + IMG_H)

        # Check if there's any intersection
        if intersect_xmin >= intersect_xmax or intersect_ymin >= intersect_ymax:
            return False

        # Calculate areas
        intersect_area = (intersect_xmax - intersect_xmin) * (intersect_ymax - intersect_ymin)
        bbox_area = (xmax - xmin) * (ymax - ymin)

        # Return True if more than half of bbox is inside crop
        return intersect_area > (bbox_area * 0.5)
    

    def data_distribution_table(self):
        label_to_name = {
            1: "Car",
            2: "Car with a Trailer",
            3: "Truck",
            4: "Truck with a Trailer",
            5: "Van",
            6: "Van with a Trailer",
            7: "Long truck",
            8: "Bus"
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