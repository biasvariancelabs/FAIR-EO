import numpy as np
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import csv
import json

from .semantic_segmentation import SemanticSegmentationDataset
from ..utils import image_loader
from .schemas import FAIREOSchema

"""
VALID stands for Virtual Aerial Image Dataset. It consits of 6690 high-resolution images, all annotated with 
panoptic segmentation on 30 categories, object detection with the oriented bounding box, and binocular depth maps, 
collected in 6 different virtual scenes and 5 various ambient conditions (sunny, dusk, night, snow and fog). This 
dataloader considers the semantic segmentation task.
"""


class VALIDSemanticSegmentationDataset(SemanticSegmentationDataset):
    url = "https://drive.google.com/drive/folders/1nKQqR92NqNEWj-KXvxAKekFdQikrotLj"

    labels = ["background","tree","plant","road","pavement","land","water","pool","ice","stone","pier rubble","bridge","sign",
              "small vehicle","large vehicle","building","animal","person","chair","fence","garbage bin","low obstacle",
              "telegraph pole","traffic light","bus stop","lamp","high obstacle","tunnel","ship","plane","harbor"]
    color_mapping = [[55, 181, 57],[89, 121, 72],[190, 225, 64],[206, 190, 59],[11, 236, 9],[153, 108, 6],[135, 169, 180],[115, 176, 195],
                     [112, 105, 191],[0, 53, 65],[211, 80, 208],[35, 196, 244],[81, 13, 36],[196, 30, 8],[205, 120, 161],[220, 163, 49],
                     [146, 52, 70],[242, 107, 146],[121, 67, 28],[68, 218, 116],[29, 26, 199], [54, 72, 205],[226, 149, 143],[151, 126, 171],
                     [103, 252, 157],[102, 16, 239],[189, 135, 188],[161, 171, 27],[124, 21, 123],[19, 132, 69],[86, 254, 214]] 
                     
    name = "VALID Semantic Segmentation Dataset"
    schema = FAIREOSchema

    def __init__(self, config):
        # now call the constructor to validate the schema and split the data
        super().__init__(config)
        self.data_dir = self.config.data_dir
        self.selection = self.config.selection
        self.csv_file = self.config.csv_file

    def __getitem__(self, index):
        img_name = self.images[index]
        image = image_loader(os.path.join(self.data_dir,img_name))[:,:,:3] # rgb image
        
        # the semantic labels are in the 'semantic' folder but the image name and corresponding semantic file name are in the 'label' json files
        for ann in os.listdir(os.path.join(self.data_dir, 'label')):
            annotations_file_path = os.path.join(self.data_dir, 'label', ann)
            annotations = json.load(open(annotations_file_path, "r"))
            name = annotations["file_name"]
            if img_name == name:
                segmentation = annotations["segmentation"]["semantic_filename"]

        mask_rgb = image_loader(os.path.join(self.data_dir, segmentation))
        
        # Flatten color mapping for vectorized matching
        color_array = np.array(self.color_mapping, dtype=np.uint8)
        mask_rgb_flat = mask_rgb.reshape(-1, 3)
        # Compute matches
        mask_indices = np.zeros(mask_rgb_flat.shape[0], dtype=np.uint8)
        for idx, color in enumerate(color_array):
            matches = np.all(mask_rgb_flat == color, axis=1)
            mask_indices[matches] = idx
        
        single_band_mask = mask_indices.reshape(mask_rgb.shape[:2])
        masks = [(single_band_mask == v) for v in range(len(self.labels))]
        mask = np.stack(masks, axis=-1).astype("float32")

        return self.apply_transformations(image, mask)

    def load_dataset(self, data_dir, csv_file):
        if not self.labels:
            raise ValueError("You need to provide the list of labels for the dataset")
        
        images_list = []
        split_list = []
        img_names = []
        
        for i in os.listdir(data_dir+'images/'):
            for j in os.listdir(os.path.join(data_dir,'images',i)):
                for k in os.listdir(os.path.join(data_dir,'images',i,j)):
                    img_names.append(k)
                    images_list.append(os.path.join(i,j,k))
            
        if csv_file: # train/val or test split
            with open(csv_file, mode ='r')as file:
                csvFile = csv.reader(file)
                for lines in csvFile:
                    split_list.append(lines[1]+'/'+lines[0])
            indices =[]
            for n,m in enumerate(images_list):
                if m not in split_list:
                    indices.append(n)
            for i in sorted(indices, reverse=True):
                del images_list[i]

        self.images = [os.path.join("images", image_id) for image_id in images_list]

    def data_distribution_table(self):
        label_dist = {key: 0 for key in self.labels}
        for image, mask in self.dataloader():
            for index, label in enumerate(self.labels):
                label_dist[self.labels[index]] += mask[:, :, :, index].sum()
        label_count = pd.DataFrame.from_dict(label_dist, orient='index')
        label_count.columns = ["Number of pixels"]
        label_count = label_count.astype(float)
        return label_count
    
    def data_distribution_barchart(self, show_title=True):
        label_count = self.data_distribution_table()
        fig, ax = plt.subplots(figsize=(12, 12))
        sns.barplot(data=label_count, x=label_count.index, y='Number of pixels', ax=ax)
        fig.autofmt_xdate()
        if show_title:
            ax.set_title(
                "Labels distribution for {}".format(self.get_name()), pad=20, fontsize=18
            )
        return fig