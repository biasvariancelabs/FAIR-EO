import numpy as np
import os
import pandas as pd
import csv

from .semantic_segmentation import SemanticSegmentationDataset
from ..utils import image_loader
from .schemas import FAIREOSchema

"""
The dataset was published by Humans in the Loop for a joint project with the Mohammed Bin Rashid 
Space Center in Dubai, the UAE. It consists of aerial imagery of Dubai obtained by MBRSC satellites 
and annotated with pixel-wise semantic segmentation in 6 classes. The total volume of the dataset 
is 72 images grouped into 8 larger tiles.
"""


class KaggleSemanticSegmentationDataset(SemanticSegmentationDataset):
    url = "https://www.kaggle.com/datasets/humansintheloop/semantic-segmentation-of-aerial-imagery"

    labels = ["building","land","road","vegetation","water","unlabeled"]
    color_mapping = [[60,16,152],[132,41,246],[110,193,228],[254,221,58],[226,169,41],[155,155,155]] 
    name = "Kaggle Semantic Segmentation"
    schema = FAIREOSchema

    def __init__(self, config):
        # now call the constructor to validate the schema and split the data
        super().__init__(config)
        self.data_dir = self.config.data_dir
        self.selection = self.config.selection
        self.csv_file = self.config.csv_file


    def __getitem__(self, index):
        image = image_loader(self.images[index])
        mask = image_loader(self.masks[index])
        if mask.ndim>2:
            single_band_mask = np.zeros([mask.shape[0], mask.shape[1]], np.uint8)
            for i in np.arange(mask.shape[0]):
                for j in  np.arange(mask.shape[1]):
                    if mask[i,j,0] == 60 and mask[i,j,1] == 16 and mask[i,j,2] == 152:
                        single_band_mask[i,j] = 0
                    if mask[i,j,0] == 132 and mask[i,j,1] == 41 and mask[i,j,2] == 246:
                        single_band_mask[i,j] = 1
                    if mask[i,j,0] == 110 and mask[i,j,1] == 193 and mask[i,j,2] == 228:
                        single_band_mask[i,j] = 2
                    if mask[i,j,0] == 254 and mask[i,j,1] == 221 and mask[i,j,2] == 58:
                        single_band_mask[i,j] = 3
                    if mask[i,j,0] == 226 and mask[i,j,1] == 169 and mask[i,j,2] == 41:
                        single_band_mask[i,j] = 4
                    if mask[i,j,0] == 155 and mask[i,j,1] == 155 and mask[i,j,2] == 155:
                        single_band_mask[i,j] = 5
        else:
            single_band_mask = mask 
        
        masks = [(single_band_mask == v) for v, label in enumerate(self.labels)]
        mask = np.stack(masks, axis=-1).astype("float32")

        return self.apply_transformations(image, mask)

    def load_dataset(self, data_dir, csv_file):
        if not self.labels:
            raise ValueError("You need to provide the list of labels for the dataset")

        ids = os.listdir(os.path.join(data_dir, "images"))
        
        split_list = []
        ids_final = []
        if csv_file: 
            with open(csv_file, mode ='r')as file:
                csvFile = csv.reader(file)
                for lines in csvFile:
                    split_list.append(lines[0])
            for i in ids:
                i_csv = 'images/'+ i
                if i_csv in split_list:
                    ids_final.append(i)
            ids = ids_final

        self.images = [os.path.join(data_dir, "images", image_id) for image_id in ids]
        self.masks = [os.path.join(data_dir, "labels", image_id[: image_id.rfind('.jpg')]+'.png') for image_id in ids]
        
    def data_distribution_table(self):
        label_dist = {key: 0 for key in self.labels}
        for i in np.arange(len(self.images)):  #because images have different sizes
            mask = self[i][1]
            for index, label in enumerate(self.labels):
                label_dist[self.labels[index]] += mask[:, :, index].sum()
        label_count = pd.DataFrame.from_dict(label_dist, orient='index')
        label_count.columns = ["Number of pixels"]
        label_count = label_count.astype(float)
        return label_count