import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import h5py

from .semantic_segmentation import BaseDataset
from .schemas import FAIREOSchema
from ..utils import image_loader


"""
QuakeSet is a dataset to analyze different attributes of earthquakes. It contains bi-temporal time series of images and ground truth 
annotations for magnitudes, hypocenters, and affected areas. The images are taken from the Sentinel-1 mission using the Interferometric 
Wide swath mode. The International Seismological Centre provides information about earthquakes.
"""

class QuakeSetDataset(BaseDataset):

    url = "https://huggingface.co/datasets/DarthReca/quakeset"
    name = "QuakeSet"
    schema = FAIREOSchema
    labels = ["no change","earthquake change"]
    color_mapping =  [[0,0,0],[255, 255, 255]]


    def __init__(self, config):
        super().__init__(config)
        self.data_dir = self.config.data_dir
        self.mode = self.config.mode
        self.selection = self.config.selection
        self.data = self.load_dataset(self.data_dir)

    def to_db(x):
        return 10 * np.log10(x + 1e-6)

    def __getitem__(self, index):
        # if the patch contains a 'before' image it's label is 0 (no change) and it's image pair is 'before' and 'pre'
        # if the patch doesn't contain a 'before' image it's label is 1 (earthquake change) and it's image pair is 'pre' and 'post'
        with h5py.File(self.data_dir, 'r') as h5_file:
            key = self.data[index][0]
            patch = self.data[index][1]
            label = self.data[index][2]

            if label == 0: # no earthquake
                pre_image = h5_file[key][patch]['before'][:]
                post_image = h5_file[key][patch]['pre'][:]
            elif label == 1: # earthquake
                pre_image = h5_file[key][patch]['pre'][:]
                post_image = h5_file[key][patch]['post'][:]

            pre_vv = pre_image[:,:,0]
            pre_vh = pre_image[:,:,1]
            pre_vv_vh = pre_vv/pre_vh
            pre_image = np.dstack((pre_vv, pre_vh, pre_vv_vh))

            post_vv = post_image[:,:,0]
            post_vh = post_image[:,:,1]
            post_vv_vh = post_vv/post_vh
            post_image = np.dstack((post_vv, post_vh, post_vv_vh))

            pre_image = np.nan_to_num(pre_image, nan=0)
            post_image = np.nan_to_num(post_image, nan=0)
            
            def normalize_per_channel(img):
                out = np.zeros_like(img, dtype=np.float32)
                for c in range(img.shape[2]):
                    cmin, cmax = img[:,:,c].min(), img[:,:,c].max()
                    out[:,:,c] = (img[:,:,c] - cmin) / (cmax - cmin + 1e-8)
                return out

            pre_image = normalize_per_channel(pre_image)
            post_image = normalize_per_channel(post_image)
    
            if self.transform:
                pre_image = self.transform(pre_image)
                post_image = self.transform(post_image)
            if self.target_transform:
                label = self.target_transform(label)
        
        return (pre_image, post_image), label


    def load_dataset(self, data_dir):
        self.data = []
        with h5py.File(data_dir, 'r') as h5_file:
            keys = list(h5_file.keys())
            for i in np.arange(len(keys)):
                key = keys[i]
                patches = list(h5_file[key].keys())[: -2] # exclude x and y arrays
                attributes = dict(h5_file[key].attrs)
                split = attributes['split']
                if split == self.mode:
                    for patch in patches:
                        try:
                            pre_image = h5_file[key][patch]['before']
                            label = 0
                            self.data.append((key, patch, label))
                        except:
                            label = 1
                            self.data.append((key, patch, label))
                    
        return self.data

    def __len__(self):
        return len(self.data)
    
    def get_labels(self):
        return self.labels
    
    def show_image(self, index):
        (pre_image, post_image), label = self[index]

        label_name = self.labels[label]

        fig = plt.figure(figsize=(15, 6))
        fig.suptitle(
            f"Images with index {index} from the dataset {self.get_name()}, with label {label_name}",
            fontsize=14,
        )

        plt.subplot(1, 2, 1)
        plt.imshow(pre_image)
        plt.title("Pre-event")
        plt.axis("off")

        plt.subplot(1, 2, 2)
        plt.imshow(post_image)
        plt.title("Post-event")
        plt.axis("off")

        return fig

    def data_distribution_table(self):
        df = pd.DataFrame(self.data, columns=['Key', 'Patch', 'Label'])
        
        label_count = df.groupby("Label").count().reset_index()
        label_count = label_count.drop(columns=["Patch"])
        label_count.columns = ["Label", "Count"]
        label_count["Label"] = ['No change','Earthquake change']
        return label_count

    def data_distribution_barchart(self):
        label_count = self.data_distribution_table()
        for i in np.arange(len(label_count)):
            if label_count["Count"][i]<10:
                label_count = label_count.drop([i])

        fig, ax = plt.subplots(figsize=(12, 10))
        sns.barplot(y="Label", x="Count", data=label_count, ax=ax)
        ax.set_title(
            "Labels distribution for {}".format(self.get_name()), pad=20, fontsize=18
        )
        return fig