import os
import random
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import pandas as pd
import seaborn as sns
import torch

from aitlas.datasets.multiclass_classification import MultiClassClassificationDataset
from .schemas import FAIREOSchema

'''
S2UV stands for Satellite & Street-view images for Urban Village classification. The dataset 
contains 856 and 1714 image samples corresponding to the two categories of urban villages and 
non-urban villages, respectively, which are sampled in Beijing, Tianjin and Shijiazhuang.
'''

class S2UVDataset(MultiClassClassificationDataset):
    url = "https://www.scidb.cn/en/detail?dataSetId=14e27d8c51ec40079b84591e9bb24df6#p2"
    labels = ["Non-Urban Village", "Urban Village"]
    name = "S2UV"
    schema = FAIREOSchema

    def __init__(self, config):
        # now call the constructor to validate the schema and load the data
        super().__init__(config)
        # load the data
        self.data_dir = self.config.data_dir
        self.selection = self.config.selection

    def __getitem__(self, index):

        sample = self.image[index]
        target = self.label[index]

        if self.selection == "all": # selects the satellite image and the four street views

            sat = sample[0]
            sat = Image.fromarray(sat)
            sv0 = sample[1]
            sv0 = Image.fromarray(sv0)
            sv90 = sample[2]
            sv90 = Image.fromarray(sv90)
            sv180 = sample[3]
            sv180 = Image.fromarray(sv180)
            sv270 = sample[4]
            sv270 = Image.fromarray(sv270)

            return (sat, sv0, sv90, sv180, sv270), target

        else: # selects one specific image
            img = torch.tensor(sample[int(self.selection)], dtype=torch.float32).permute(2,0,1)
            img = self.transform(img)
            target = torch.tensor(target, dtype=torch.long)

            return img, target

    def load_dataset(self):

        label_dir = os.path.join(self.data_dir, f"{self.data_dir.split('/')[-1]}_Y.npy")
        self.label = np.load(label_dir)
        image_dir = os.path.join(self.data_dir, f"{self.data_dir.split('/')[-1]}_X.npy")
        self.image = np.load(image_dir)


    def __len__(self):
        return len(self.image)

    def show_image(self, index):
        if self.selection == "all":
            (sat, sv0, sv90, sv180, sv270), target = self.__getitem__(index)
            label = self.labels[self[index][1]]
            fig, axs = plt.subplots(1, 5, figsize=(20, 4))

            images = [sat, sv0, sv90, sv180, sv270]
            titles = ['sat', 'sv0', 'sv90', 'sv180', 'sv270']

            fig.suptitle(
                f"Images with index {index} from dataset {self.get_name()}, label: {label}",
                fontsize=16
            )

            for ax, img, title in zip(axs, images, titles):
                ax.imshow(img)
                ax.set_title(title)
                ax.axis('off')

            plt.tight_layout(rect=[0, 0, 1, 0.95])  # Leave room for suptitle

            return fig
        else:
            print("Change config.selection to 'all' for show_image() and show_batch() functions.")

    def show_batch(self, size, show_title=True):
        if self.selection == "all":
            image_indices = random.sample(range(len(self.data)), size)
            figure, axs = plt.subplots(size, 5, figsize=(20, 4 * size))

            if show_title:
                figure.suptitle(
                    "Example images with labels from {}".format(self.get_name()),
                    fontsize=32,
                    y=1.006,
                )
            titles = ['sat', 'sv0', 'sv90', 'sv180', 'sv270']

            for row_idx, idx in enumerate(image_indices):
                (sat, sv0, sv90, sv180, sv270), target = self.__getitem__(idx)
                label = self.labels[self[idx][1]]
                images = [sat, sv0, sv90, sv180, sv270]

                for col_idx, (ax, img, t) in enumerate(zip(axs[row_idx], images, titles)):
                    ax.imshow(img)
                    if col_idx == 0:
                        ax.set_title(f"{label}", fontsize=16, pad=10)
                    else:
                        ax.set_title(t, fontsize=12)
                    ax.axis('off')

            plt.tight_layout(rect=[0, 0, 1, 0.95])
            return figure
        else:
            print("Change config.selection to 'all' for show_image() and show_batch() functions.")


    def show_samples(self):
        df = pd.DataFrame({
            'Index': list(range(len(self.label))),
            'Label': [self.labels[i] for i in self.label]
        })
        return df.head(20)

    def data_distribution_table(self):
        count=0
        for label in self.label:
            if label:
                count+=1

        df = pd.DataFrame({
            'Label': self.labels,
            'Count':  [count, len(self.label)-count]
        })

        return df

    def data_distribution_barchart(self):
        label_count = self.data_distribution_table()
        fig, ax = plt.subplots(figsize=(12, 10))
        sns.barplot(y="Label", x="Count", data=label_count, ax=ax)
        ax.set_title(
            "Labels distribution for {}".format(self.get_name()), pad=20, fontsize=18
        )
        return fig