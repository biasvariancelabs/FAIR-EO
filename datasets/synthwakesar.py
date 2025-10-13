from .multiclass_classification import MultiClassClassificationDataset
from ..utils import image_loader
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import random

'''
SynthWakeSAR is a synthetic ship classification dataset of synthetic aperture radar (SAR) images 
of the sea surface. It includes 10 real ship models for a total of 46080 simulated images containing 
visible ship wakes. The dataset was created using the 'AssenSAR Image Simulator', also available via 
the University of Bristol.
'''

LABELS = ["Cargo I","Cargo II","Tanker I","Tanker II","Passenger Vessel I","Passenger Vessel II",
          "High Speed Craft I","High Speed Craft II","Fishing Vessel I","Fishing Vessel II"]


class SynthWakeSARDataset(MultiClassClassificationDataset):

    url = "https://data.bris.ac.uk/data/dataset/30kvuvmatwzij2mz1573zqumfx"
    labels = LABELS
    name = "SynthWakeSAR dataset"

    def __init__(self, config):
        # now call the constructor to validate the schema and load the data
        super().__init__(config)
        # load the data
        self.data_dir = self.config.data_dir
        self.csv_file = self.config.csv_file
        self.data = self.load_dataset()

    def __getitem__(self, index):
        """
        :param index: Index
        :type index: int
        :return: tuple where target is index of the target class.
        :rtype: tuple (image, target)

        """
        # load image
        img = image_loader(self.data[index][0])
        img = np.asarray(Image.fromarray(img).convert('RGB'))

        if self.transform:
            img = self.transform(img)
        target = self.data[index][1]
        if self.target_transform:
            target = self.target_transform(target)
        return img, target

    def show_image(self, index):
        label = self.labels[self[index][1]]
        fig = plt.figure(figsize=(8, 6))
        plt.title(
            f"Image with index {index} from the dataset {self.get_name()}, with label {label}\n",
            fontsize=14,
        )
        plt.axis("off")
        plt.imshow(self[index][0][:,:,0])
        return fig

    def show_batch(self, size, show_title=True):
        if size % 5:
            raise ValueError("The provided size should be divided by 5!")
        image_indices = random.sample(range(0, len(self.data)), size)
        figure, ax = plt.subplots(
            int(size / 5), 5, figsize=(13.75, 2.8 * int(size / 5))
        )
        if show_title:
            figure.suptitle(
                "Example images with labels from {}".format(self.get_name()),
                fontsize=32,
                y=1.006,
            )
        for axes, image_index in zip(ax.flatten(), image_indices):
            axes.imshow(self[image_index][0][:,:,0])
            axes.set_title(self.labels[self[image_index][1]], fontsize=18, pad=10)
            axes.set_xticks([])
            axes.set_yticks([])
        figure.tight_layout()
        return figure