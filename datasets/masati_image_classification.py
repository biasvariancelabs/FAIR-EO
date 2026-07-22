from aitlas.datasets.multiclass_classification import MultiClassClassificationDataset
from aitlas.utils import image_loader
import numpy as np

'''
The MASATI (MAritime SATellite Imagery) dataset provides maritime scenes of optical aerial images. 
It can be used to evaluate ship detection methods. It contains 7389 satellite images and can be used 
for image classification, or object detection for the 'coast_ship', 'ship' and 'multi' classes. This 
description considers the image classification task.
'''

LABELS = ["land","coast","ship","multi","coast_ship","detail","water"]


class MASATIImageClassificationDataset(MultiClassClassificationDataset):

    url = "https://docs.google.com/forms/d/e/1FAIpQLSe1qeo_XH2v_obix_huLVVqWpGL_1DAZlMTVlEeMuy-EHdH9w/viewform?c=0&w=1"
    labels = LABELS
    name = "MASATI dataset"

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
        if img.ndim >2:
            img = img[:,:,0:3]
        # apply transformations
        if self.transform:
            img = self.transform(img)
        target = self.data[index][1]
        if self.target_transform:
            target = self.target_transform(target)
        return img, target