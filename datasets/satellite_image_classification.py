from .multiclass_classification import MultiClassClassificationDataset
from ..utils import image_loader

'''
The dataset is created for remote sensing image interpretation. It consists of 5631 images in four classes.
'''

LABELS = ["cloudy","desert","green_area","water"]


class SatelliteImageClassificationDataset(MultiClassClassificationDataset):

    url = "https://www.kaggle.com/datasets/mahmoudreda55/satellite-image-classification"
    labels = LABELS
    name = "Satellite Image Classification dataset"

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
        img = image_loader(self.data[index][0])[:,:,0:3]
        # apply transformations
        if self.transform:
            img = self.transform(img)
        target = self.data[index][1]
        if self.target_transform:
            target = self.target_transform(target)
        return img, target