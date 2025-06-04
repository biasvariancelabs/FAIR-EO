from .multiclass_classification import MultiClassClassificationDataset
from ..utils import image_loader

'''
CV-BrCT stands for Cross-View Brazilian Construction Type. The dataset is composed of 24000 pairs of images, 
each one containing a street-level scene and a high-resolution aerial image. Those samples are labeled in 8 
different classes.
'''

LABELS = ["apartment","hospital","house","industrial","parking_lot","religious","school","store","vacant_lot"]


class CVBrCTDataset(MultiClassClassificationDataset):

    url = "https://drive.google.com/file/d/1GK5haKw7-Cr479NW2JPosMOkl14W4cxY/view"
    labels = LABELS
    name = "CV-BrCT dataset"

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