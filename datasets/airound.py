from aitlas.datasets.multiclass_classification import MultiClassClassificationDataset
from aitlas.utils import image_loader

'''
AiRound is composed of 11753 images distributed among 11 classes. Each sample is composed by a triplet, 
that contains images in 3 distinct points of view: a ground perspective image; a high resolution RGB 
aerial image; and a multi-spectral image taken from the Sentinel-2 satellite.
'''

LABELS = ["airport","bridge","church","forest","lake","park","river","skyscraper","stadium","statue","tower"]


class AiRoundDataset(MultiClassClassificationDataset):

    url = "https://www.dropbox.com/scl/fi/qgjxoene22cor54fzr20v/Gabriel-airoundv2.zip?rlkey=3n3knpid5r48jilf1bag43m6h&e=1&dl=0"
    labels = LABELS
    name = "AiRound dataset"

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
        img = img[:,:,0:3]
        # apply transformations
        if self.transform:
            img = self.transform(img)
        target = self.data[index][1]
        if self.target_transform:
            target = self.target_transform(target)
        return img, target