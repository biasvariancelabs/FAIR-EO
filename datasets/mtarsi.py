from .multiclass_classification import MultiClassClassificationDataset
from ..utils import image_loader

'''
MTARSI has a total of 9385 remote sensing images acquired from Google Earth satellite imagery and manually expanded. 
The dataset contains 20 different types of aircraft covering 36 airports. Each image contains one and only one complete 
aircraft.
'''

LABELS = ["A","B-1","B-2","type-20","type-20(B-29)","B-52","Boeing","C-130","C-135","C-17","C-5","E-3","F","type-18",
          "type-18(F-16)","F-22","KC-10","type-15(A-10)","type-16","type-16(A-26)","type-17","type-17(P-63)","type-19",
          "type-19(T-6)","type-21","type-21(t-43)","type-12","type-12(C-21)","type-13","type-13(U-2)","type-14","type-14(F-22)",
          "type-15","background","background-1","background-2"]


class MTARSIDataset(MultiClassClassificationDataset):

    url = "https://zenodo.org/records/3464319#.Yxi6e-xByL4"
    labels = LABELS
    name = "MTARSI dataset"

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