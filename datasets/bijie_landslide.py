from .multiclass_classification import MultiClassClassificationDataset

'''
The Bijie landslide dataset is a remote sensing landslide dataset for developing automatic landslide 
detection methods. The dataset consists of satellite optical images, shapefiles of landslides' boundaries 
and digital elevation models. All the images in this dataset, i.e. 770 landslide images and 2003 non-landslide 
images were cropped from the TripleSat satellite images captured from May to August 2018.
'''

LABELS = ["non-landslide","landslide"]


class BijieLandslideDataset(MultiClassClassificationDataset):

    url = "http://gpcv.whu.edu.cn/data/Bijie_pages.html"
    labels = LABELS
    name = "Bijie Landslide dataset"

    def __init__(self, config):
        # now call the constructor to validate the schema and load the data
        super().__init__(config)