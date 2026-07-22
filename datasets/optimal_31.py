from aitlas.datasets.multiclass_classification import MultiClassClassificationDataset

'''
The dataset was created for scene classification with images collected from Google Earth. 
It contains 31 classes and each class is formed by 60 images.
'''

LABELS = ["airplane","airport","baseball_diamond","basketball_court","beach","bridge","chaparral",
          "church","circular_farmland","commercial_area","dense_residential","desert","forest","freeway",
          "golf_course","ground_track_field","harbor","industrial_area","intersection","island","lake",
          "meadow","medium_residential","mobile_home_park","mountain","overpass","parking_lot","railway",
          "rectangular_farmland","roundabout","runway"]


class OPTIMAL31Dataset(MultiClassClassificationDataset):

    url = "https://drive.google.com/file/d/1Fk9a0DW8UyyQsR8dP2Qdakmr69NVBhq9/view"
    labels = LABELS
    name = "OPTIMAL-31 dataset"

    def __init__(self, config):
        # now call the constructor to validate the schema and load the data
        super().__init__(config)