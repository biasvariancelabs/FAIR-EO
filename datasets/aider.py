from aitlas.datasets.multiclass_classification import MultiClassClassificationDataset

'''
AIDER stands for Aerial Image Database for Emergency Response applications. The dataset construction 
involved manually collecting all images for four disaster events, namely Fire/Smoke, Flood, Collapsed 
Building/Rubble, and Traffic Accidents, as well as one class for the Normal case.
'''

LABELS = ["fire", "flooded_areas", "collapsed_building", "traffic_incident","normal"]


class AIDERDataset(MultiClassClassificationDataset):

    url = "https://zenodo.org/records/3888300"
    labels = LABELS
    name = "AIDER dataset"

    def __init__(self, config):
        # now call the constructor to validate the schema and load the data
        super().__init__(config)