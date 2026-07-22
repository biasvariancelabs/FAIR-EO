from aitlas.datasets.multiclass_classification import MultiClassClassificationDataset

'''
The dataset was created to identify if a free car space is available. The dataset contains 
raspberry pi images of a car park space used to identify and send a message to the user if a 
free space is available.
'''

LABELS = ["Free","Full"]


class FindACarParkDataset(MultiClassClassificationDataset):

    url = "https://www.kaggle.com/datasets/daggysheep/find-a-car-park"
    labels = LABELS
    name = "Find A Car Park dataset"

    def __init__(self, config):
        # now call the constructor to validate the schema and load the data
        super().__init__(config)