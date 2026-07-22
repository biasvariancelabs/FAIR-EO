from aitlas.datasets.multiclass_classification import MultiClassClassificationDataset
from aitlas.utils import image_loader
import csv

'''
RSI-CB stands for Large Scale Remote Sensing Image Classification Benchmark via Crowdsource Data. RSI-CB128 
containss more than 36000 images of size 128 x 128 pixel, and 45 categories.
'''

LABELS = ["airport_runway","artificial_grassland","avenue","bare_land","bridge","city_avenue","city_building",
          "city_green_tree","city_road", "coastline","container","crossroads","dam","desert","dry_farm","forest",
          "fork_road","grave","green_farmland","highway", "hirst","lakeshore","mangrove","marina","mountain",
          "mountain_road","natural_grassland","overpass","parkinglot","pipeline","rail","residents","river",
          "river_protection_forest","sandbeach","sapling","sea","shrubwood","snow_mountain","sparse_forest",
          "storage_room","stream","tower","town","turning_circle"]

class RSICB128Dataset(MultiClassClassificationDataset):

    url = "https://www.kaggle.com/datasets/noamaanabdulazeem/myfile"
    labels = LABELS
    name = "RSI-CB128 dataset"

    def __init__(self, config):
        # now call the constructor to validate the schema and load the data
        super().__init__(config)