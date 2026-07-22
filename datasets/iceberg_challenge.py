import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import fnmatch

from aitlas.datasets.multiclass_classification import MultiClassClassificationDataset
from .schemas import FAIREOSchema

"""
The Statoil/C-CORE Iceberg Classifier Challenge dataset contains Sentinel-1
SAR images for distinguishing between ships and icebergs in Arctic waters.
Each sample includes dual-polarization (HH and HV) radar data represented
as 75x75 pixel patches, with labels indicating whether the object is an iceberg
or a ship.
"""


class IcebergClassificationChallengeDataset(MultiClassClassificationDataset):
    url = ["https://www.kaggle.com/competitions/statoil-iceberg-classifier-challenge/overview"]
    name = "Iceberg Classifier Dataset"
    schemas = FAIREOSchema
    labels = ["ship", "iceberg"]

    def __init__(self, config):
        # now call the constructor to validate the schema and split the data
        super().__init__(config)
        self.data_dir = self.config.data_dir

    def __getitem__(self, index):
        if fnmatch.fnmatch(self.data_dir, '*train*'):
            target = int(self._json["is_iceberg"][index])
        else:
            target = None

        band_1 = np.array(self._json["band_1"][index])
        band_2 = np.array(self._json["band_2"][index])

        # from 1d vector to 2d image
        band_1_2d = band_1.reshape(75, 75) 
        band_2_2d = band_2.reshape(75, 75)

        band_1_norm = (band_1_2d - band_1_2d.min()) / (
            band_1_2d.max() - band_1_2d.min()
        )
        band_2_norm = (band_2_2d - band_2_2d.min()) / (
            band_2_2d.max() - band_2_2d.min()
        )

        # create RGB image: R=Band1, G=Band2, B=Band1/Band2
        rgb_image = np.zeros((75, 75, 3))
        rgb_image[:, :, 0] = band_1_norm  
        rgb_image[:, :, 1] = band_2_norm  
        rgb_image[:, :, 2] = band_1_norm / (band_2_norm + 1e-6)  # prevent division by zero

        if self.transform:
            rgb_image = self.transform(rgb_image)
        if self.target_transform:
            target = self.target_transform(target)

        return (rgb_image, target)

    def load_dataset(self):
        img_ids = []
        df = pd.read_json(self.data_dir)
        self._json = df
        if fnmatch.fnmatch(self.data_dir, '*train*'):
            self.annot = [{"Label": int(i)} for i in list(df["is_iceberg"].values)]
        img_ids = [i for i in range(1, len(self._json))]

        return img_ids

    def data_distribution_table(self):
        df = pd.DataFrame(self.annot)
        rename_map = {
            0: "ship",
            1: "iceberg"
        }

        df["Label"] = df["Label"].replace(rename_map)
        label_count = df.groupby("Label").size().reset_index(name="Count")
        return label_count

    def data_distribution_barchart(self):
        label_count = self.data_distribution_table()
        fig, ax = plt.subplots(figsize=(12, 10))
        sns.barplot(x="Label", y="Count", data=label_count, ax=ax)
        ax.set_title(
            "Labels distribution for {}".format(self.get_name()), pad=20, fontsize=18
        )
        return fig