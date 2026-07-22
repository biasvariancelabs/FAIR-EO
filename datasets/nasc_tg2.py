from aitlas.datasets.multiclass_classification import MultiClassClassificationDataset
from PIL import Image
import pandas as pd
import numpy as np
import io

'''
NaSC-TG2 stands for Natural Scene Classification With Tiangong-2 Remotely Sensed Imagery. 
It is a novel benchmark dataset for remote sensing natural scene classification from Tiangong-2 
remotely sensed imagery. It contains 20000 images, which are divided into ten scene classes, 
with 2000 images for each. 
'''

LABELS = ["beach", "circular_farmland", "cloud", "desert", "forest", 
        "mountain", "rectangular_farmland", "residential", "river", "snowberg"]

class NaSCTG2Dataset(MultiClassClassificationDataset):
    url = "https://huggingface.co/datasets/jonathan-roberts1/NaSC-TG2/tree/main"
    labels = LABELS
    name = "NaSC-TG2 dataset"

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
        # get raw bytes and label index tuple
        img_bytes, target = self.data[index]
        # load image from bytes directly
        img = np.asarray(Image.open(io.BytesIO(img_bytes)))
        # apply transformations
        if self.transform:
            img = self.transform(img)
        if self.target_transform:
            target = self.target_transform(target)
        
        return img, target    

    def load_dataset(self):
        # Read parquet into DataFrame
        df = pd.read_parquet(self.data_dir)

        if self.csv_file: # For splits, csv file contains indexes
            idx_df = pd.read_csv(self.csv_file, header=None, names=['index', 'label'])
            # Filter parquet df to only rows with those indexes
            df = df.iloc[idx_df['index'].values]

        data = []
        for _, row in df.iterrows():
            # row['image'] contains a dict with 'bytes'
            img_bytes = row['image']['bytes']  # raw image bytes
            label = row['label']
            data.append((img_bytes, label))
        
        if not self.labels:
            raise ValueError("You need to provide the list of labels for the dataset")
        
        return data
    
    def show_samples(self):
        df = pd.read_parquet(self.data_dir)
        df = df.rename(columns={"image": "Image", "label": "Label"})
        df["Label"] = df["Label"].apply(lambda x: self.labels[x])
        return df.head(20)


    def data_distribution_table(self):
        df = pd.read_parquet(self.data_dir)

        # Count labels
        label_counts = df['label'].value_counts().reset_index()
        label_counts.columns = ['LabelIndex', 'Count']
        label_counts['Label'] = label_counts['LabelIndex'].apply(lambda x: self.labels[x])

        return label_counts[['Label', 'Count']]