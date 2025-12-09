import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
import torch
from matplotlib.patches import Patch
from scipy.ndimage import zoom
import seaborn as sns
from netCDF4 import Dataset

from ..base import BaseDataset
from .schemas import Sen4AgriNetSchema


"""
Sen4AgriNet is a Sentinel-2 based time series multi-country benchmark dataset, tailored for agricultural 
monitoring applications with Machine and Deep Learning.
"""

CROP_ENCODING = {
    'Cereals': 100,
    'Wheat': 110,
    'Maize': 120,
    'Rice': 130,
    'Sorghum': 140,
    'Barley': 150,
    'Rye': 160,
    'Oats': 170,
    'Millets': 180,
    'Other cereals n.e.c.': 190,
    'Mixed cereals': 191,
    'Other': 192,
    'Vegetables and melons': 200,
    'Leafy or stem vegetables': 210,
    'Artichokes': 211,
    'Asparagus': 212,
    'Cabbages': 213,
    'Cauliflowers and broccoli': 214,
    'Lettuce': 215,
    'Spinach': 216,
    'Chicory': 217,
    'Other leafy or stem vegetables n.e.c.': 219,
    'Fruit bearing vegetables': 220,
    'Cucumbers': 221,
    'Eggplants (aubergines)': 222,
    'Tomatoes': 223,
    'Watermelons': 224,
    'Cantaloupes and other melons': 225,
    'Pumpkin squash and gourds': 226,
    'Other fruit bearing vegetables n.e.c.': 227,
    'Root bulb or tuberous vegetables': 230,
    'Carrots': 231,
    'Turnips': 232,
    'Garlic': 233,
    'Onions (incl. shallots)': 234,
    'Leeks and other alliaceous vegetables': 235,
    'Other root bulb or tuberous vegetables n.e.c.': 236,
    'Mushrooms and truffles': 240,
    'Vegetables n.e.c.': 250,
    'Fruit and nuts': 300,
    'Tropical and subtropical fruits': 310,
    'Avocados': 311,
    'Bananas and plantains': 312,
    'Dates': 313,
    'Figs': 314,
    'Mangoes': 315,
    'Papayas': 316,
    'Pineapples': 317,
    'Other tropical and subtropical fruits n.e.c.': 318,
    'Citrus fruits': 320,
    'Grapefruit and pomelo': 321,
    'Lemons and Limes': 322,
    'Oranges': 323,
    'Tangerines mandarins clementines': 324,
    'Other citrus fruit n.e.c.': 325,
    'Grapes': 330,
    'Berries': 340,
    'Currants': 341,
    'Gooseberries': 342,
    'Kiwi fruit': 343,
    'Raspberries': 344,
    'Strawberries': 345,
    'Blueberries': 346,
    'Other berries': 347,
    'Pome fruits and stone fruits': 350,
    'Apples': 351,
    'Apricots': 352,
    'Cherries and sour cherries': 353,
    'Peaches and nectarines': 354,
    'Pears and quinces': 355,
    'Plums and sloes': 356,
    'Other pome fruits and stone fruits n.e.c.': 357,
    'Nuts': 360,
    'Almonds': 361,
    'Cashew nuts': 362,
    'Chestnuts': 363,
    'Hazelnuts': 364,
    'Pistachios': 365,
    'Walnuts': 366,
    'Other nuts n.e.c.': 367,
    'Other fruits': 380,
    'Oilseed crops': 400,
    'Soya beans': 410,
    'Groundnuts': 420,
    'Other temporary oilseed crops': 430,
    'Castor bean': 431,
    'Linseed': 432,
    'Mustard': 433,
    'Niger seed': 434,
    'Rapeseed': 435,
    'Safflower': 436,
    'Sesame': 437,
    'Sunflower': 438,
    'Other temporary oilseed crops n.e.c.': 439,
    'Permanent oilseed crops': 440,
    'Coconuts': 441,
    'Olives': 442,
    'Oil palms': 443,
    'Other oleaginous fruits n.e.c.': 444,
    'Root tuber crops with high starch or inulin content': 500,
    'Potatoes': 510,
    'Sweet potatoes': 520,
    'Cassava': 530,
    'Yams': 540,
    'Other roots and tubers n.e.c.': 550,
    'Beverage and spice crops': 600,
    'Beverage crops': 610,
    'Coffee': 611,
    'Tea': 612,
    'Mate': 613,
    'Cocoa': 614,
    'Other beverage crops n.e.c.': 615,
    'Spice crops': 620,
    'Chilies and peppers (capsicum spp.)': 621,
    'Anise badian and fennel': 622,
    'Other temporary spice crops n.e.c.': 623,
    'Pepper (piper spp.)': 624,
    'Nutmeg mace cardamoms': 625,
    'Cinnamon (canella)': 626,
    'Cloves': 627,
    'Ginger': 628,
    'Vanilla': 629,
    'Other permanent spice crops n.e.c.': 630,
    'Leguminous crops': 700,
    'Beans': 710,
    'Broad beans': 720,
    'Chick peas': 730,
    'Cow peas': 740,
    'Lentils': 750,
    'Lupins': 760,
    'Peas': 770,
    'Pigeon peas': 780,
    'Leguminous crops n.e.c.': 790,
    'Sugar crops': 800,
    'Sugar beet': 810,
    'Sugar cane': 820,
    'Sweet sorghum': 830,
    'Other sugar crops n.e.c.': 840,
    'Other crops and Classes': 900,
    'Grasses and other fodder crops': 910,
    'Temporary grass crops': 911,
    'Permanent grass crops': 912,
    'Fiber crops': 920,
    'Cotton': 921,
    'Jute kenaf other similar crops': 922,
    'Flax hemp and other similar products': 923,
    'Other temporary fibre crops': 924,
    'Permanent fibre crops': 925,
    'Medicinal aromatic pesticidal or similar crops': 930,
    'Temporary medicinal etc. crops': 931,
    'Permanent medicinal etc. crops': 932,
    'Rubber': 940,
    'Flower crops': 950,
    'Temporary flower crops': 951,
    'Permanent flower crops': 952,
    'Tobacco': 960,
    'Other Classes': 970,
    'Artificial Surfaces': 971,
    'Forest': 972,
    'Wetlands': 973,
    'Water bodies': 974,
    'Fallow land': 975,
    'Baren land': 976,
    'No Data Available': 977,
    'Other crops': 980,
    'Other crops temporary': 981,
    'Other crops permanent': 982,
    'Unknown crops': 998
}

CORE_CLASSES = {
    "background/other": 0,
    "wheat": 110,
    "maize": 120,
    "sorghum": 140,
    "barley": 150,
    "rye": 160,
    "oats": 170,
    "grapes": 330,
    "rapeseed": 435,
    "sunflower": 438,
    "potatoes": 510,
    "peas": 770
}

COLOR_MAPPING = [
    [31, 119, 180],
    [174, 199, 232],
    [255, 187, 120],
    [152, 223, 138],
    [255, 152, 150],
    [197, 176, 213],
    [140, 86, 75],
    [227, 119, 194],
    [127, 127, 127],
    [188, 189, 34],
    [23, 190, 207],
    [158, 218, 229]
    ]


class Sen4AgriNetDataset(BaseDataset):
    url = "https://github.com/Orion-AI-Lab/S4A"
    name = "Sen4AgriNet"
    schema = Sen4AgriNetSchema

    def __init__(self, config):
        # now call the constructor to validate the schema and split the data
        super().__init__(config)
        self.data_dir = self.config.data_dir
        self.selection = self.config.selection
        self.csv_file = self.config.csv_file
        self.date = int(self.config.date) if self.config.date is not None else None
        self.class_selection = self.config.class_selection

        if self.class_selection == "all":
            self.labels = list(CROP_ENCODING.keys())
            self.label_ids = list(CROP_ENCODING.values())
        elif self.class_selection == "12":
            self.labels = list(CORE_CLASSES.keys())
            self.label_ids = list(CORE_CLASSES.values())
        else:
            raise ValueError(f"Invalid class_selection '{self.class_selection}'. Choose 'all' or 'core'.")

        self.num_classes = len(self.labels)
        colors = plt.get_cmap("hsv")(np.linspace(0, 1, self.num_classes))[:, :3] * 255
        if self.class_selection == "all":
            self.color_mapping = colors.astype(int).tolist()
        elif self.class_selection == "12":
            self.color_mapping = COLOR_MAPPING

        self.patch_files = self.load_dataset(self.data_dir,self.csv_file)

    def dataloader(self):
        """Create and return a dataloader for the dataset"""
        return torch.utils.data.DataLoader(
            self,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=lambda x: x,
        )

    def __getitem__(self, index):
        nc_path = self.patch_files[index]

        if self.selection == "rgb":
            bands = ["B04", "B03", "B02"]
        elif self.selection == "all":
            bands = ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B10", "B11", "B12"]
        else:
            raise ValueError(f"Invalid selection: {self.selection}")
        
        ds_ref = Dataset(nc_path, mode="r")
        target_shape = ds_ref.groups["B02"].variables["B02"][:].shape  

        band_arrays = []
        for band in bands:
            ds_band = Dataset(nc_path, mode="r") 
            arr = ds_band.groups[band].variables[band][:]  # shape (T, H, W)

            # Resample only if spatial shape is different (differnt bands have different sizes)
            if (arr.shape[1], arr.shape[2]) != (target_shape[1], target_shape[2]):
                zoom_factors = (
                    1,  # time dimension unchanged
                    target_shape[1] / arr.shape[1],
                    target_shape[2] / arr.shape[2]
                )
                arr = zoom(arr, zoom_factors, order=1)  

            if self.date is not None:
                # select only the specified time slice
                arr = arr[self.date, :, :]  
                arr = np.expand_dims(arr, axis=0)  

            arr = np.nan_to_num(arr)
            band_arrays.append(arr)

        image = np.stack(band_arrays, axis=-1) # stack bands

        if self.date is not None:
            image = np.squeeze(image, axis=0)

        image = image.astype(np.float32) / 10000.0
        image = np.clip(image, 0, 1)

        if self.date is None:
            image = torch.tensor(image).permute(3, 0, 1, 2) 
        else:
            image = torch.tensor(image).permute(2, 0, 1) 

        ds_labels = Dataset(nc_path, mode="r")
        mask = ds_labels.groups["labels"].variables["labels"][:]
        masks = [(mask == v) for v in self.label_ids]
        mask = np.stack(masks, axis=-1).astype(np.float32)
        mask = torch.tensor(mask).permute(2, 0, 1)  

        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            mask = self.target_transform(mask)
        return image, mask
    
    def __len__(self):
        return len(self.patch_files)
    
    def load_dataset(self, data_dir, csv_file=None):
        if not self.labels:
            raise ValueError("You need to provide the list of labels for the dataset")

        self.patch_files = []

        allowed_files = None
        if csv_file:
            df = pd.read_csv(csv_file)
            allowed_files = set(df['patch_file'].values)

        for year_folder in os.listdir(data_dir):
            year_path = os.path.join(data_dir, year_folder)
            if os.path.isdir(year_path):
                for tile_folder in os.listdir(year_path):
                    tile_path = os.path.join(year_path, tile_folder)
                    if os.path.isdir(tile_path):
                        for file_name in os.listdir(tile_path):
                            if file_name.endswith('.nc'):
                                if (allowed_files is None) or (file_name in allowed_files):
                                    full_path = os.path.join(tile_path, file_name)
                                    self.patch_files.append(full_path)
        return self.patch_files
                         
    def show_image(self, index, show_title=False):
        img, mask = self.__getitem__(index)
        
        if isinstance(img, torch.Tensor):
            img = img.permute(1, 2, 0).cpu().numpy()

        img_mask = np.zeros([mask.shape[1], mask.shape[2], 3], dtype=np.uint8)
        legend_elements = []
        for i, label in enumerate(self.labels):
            legend_elements.append(
                Patch(
                    facecolor=tuple([x / 255 for x in self.color_mapping[i]]),
                    label=label,
                )
            )
            img_mask[np.where(mask[i] == 1)] = self.color_mapping[i]

        fig = plt.figure(figsize=(10, 8))
        height_factor = 0.81
        legend_width = 0.6 if self.class_selection == "12" else 1.0
        font_size = 12 if self.class_selection == "12" else 9
        xanchor = 0.2 if self.class_selection == "12" else 0

        fig.legend(handles=legend_elements, bbox_to_anchor=(xanchor, height_factor, legend_width, 0.2), ncol=3, mode='expand',
                   loc='lower left', prop={'size': font_size})
        plt.subplot(1, 2, 1)
        plt.imshow(img)
        plt.axis("off")

        plt.subplot(1, 2, 2)
        plt.imshow(img_mask)
        plt.axis("off")

        fig.tight_layout()
        plt.show()
        return fig


    def show_timeseries(self, index, latitude, longitude):
        image, mask = self.__getitem__(index)

        if image.ndim == 3: 
            raise ValueError("No time dimension available; set 'date=None' to get timeseries data.")

        # Define band names based on selection
        if self.selection == "rgb":
            bands = ["B04", "B03", "B02"]
        elif self.selection == "all":
            bands = ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B10", "B11", "B12"]
        else:
            raise ValueError(f"Invalid selection: {self.selection}")

        timeseries = []
        for band_idx in range(image.shape[0]):
            ts = image[band_idx, :, latitude, longitude].cpu().numpy()
            timeseries.append(ts)

        timeseries = np.array(timeseries)

        # find the pixel label
        mask_np = mask.cpu().numpy()
        label_vector = mask_np[:, latitude, longitude]
        crop_name = "Unknown"
        if label_vector.sum() > 0:
            crop_idx = label_vector.argmax()
            crop_name = self.labels[crop_idx]

        fig, ax = plt.subplots(figsize=(12, 6))
        ax.set_title(f"Timeseries with index {index} from the region with pixel coordinates ({latitude}, {longitude}), with label {crop_name}")

        for band_idx in range(timeseries.shape[0]):
            ax.plot(timeseries[band_idx], label=bands[band_idx])  

        ax.set_xlabel("Time step")
        ax.set_ylabel("Reflectance (ρ)")
        ax.legend(loc='upper right')
        ax.grid(True)
        plt.tight_layout()
        plt.show()

        return fig

    def data_distribution_table(self):
        label_dist = {key: 0 for key in self.labels}
        for i in range(len(self.patch_files)): 
            _, mask = self[i]  # mask shape: [C, H, W]
            mask_np = mask.cpu().numpy() if torch.is_tensor(mask) else mask
            for idx, label in enumerate(self.labels):
                label_dist[label] += mask_np[idx].sum()
        label_count = pd.DataFrame.from_dict(label_dist, orient='index', columns=["Number of pixels"])
        label_count = label_count.astype(float)
        return label_count

    def data_distribution_barchart(self, show_title=True):
        label_count = self.data_distribution_table()
        fig, ax = plt.subplots(figsize=(12, 10))
        sns.barplot(data=label_count, x=label_count.index, y='Number of pixels', ax=ax)
        if show_title:
            ax.set_title(
                "Labels distribution for {}".format(self.get_name()), pad=20, fontsize=18
            )
        return fig