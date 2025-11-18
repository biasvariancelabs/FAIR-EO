from marshmallow import fields, validate
from ..base.schemas import BaseDatasetSchema

class FAIREOSchema(BaseDatasetSchema):
    """
    Schema for configuring FAIR-EO datasets.
    """
    csv_file = fields.String(
        missing=None, description="CSV file on disk", example="./data/train.csv"
    )
    data_dir = fields.String(
        missing="/", description="Dataset path on disk", example="./data/CESBIO/"
    )
    selection = fields.String(
        missing="rgb", description="Read RGB channels or more than 3 channels", example="all/rgb"
    )
    mode = fields.String(
        missing=None, description="train/val or test task", example="train"
    )

class FAIREOObjectDetectionYoloSchema(BaseDatasetSchema):
    """
    Schema for configuring an object detection dataset given in YOLO format.
    """
    data_dir = fields.String(
        missing="/", description="Dataset path on disk", example="./data/DIOR/"
    )
    annotation_dir = fields.String(
        missing='/',
        description="txt Yolo files path on disk",
        example="./data/annotations/"
    )
    csv_file = fields.String(
        missing=None, description="CSV file on disk", example="./data/train.csv"
    )
    task = fields.String(
        missing=None, description="training, validation or test task", example="train"
    )

class FAIREOObjectDetectionPascalSchema(BaseDatasetSchema):
    """
    Schema for configuring an object detection dataset given in PASCAL VOC format.
    """
    image_dir = fields.String(
        missing="/", description="Folder to the images on disk", example="./data/DIOR/"
    )
    annotations_dir = fields.String(
        missing="/",
        description="Folder with the XML annotations in VOC format",
        example="./data/DIOR/Annons/",
    )
    csv_file = fields.String(
        missing=None, description="CSV file on disk", example="./data/train.csv"
    )

class FAIREOObjectDetectionCocoSchema(BaseDatasetSchema):
    """
    Schema for configuring an object detection dataset given in COCO format.
    """
    data_dir = fields.String(
        missing="/", description="Dataset path on disk", example="./data/DIOR/"
    )
    annotation_dir = fields.String(
        missing='/', description="json COCO files path on disk", example="./data/annotations/"
    )
    csv_file = fields.String(
        missing=None, description="CSV file on disk", example="./data/train.csv"
    )
    selection = fields.String(
        missing=None, description="Include all images or only images with labels", example="all"
    )

class MatDatasetSchema(BaseDatasetSchema):
    """
    Schema for configuring datasets given in .mat format.
    """
    mat_file = fields.String(
        missing="/", description="Dataset path on disk", example="./data/CESBIO/"
    )
    mode = fields.String(
	missing=None, description="train/val/test", example="train"
    )
    csv_file = fields.String(
        missing=None, description="CSV file on disk", example="./data/train.csv"
    )

class LLCCDatasetSchema(BaseDatasetSchema):
    """
    Schema for configuring the LLCC image classification dataset which has triplets of images.
    """
    data_dir = fields.String(
        missing="/", description="Dataset path on disk", example="./data/BigEarthNet/"
    )
    csv_file = fields.String(
        missing=None, description="CSV file on disk", example="./data/train.csv",
    )
    selection = fields.String(
        missing="/", description="Use images as triplets or as individual images", example="individual"
    )
    task = fields.String(
        missing=None, description="training, validation or test task", example="training"
    )

class OSMDatasetSchema(BaseDatasetSchema):
    """
    Schema for configuring the OSM dataset.
    """
    csv_file = fields.String(
        missing=None, description="CSV file on disk", example="./data/train.csv"
    )
    data_dir = fields.String(
        missing="/", description="Dataset path on disk", example="./data/CESBIO/"
    )
    classification = fields.String(
        missing="/", description="Choose the classification schema (land_use, water_and_roads, buildings)", example="land_use"
    )
    mode = fields.String(
        missing=None, description="train/val or test task", example="train"
    )

class SpaceNet8DatasetSchema(BaseDatasetSchema):
    """
    Schema for configuring the SpaceNet8 dataset.
    """
    data_dir = fields.String(
        missing="/", description="Dataset path on disk", example="./Germany_Training_Public/"
    )
    mode = fields.String(
        missing="post-event",
        description="Use only post-event images or pre-/post-event pairs",
        example="pairs",
    )

class SACropTypeSchema(BaseDatasetSchema):
    """
    Schema for configuring SA Crop Type dataset.
    """
    csv_file = fields.String(
        missing=None, description="CSV file on disk", example="./data/train.csv"
    )
    data_dir = fields.String(
        missing="/", description="Dataset path on disk", example="./data/CESBIO/"
    )
    selection = fields.String(
        missing="rgb", description="Read RGB channels or 13 channels", example="all/rgb"
    )
    date = fields.Int(
        missing=None, description="Read only one image out of a time series", example=1
    )

class MTS12Schema(BaseDatasetSchema):
    """
    Schema for configuring the MTS12 dataset.
    """
    data_dir = fields.String(
        missing="/", description="Dataset path on disk", example="./data/CESBIO/"
    )
    selection = fields.String(
        missing="rgb", description="Read RGB channels or more than 3 channels", example="all/rgb"
    )
    date = fields.String(
        missing=None, description="Visualise images for a particular date", example="1"
    )  
    imagery = fields.String(
        missing="s2", description="Select which Sentinel imagery to use: 's1' for Sentinel-1, 's2' for Sentinel-2", example="s2"
    )
    split = fields.String(
        missing="train", description="Select which split to use, train, val or test", example="train"
    )

class Sen4AgriNetSchema(BaseDatasetSchema):
    csv_file = fields.String(
        missing=None, description="CSV file on disk", example="./data/train.csv"
    )
    data_dir = fields.String(
        missing="/", description="Dataset path on disk", example="./data/CESBIO/"
    )
    selection = fields.String(
        missing="rgb", description="Read RGB channels or more than 3 channels", example="all/rgb"
    )
    date = fields.String(
        missing=None, description="Visualise images for a particular date", example="1"
    )
    class_selection = fields.String(
        missing = "12", description = "Select the 12 core classes or all 168 classes", example="12/all"
    )

class BrazilDAMSchema(BaseDatasetSchema):
    """
    Schema for configuring the BrazilDAM dataset.
    """
    data_dir = fields.String(
        missing="/", description="Dataset path on disk", example="./data/CESBIO/"
    )
    csv_file = fields.String(
        missing=None, description="CSV file on disk", example="./data/train.csv"
    )
    imagery = fields.String(
        missing="sentinel", description="Select which imagery to use: 'sentinel', 'landsat' or 'sentinel and landsat'", example="sentinel"
    )