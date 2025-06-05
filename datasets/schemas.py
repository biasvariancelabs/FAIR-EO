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