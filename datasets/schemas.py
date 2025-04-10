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