<div align="center">
<h2>FAIR-EO: AI-ready Earth observation datasets for the AiTLAS toolbox</h2>

**Bias Variance Labs**

<br>
<a href="https://github.com/biasvariancelabs/aitlas"><img src="https://img.shields.io/badge/AiTLAS-2.0.2-1C6B73"></a>
<a href="https://eodata.bvlabs.ai/ai4eo/#/data"><img src="https://img.shields.io/badge/FAIR--EO%20Hub-dataset%20metadata-E8961E"></a>
<a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-blue"></a>
</div>

## Summary

FAIR-EO provides data loaders, example notebooks and data splits for 278 publicly available Earth observation (EO) datasets, ready to use with the [AiTLAS](https://github.com/biasvariancelabs/aitlas) toolbox. It was developed as part of the OSCARS-funded [FAIR-EO](https://oscars-project.eu/projects/fair-eo-fair-open-and-ai-ready-earth-observation-resources) project.

EO datasets come in many formats, with different label conventions and often without fixed splits, which makes them slow to use and makes results hard to compare. FAIR-EO gives every dataset a loader with the same interface, an example notebook that loads, explores and trains on it, and, where the original authors did not define them, train, validation and test splits. In line with the FAIR principles (Findable, Accessible, Interoperable and Reusable), the metadata of each dataset, including its license and provenance, is available in the [FAIR-EO Hub](https://eodata.bvlabs.ai/ai4eo/#/data).

## What is in this repository

| Folder | Contents |
| --- | --- |
| [`datasets/`](datasets) | A loader for each dataset, built on AiTLAS |
| [`examples/`](examples) | Jupyter notebooks: one for each dataset, plus examples with foundation models |
| [`splits/`](splits) | Train, validation and test splits for the datasets that had none |

## Installation

AiTLAS needs Python 3.12. Clone the repository and install the requirements, which pin AiTLAS 2.0.2 and bring PyTorch and the other dependencies with it:

```bash
git clone https://github.com/biasvariancelabs/FAIR-EO.git
cd FAIR-EO
pip install -r requirements.txt
```

We do not host the datasets. Download each one from its original source, linked in the tables below and in its loader.

## Quick start

Each notebook in [`examples/`](examples) loads, explores and trains on one dataset: set `data_dir`, and `csv_file` where it is used, to where you saved the data, and run it.

In your own code, import the loaders from the `datasets` package at the root of this repository. For example, to train a Vision Transformer on [AIDER](https://zenodo.org/records/3888300) with our splits, run this from the root of the repository:

```python
from aitlas.models import VisionTransformer
from aitlas.transforms import ResizeCenterCropToTensor
from datasets import AIDERDataset

train = AIDERDataset({"data_dir": "data/AIDER", "csv_file": "splits/aider_train.csv",
                      "batch_size": 32, "shuffle": True})
val = AIDERDataset({"data_dir": "data/AIDER", "csv_file": "splits/aider_val.csv",
                    "batch_size": 32})
train.transform = val.transform = ResizeCenterCropToTensor()

model = VisionTransformer({"num_classes": 5, "pretrained": True})
model.prepare()
model.train_and_evaluate_model(train_dataset=train, val_dataset=val, epochs=10,
                               model_directory="experiments/AIDER")
```

From a script or notebook elsewhere, add the repository root to the path first, as the notebooks in `examples/` do with `sys.path.insert(0, "..")`.

## Datasets

The loaders cover 278 datasets in seven tasks. A few datasets have more than one loader, for example BigEarthNet, with one for 19 and one for 43 classes, so there are 283 loaders in all.

| Task | Loaders | Examples |
| --- | --- | --- |
| [Multi-class classification](#multi-class-classification) | 60 | EuroSAT, RESISC45, UC Merced, So2Sat |
| [Multi-label classification](#multi-label-classification) | 13 | BigEarthNet, MLRSNet, Planet UAS, RapidAI4EO |
| [Semantic segmentation](#semantic-segmentation) | 115 | LoveDA, DeepGlobe, Potsdam, CloudSEN12 |
| [Object detection](#object-detection) | 49 | xView, FAIR1M, VEDAI, HRSID |
| [Change detection](#change-detection) | 29 | LEVIR-CD, S2Looking, OSCD, xView2 |
| [Time series](#time-series) | 15 | PASTIS, BreizhCrops, CropHarvest, BioMassters |
| [Anomaly detection](#anomaly-detection) | 2 | Sentinel-1 and Sentinel-2 anomalies (AI4QC) |

Each table links the dataset's source, its loader, its example notebook and, where we made them, its splits. "Predefined" means that the dataset comes with its own splits.

### Multi-class classification

<details>
<summary>Show the list</summary>

| Dataset | Loader | Notebook | Splits |
| --- | --- | --- | --- |
| [Aerial Cactus Identification](https://www.kaggle.com/c/aerial-cactus-identification/data) | [`aerial_cactus_identification.py`](datasets/aerial_cactus_identification.py) | [Notebook](examples/multiclass_classification_example_aerial_cactus_identification.ipynb) | Predefined |
| [AID](https://www.kaggle.com/datasets/jiayuanchengala/aid-scene-classification-datasets) | [`aid.py`](datasets/aid.py) | [Notebook](examples/multiclass_classification_example_aid.ipynb) | Predefined |
| [AIDER](https://zenodo.org/records/3888300) | [`aider.py`](datasets/aider.py) | [Notebook](examples/multiclass_classification_example_aider.ipynb) | [train](splits/aider_train.csv) · [val](splits/aider_val.csv) · [test](splits/aider_test.csv) |
| [Airbus Wind Turbines](https://www.kaggle.com/datasets/airbusgeo/airbus-wind-turbines-patches) | [`airbus_wind_turbines.py`](datasets/airbus_wind_turbines.py) | [Notebook](examples/multiclass_classification_example_airbus_wind_turbines.ipynb) | Predefined |
| [AiRound](https://www.dropbox.com/scl/fi/qgjxoene22cor54fzr20v/Gabriel-airoundv2.zip?rlkey=3n3knpid5r48jilf1bag43m6h&e=1&dl=0) | [`airound.py`](datasets/airound.py) | [Notebook](examples/multiclass_classification_example_airound.ipynb) | [train](splits/airound_train.csv) · [val](splits/airound_val.csv) · [test](splits/airound_test.csv) |
| [Austin Zoning](https://www.kaggle.com/datasets/franchenstein/austin-zoning-satellite-images) | [`austin_zoning.py`](datasets/austin_zoning.py) | [Notebook](examples/multiclass_classification_example_austin_zoning.ipynb) | [train](splits/austin_zoning_train.csv) · [val](splits/austin_zoning_val.csv) · [test](splits/austin_zoning_test.csv) |
| [Bijie Landslide](http://gpcv.whu.edu.cn/data/Bijie_pages.html) | [`bijie_landslide.py`](datasets/bijie_landslide.py) | [Notebook](examples/multiclass_classification_example_bijie_landslide.ipynb) | [train](splits/bijie_landslide_train.csv) · [val](splits/bijie_landslide_val.csv) · [test](splits/bijie_landslide_test.csv) |
| [BrazilDAM](https://drive.google.com/drive/folders/1v1F4faAD8zCm_vocGxILiIUncaRz1pZB) | [`brazildam.py`](datasets/brazildam.py) | [Notebook](examples/multiclass_classification_example_brazildam.ipynb) | [train](splits/brazildam_train.csv) · [val](splits/brazildam_val.csv) · [test](splits/brazildam_test.csv) |
| [Brazilian Coffee Scenes](http://www.patreo.dcc.ufmg.br/wp-content/uploads/2017/11/brazilian_coffee_dataset.zip) | [`brazilian_coffee_scenes.py`](datasets/brazilian_coffee_scenes.py) | [Notebook](examples/multiclass_classification_example_brazilian_coffee_scenes.ipynb) | Predefined |
| [Cactus Aerial Photos](https://www.kaggle.com/datasets/irvingvasquez/cactus-aerial-photos) | [`cactus_aerial_photos.py`](datasets/cactus_aerial_photos.py) | [Notebook](examples/multiclass_classification_example_cactus_aerial_photos.ipynb) | Predefined |
| [Canadian Cropland](https://drive.google.com/drive/folders/1mNI8B5EMk0Xgvx2Pc9ztnQRaW9pXh8yb) | [`canadian_cropland.py`](datasets/canadian_cropland.py) | [Notebook](examples/multiclass_classification_example_canadian_cropland.ipynb) | Predefined |
| [CLRS](https://github.com/lehaifeng/CLRS) | [`clrs.py`](datasets/clrs.py) | [Notebook](examples/multiclass_classification_example_clrs.ipynb) | Predefined |
| [COWC](https://gdo152.llnl.gov/cowc/download/) | [`cowc_image_classification.py`](datasets/cowc_image_classification.py) | [Notebook](examples/multiclass_classification_example_cowc_image_classification.ipynb) | [train](splits/cowc_image_classification_train.csv) · [val](splits/cowc_image_classification_val.csv) · [test](splits/cowc_image_classification_test.csv) |
| [COWC-M](https://gdo152.llnl.gov/cowc/download/) | [`cowc_m_image_classification.py`](datasets/cowc_m_image_classification.py) | [Notebook](examples/multiclass_classification_example_cowc_m_image_classification.ipynb) | [train](splits/cowc_m_image_classification_train.csv) · [val](splits/cowc_m_image_classification_val.csv) · [test](splits/cowc_m_image_classification_test.csv) |
| [CV-BrCT](https://drive.google.com/file/d/1GK5haKw7-Cr479NW2JPosMOkl14W4cxY/view) | [`cv_brct.py`](datasets/cv_brct.py) | [Notebook](examples/multiclass_classification_example_cv_brct.ipynb) | [train](splits/cvbrct_train.csv) · [val](splits/cvbrct_val.csv) · [test](splits/cvbrct_test.csv) |
| [ERA](https://drive.google.com/file/d/1yxXjDNAq5RAufSgSOE4QmdsNfAc4dOfM/view) | [`era.py`](datasets/era.py) | [Notebook](examples/multiclass_classification_example_era.ipynb) | [train](splits/era_train.csv) · [val](splits/era_val.csv) · [test](splits/era_test.csv) |
| [EuroCropsML](https://zenodo.org/records/15095445) | [`eurocropsml.py`](datasets/eurocropsml.py) | [Notebook](examples/multiclass_classification_example_eurocropsml.ipynb) | Predefined |
| [EuroSAT](https://github.com/phelber/EuroSAT) | [`eurosat.py`](datasets/eurosat.py) | [Notebook](examples/multiclass_classification_example_eurosat.ipynb) | Predefined |
| [Find A Car Park](https://www.kaggle.com/datasets/daggysheep/find-a-car-park) | [`find_a_car_park.py`](datasets/find_a_car_park.py) | [Notebook](examples/multiclass_classification_example_find_a_car_park.ipynb) | [train](splits/find_a_car_park_train.csv) · [val](splits/find_a_car_park_val.csv) · [test](splits/find_a_car_park_test.csv) |
| [FireRisk](https://drive.google.com/file/d/1J5GrJJPLWkpuptfY_kgqkiDtcSNP88OP/view) | [`firerisk.py`](datasets/firerisk.py) | [Notebook](examples/multiclass_classification_example_firerisk.ipynb) | [train](splits/firerisk_train.csv) · [val](splits/firerisk_val.csv) · [test](splits/firerisk_test.csv) |
| [Hephaestus](https://www.dropbox.com/scl/fi/qx99z7tlx6994r7zn4fgh/Hephaestus_Classification.zip?rlkey=lnnmakeu50au0r50yjbjonivx&e=1&dl=0) | [`hephaestus_classification.py`](datasets/hephaestus_classification.py) | [Notebook](examples/multiclass_classification_example_hephaestus_classification.ipynb) | [train](splits/hephaestus_classification_train.csv) · [val](splits/hephaestus_classification_val.csv) · [test](splits/hephaestus_classification_test.csv) |
| [HistAerial](http://eidolon.univ-lyon2.fr/~remi1/HistAerialDataset/) | [`histaerial.py`](datasets/histaerial.py) | [Notebook](examples/multiclass_classification_example_histaerial.ipynb) | Predefined |
| [Iceberg Classifier](https://www.kaggle.com/competitions/statoil-iceberg-classifier-challenge/overview) | [`iceberg_challenge.py`](datasets/iceberg_challenge.py) | [Notebook](examples/multiclass_classification_example_iceberg_challenge.ipynb) | Predefined |
| [m-brick-kiln](https://zenodo.org/records/8276815) | [`m_brick_kiln.py`](datasets/m_brick_kiln.py) | [Notebook](examples/multiclass_classification_example_m_brick_kiln.ipynb) | Predefined |
| [m-pv4ger](https://github.com/allenai/satlas/blob/main/SatlasPretrain.md) | [`m_pv4ger.py`](datasets/m_pv4ger.py) | [Notebook](examples/multiclass_classification_example_m_pv4ger.ipynb) | Predefined |
| [MASATI](https://docs.google.com/forms/d/e/1FAIpQLSe1qeo_XH2v_obix_huLVVqWpGL_1DAZlMTVlEeMuy-EHdH9w/viewform?c=0&w=1) | [`masati_image_classification.py`](datasets/masati_image_classification.py) | [Notebook](examples/multiclass_classification_example_masati_image_classification.ipynb) | [train](splits/masati_image_classification_train.csv) · [val](splits/masati_image_classification_val.csv) · [test](splits/masati_image_classification_test.csv) |
| [MSTAR-8](https://www.kaggle.com/datasets/atreyamajumdar/mstar-dataset-8-classes/data) | [`mstar8.py`](datasets/mstar8.py) | [Notebook](examples/multiclass_classification_example_mstar8.ipynb) | [train](splits/mstar8_train.csv) · [val](splits/mstar8_val.csv) · [test](splits/mstar8_test.csv) |
| [MTARSI](https://zenodo.org/records/3464319#.Yxi6e-xByL4) | [`mtarsi.py`](datasets/mtarsi.py) | [Notebook](examples/multiclass_classification_example_mtarsi.ipynb) | [train](splits/mtarsi_train.csv) · [val](splits/mtarsi_val.csv) · [test](splits/mtarsi_test.csv) |
| [NaSC-TG2](https://huggingface.co/datasets/jonathan-roberts1/NaSC-TG2/tree/main) | [`nasc_tg2.py`](datasets/nasc_tg2.py) | [Notebook](examples/multiclass_classification_example_nasc_tg2.ipynb) | [train](splits/nasc_tg2_train.csv) · [val](splits/nasc_tg2_val.csv) · [test](splits/nasc_tg2_test.csv) |
| [OpenSARShip](https://opensar.sjtu.edu.cn/index.html) | [`opensarship.py`](datasets/opensarship.py) | [Notebook](examples/multiclass_classification_example_opensarship.ipynb) | [train](splits/opensarship_train.csv) · [val](splits/opensarship_val.csv) · [test](splits/opensarship_test.csv) |
| [OPTIMAL-31](https://drive.google.com/file/d/1Fk9a0DW8UyyQsR8dP2Qdakmr69NVBhq9/view) | [`optimal_31.py`](datasets/optimal_31.py) | [Notebook](examples/multiclass_classification_example_optimal_31.ipynb) | [train](splits/optimal_31_train.csv) · [val](splits/optimal_31_val.csv) · [test](splits/optimal_31_test.csv) |
| [Overhead-MNIST](https://www.kaggle.com/datasets/datamunge/overheadmnist) | [`overhead_mnist.py`](datasets/overhead_mnist.py) | [Notebook](examples/multiclass_classification_example_overhead_mnist.ipynb) | Predefined |
| [PatternNet](https://sites.google.com/view/zhouwx/dataset) | [`pattern_net.py`](datasets/pattern_net.py) | [Notebook](examples/multiclass_classification_example_pattern_net.ipynb) | Predefined |
| [PlanesNet](https://www.kaggle.com/datasets/rhammell/planesnet/data) | [`planesnet.py`](datasets/planesnet.py) | [Notebook](examples/multiclass_classification_example_planesnet.ipynb) | [train](splits/planesnet_train.csv) · [val](splits/planesnet_val.csv) · [test](splits/planesnet_test.csv) |
| [Post Hurricane](https://www.kaggle.com/datasets/kmader/satellite-images-of-hurricane-damage) | [`post_hurricane.py`](datasets/post_hurricane.py) | [Notebook](examples/multiclass_classification_example_post_hurricane.ipynb) | Predefined |
| [Representation Learning for Land Cover Classification](https://sustainlab-group.github.io/sustainbench/docs/datasets/sdg15/land_cover_representation.html) | [`llcc.py`](datasets/llcc.py) | [Notebook](examples/multiclass_classification_example_llcc.ipynb) | Predefined |
| [RESISC45](https://www.tensorflow.org/datasets/catalog/resisc45) | [`resisc45.py`](datasets/resisc45.py) | [Notebook](examples/multiclass_classification_example_resisc45.ipynb) | Predefined |
| [RSC11](https://huggingface.co/datasets/jonathan-roberts1/RS_C11) | [`rsc11.py`](datasets/rsc11.py) | [Notebook](examples/multiclass_classification_example_rsc11.ipynb) | [train](splits/rsc11_train.csv) · [val](splits/rsc11_val.csv) · [test](splits/rsc11_test.csv) |
| [RSD46-WHU](https://github.com/RSIA-LIESMARS-WHU/RSD46-WHU) | [`rsd46_whu.py`](datasets/rsd46_whu.py) | [Notebook](examples/multiclass_classification_example_rsd46_whu.ipynb) | Predefined |
| [RSI-CB128](https://www.kaggle.com/datasets/noamaanabdulazeem/myfile) | [`rsi_cb128.py`](datasets/rsi_cb128.py) | [Notebook](examples/multiclass_classification_example_rsi_cb128.ipynb) | [train](splits/rsi_cb128_train.csv) · [val](splits/rsi_cb128_val.csv) · [test](splits/rsi_cb128_test.csv) |
| [RSI-CB256](https://figshare.com/articles/dataset/RSI-CB256/22139921?file=39360821) | [`rsi_cb256.py`](datasets/rsi_cb256.py) | [Notebook](examples/multiclass_classification_example_rsi_cb256.ipynb) | [train](splits/rsi_cb256_train.csv) · [val](splits/rsi_cb256_val.csv) · [test](splits/rsi_cb256_test.csv) |
| [RSSCN7](https://www.kaggle.com/datasets/nifulislam/rsscn7-dataset) | [`rsscn7.py`](datasets/rsscn7.py) | [Notebook](examples/multiclass_classification_example_rsscn7.ipynb) | Predefined |
| [S2UV](https://www.scidb.cn/en/detail?dataSetId=14e27d8c51ec40079b84591e9bb24df6#p2) | [`s2uv.py`](datasets/s2uv.py) | [Notebook](examples/multiclass_classification_example_s2uv.ipynb) | Predefined |
| [SAR-ACD](https://github.com/AICyberTeam/SAR-ACD) | [`sar_acd.py`](datasets/sar_acd.py) | [Notebook](examples/multiclass_classification_example_sar_acd.ipynb) | [train](splits/sar_acd_train.csv) · [val](splits/sar_acd_val.csv) · [test](splits/sar_acd_test.csv) |
| [SAT-4](https://www.kaggle.com/datasets/crawford/deepsat-sat4/data) | [`sat4.py`](datasets/sat4.py) | [Notebook](examples/multiclass_classification_example_sat4.ipynb) | Predefined |
| [SAT-6](http://csc.lsu.edu/~saikat/deepsat/) | [`sat6.py`](datasets/sat6.py) | [Notebook](examples/multiclass_classification_example_sat6.ipynb) | Predefined |
| [Satellite Image Classification](https://www.kaggle.com/datasets/mahmoudreda55/satellite-image-classification) | [`satellite_image_classification.py`](datasets/satellite_image_classification.py) | [Notebook](examples/multiclass_classification_example_satellite_image_classification.ipynb) | [train](splits/satellite_image_classification_train.csv) · [val](splits/satellite_image_classification_val.csv) · [test](splits/satellite_image_classification_test.csv) |
| [SatlasPretrain](https://github.com/allenai/satlas/blob/main/SatlasPretrain.md) | [`satlaspretrain_image_classification.py`](datasets/satlaspretrain_image_classification.py) | [Notebook](examples/multiclass_classification_example_satlaspretrain_image_classification.ipynb) | Predefined |
| [ShipsNet](https://www.kaggle.com/datasets/rhammell/ships-in-satellite-imagery) | [`shipsnet.py`](datasets/shipsnet.py) | [Notebook](examples/multiclass_classification_example_shipsnet.ipynb) | [train](splits/shipsnet_train.csv) · [val](splits/shipsnet_val.csv) · [test](splits/shipsnet_test.csv) |
| [SIRI-WHU](http://www.lmars.whu.edu.cn/prof_web/zhongyanfei/e-code.html) | [`siri_whu.py`](datasets/siri_whu.py) | [Notebook](examples/multiclass_classification_example_siri_whu.ipynb) | Predefined |
| [So2Sat](https://github.com/zhu-xlab/So2Sat-LCZ42) | [`so2sat.py`](datasets/so2sat.py) | [Notebook](examples/multiclass_classification_example_so2sat.ipynb) | Predefined |
| [Space2Ground](https://zenodo.org/records/6458006) | [`space2ground.py`](datasets/space2ground.py) | [Notebook](examples/multiclass_classification_example_space2ground.ipynb) | Predefined |
| [SynthWakeSAR](https://data.bris.ac.uk/data/dataset/30kvuvmatwzij2mz1573zqumfx) | [`synthwakesar.py`](datasets/synthwakesar.py) | [Notebook](examples/multiclass_classification_example_synthwakesar.ipynb) | d: [train](splits/synthwakesar_d_train.csv) · [val](splits/synthwakesar_d_val.csv) · [test](splits/synthwakesar_d_test.csv)<br>n: [train](splits/synthwakesar_n_train.csv) · [val](splits/synthwakesar_n_val.csv) · [test](splits/synthwakesar_n_test.csv)<br>nf: [train](splits/synthwakesar_nf_train.csv) · [val](splits/synthwakesar_nf_val.csv) · [test](splits/synthwakesar_nf_test.csv) |
| [TenGeoP-SARwv](https://www.seanoe.org/data/00456/56796/) | [`tengeop_sarwv.py`](datasets/tengeop_sarwv.py) | [Notebook](examples/multiclass_classification_example_tengeop_sarwv.ipynb) | [train](splits/tengeop_sarwv_train.csv) · [val](splits/tengeop_sarwv_val.csv) · [test](splits/tengeop_sarwv_test.csv) |
| [TreeSatAI](https://zenodo.org/records/6780578) | [`treesatai.py`](datasets/treesatai.py) | [Notebook](examples/multiclass_classification_example_treesatai.ipynb) | Predefined |
| [UC Merced](https://www.kaggle.com/datasets/abdulhasibuddin/uc-merced-land-use-dataset) | [`uc_merced.py`](datasets/uc_merced.py) | [Notebook](examples/multiclass_classification_example_uc_merced.ipynb) | Predefined |
| [USTC-SmokeRS](http://complex.ustc.edu.cn/2019/0802/c18202a389656/page.htm) | [`ustc_smokers.py`](datasets/ustc_smokers.py) | [Notebook](examples/multiclass_classification_example_ustc_smokers.ipynb) | [train](splits/ustc_smokers_train.csv) · [val](splits/ustc_smokers_val.csv) · [test](splits/ustc_smokers_test.csv) |
| [Weakly Supervised Cropland Segmentation](https://sustainlab-group.github.io/sustainbench/docs/datasets/sdg2/weakly_supervised_cropland.html) | [`wscs_image_classification.py`](datasets/wscs_image_classification.py) | [Notebook](examples/multiclass_classification_example_wscs_image_classification.ipynb) | Predefined |
| [WHU-RS19](https://github.com/CAPTAIN-WHU/BED4RS) | [`whu_rs19.py`](datasets/whu_rs19.py) | [Notebook](examples/multiclass_classification_example_whu_rs19.ipynb) | Predefined |
| [WiDS Datathon 2019](https://www.kaggle.com/competitions/widsdatathon2019) | [`wids_datathon_2019.py`](datasets/wids_datathon_2019.py) | [Notebook](examples/multiclass_classification_example_wids_datathon_2019.ipynb) | [train](splits/widsdatathon2019_train.csv) · [val](splits/widsdatathon2019_val.csv) · [test](splits/widsdatathon2019_test.csv) |

</details>

### Multi-label classification

<details>
<summary>Show the list</summary>

| Dataset | Loader | Notebook | Splits |
| --- | --- | --- | --- |
| [AID](https://github.com/Hua-YS/AID-Multilabel-Dataset) | [`aid_multilabel.py`](datasets/aid_multilabel.py) | [Notebook](examples/multilabel_classification_example_aid_multilabel.ipynb) | Predefined |
| [BigEarthNet (19 classes)](https://bigearth.net/) | [`bigearthnet_19.py`](datasets/bigearthnet_19.py) | [Notebook](examples/multilabel_classification_example_bigearthnet_19.ipynb) | Predefined |
| [BigEarthNet (43 classes)](https://bigearth.net/) | [`bigearthnet_43.py`](datasets/bigearthnet_43.py) | [Notebook](examples/multilabel_classification_example_bigearthnet_43.ipynb) | Predefined |
| [DFC15](https://github.com/Hua-YS/DFC15-Multilabel-Dataset) | [`dfc15_multilabel.py`](datasets/dfc15_multilabel.py) | [Notebook](examples/multilabel_classification_example_dfc15_multilabel.ipynb) | Predefined |
| [DLRSD](https://sites.google.com/view/zhouwx/dataset) | [`dlrsd_multilabel.py`](datasets/dlrsd_multilabel.py) | [Notebook](examples/multilabel_classification_example_dlrsd_multilabel.ipynb) | [train](splits/dlrsd_multilabel_train.csv) · [val](splits/dlrsd_multilabel_val.csv) · [test](splits/dlrsd_multilabel_test.csv) |
| [MLRSNet](https://data.mendeley.com/datasets/7j9bv9vwsx/2) | [`mlrs_net.py`](datasets/mlrs_net.py) | [Notebook](examples/multilabel_classification_example_mlrs_net.ipynb) | Predefined |
| [MultiScene](https://drive.google.com/drive/folders/17ty4I7HdihLcqRK3k_zDVtQqmPIYwLh2) | [`multiscene.py`](datasets/multiscene.py) | [Notebook](examples/multilabel_classification_example_multiscene.ipynb) | [train](splits/multiscene_train.csv) · [val](splits/multiscene_val.csv) · [test](splits/multiscene_test.csv) |
| [Planet UAS](https://www.kaggle.com/c/planet-understanding-the-amazon-from-space/overview) | [`planet_uas.py`](datasets/planet_uas.py) | [Notebook](examples/multilabel_classification_example_planet_uas.ipynb) | Predefined |
| [RapidAI4EO](https://rapidai4eo.source.coop/) | [`rapidai4eo.py`](datasets/rapidai4eo.py) | [Notebook](examples/multilabel_classification_example_rapidai4eo.ipynb) | Predefined |
| [SSL4EO-S12-ML](https://github.com/zhu-xlab/softcon) | [`ssl4eo_s12_ml.py`](datasets/ssl4eo_s12_ml.py) | [Notebook](examples/multilabel_classification_example_ssl4eo_s12_ml.ipynb) | [train](splits/ssl4eo_s12_ml_train.csv) · [val](splits/ssl4eo_s12_ml_val.csv) · [test](splits/ssl4eo_s12_ml_test.csv) |
| [TalloS](https://github.com/RolnickLab/FoMo-Bench/tree/main) | [`tallos.py`](datasets/tallos.py) | [Notebook](examples/multilabel_classification_example_tallos.ipynb) | [train](splits/tallos_train.csv) · [val](splits/tallos_val.csv) · [test](splits/tallos_test.csv) |
| [UC Merced](https://www.dropbox.com/scl/fi/dlrn78p72m1wxdyc1y2l9/UCMercedLanduse.zip?rlkey=hsu2tnwvszcb2i7ewoyuvt34p&e=2) | [`uc_merced_multilabel.py`](datasets/uc_merced_multilabel.py) | [Notebook](examples/multilabel_classification_example_uc_merced.ipynb) | Predefined |
| [WHDLD](https://sites.google.com/view/zhouwx/dataset) | [`whdld_multilabel.py`](datasets/whdld_multilabel.py) | [Notebook](examples/multilabel_classification_example_whdld_multilabel.ipynb) | [train](splits/whdld_multilabel_train.csv) · [val](splits/whdld_multilabel_val.csv) · [test](splits/whdld_multilabel_test.csv) |

</details>

### Semantic segmentation

<details>
<summary>Show the list</summary>

| Dataset | Loader | Notebook | Splits |
| --- | --- | --- | --- |
| [AeroRIT](https://github.com/aneesh3108/AeroRIT) | [`aerorit.py`](datasets/aerorit.py) | [Notebook](examples/semantic_segmentation_example_aerorit.ipynb) | Predefined |
| [Aeroscapes](https://drive.google.com/file/d/1WmXcm0IamIA0QPpyxRfWKnicxZByA60v/view) | [`aeroscapes.py`](datasets/aeroscapes.py) | [Notebook](examples/semantic_segmentation_example_aeroscapes.ipynb) | Predefined |
| [Agriculture Vision](https://www.agriculture-vision.com/agriculture-vision-2021/dataset-2021) | [`agriculture_vision.py`](datasets/agriculture_vision.py) | [Notebook](examples/semantic_segmentation_example_agriculture_vision.ipynb) | Predefined |
| [AgriFieldNet](https://source.coop/repositories/radiantearth/agrifieldnet-competition/access) | [`agrifieldnet.py`](datasets/agrifieldnet.py) | [Notebook](examples/semantic_segmentation_example_agrifieldnet.ipynb) | [train](splits/agrifieldnet_train.csv) · [val](splits/agrifieldnet_val.csv) · [test](splits/agrifieldnet_test.csv) |
| [AI4SmallFarms](https://phys-techsciences.datastations.nl/dataset.xhtml?persistentId=doi:10.17026/dans-xy6-ngg6) | [`ai4smallfarms.py`](datasets/ai4smallfarms.py) | [Notebook](examples/semantic_segmentation_example_ai4smallfarms.ipynb) | Predefined |
| [AIR-PolSAR-Seg](https://drive.google.com/drive/folders/1sGrmLmknhQ28nvasKbCqKvVj6rY2fCqL) | [`air_polsar_seg.py`](datasets/air_polsar_seg.py) | [Notebook](examples/semantic_segmentation_example_air_polsar_seg.ipynb) | [train](splits/air_polsar_seg_train.csv) · [val](splits/air_polsar_seg_val.csv) · [test](splits/air_polsar_seg_test.csv) |
| [Airbus Ship Detection Challenge](https://www.kaggle.com/competitions/airbus-ship-detection/overview) | [`airbus_ship_detection.py`](datasets/airbus_ship_detection.py) | [Notebook](examples/semantic_segmentation_example_airbus_ship_detection.ipynb) | Predefined |
| [AIRS](https://www.airs-dataset.com/) | [`airs.py`](datasets/airs.py) | [Notebook](examples/semantic_segmentation_example_airs.ipynb) | Predefined |
| [Amazon Rainforest](https://zenodo.org/record/3233081#.YTYm_44zaUk) | [`amazon_rainforest.py`](datasets/amazon_rainforest.py) | [Notebook](examples/semantic_segmentation_example_amazon_rainforest.ipynb) | Predefined |
| [APKLOT](https://github.com/langheran/APKLOT) | [`apklot.py`](datasets/apklot.py) | [Notebook](examples/semantic_segmentation_example_apklot.ipynb) | Predefined |
| [Arctic Sea Ice Image Masking](https://www.kaggle.com/datasets/alexandersylvester/arctic-sea-ice-image-masking) | [`arctic_sea_ice_image_masking.py`](datasets/arctic_sea_ice_image_masking.py) | [Notebook](examples/semantic_segmentation_example_arctic_sea_ice_image_masking.ipynb) | [train](splits/arctic_sea_ice_image_masking_train.csv) · [val](splits/arctic_sea_ice_image_masking_val.csv) · [test](splits/arctic_sea_ice_image_masking_test.csv) |
| [ASD](http://jiangyeyuan.com/ASD/Aerial%20Image%20Segmentation%20Dataset.html) | [`asd.py`](datasets/asd.py) | [Notebook](examples/semantic_segmentation_example_asd.ipynb) | [train](splits/asd_train.csv) · [val](splits/asd_val.csv) · [test](splits/asd_test.csv) |
| [BDCI2020](https://aistudio.baidu.com/datasetdetail/56051) | [`bdci2020.py`](datasets/bdci2020.py) | [Notebook](examples/semantic_segmentation_example_bdci2020.ipynb) | Predefined |
| [BH-Pools](https://drive.google.com/file/d/1Ph1Ubtk2jAYHhEO6whTTvdBKRkhvH7bZ/edit) | [`bh_pools.py`](datasets/bh_pools.py) | [Notebook](examples/semantic_segmentation_example_bh_pools.ipynb) | Predefined |
| [BH-Watertanks](https://drive.google.com/file/d/1Ph1Ubtk2jAYHhEO6whTTvdBKRkhvH7bZ/edit) | [`bh_watertanks.py`](datasets/bh_watertanks.py) | [Notebook](examples/semantic_segmentation_example_bh_watertanks.ipynb) | Predefined |
| [C2Seg](https://github.com/danfenghong/RSE_Cross-city?tab=readme-ov-file) | [`c2seg.py`](datasets/c2seg.py) | [Notebook](examples/semantic_segmentation_example_c2seg.ipynb) | Predefined |
| [CaBuAr](https://huggingface.co/datasets/DarthReca/california_burned_areas) | [`cabuar.py`](datasets/cabuar.py) | [Notebook](examples/semantic_segmentation_example_cabuar.ipynb) | Predefined |
| [CamVid](https://github.com/alexgkendall/SegNet-Tutorial) | [`camvid.py`](datasets/camvid.py) | [Notebook](examples/semantic_segmentation_example_camvid.ipynb) | Predefined |
| [CESBIO](https://drive.google.com/drive/folders/1M0dYz7Afkm471cea0-ez5UhLl4MvcnMI?usp=sharing) | [`cesbio.py`](datasets/cesbio.py) | [Notebook](examples/semantic_segmentation_example_cesbio.ipynb) | Predefined |
| [CESBIO (AI4QC)](https://zenodo.org/records/11120395) | [`cesbio_ai4qc.py`](datasets/cesbio_ai4qc.py) | [Notebook](examples/semantic_segmentation_example_cesbio_ai4qc.ipynb) | Predefined |
| Chactun | [`chactun.py`](datasets/chactun.py) | [Notebook](examples/semantic_segmentation_example_chactun.ipynb) | Predefined |
| [Chesapeake](https://lila.science/datasets/chesapeakelandcover) | [`chesapeake.py`](datasets/chesapeake.py) | [Notebook](examples/semantic_segmentation_example_chesapeake.ipynb) | [train](splits/chesapeake_train.csv) · [val](splits/chesapeake_val.csv) · [test](splits/chesapeake_test.csv) |
| [CITY-OSM](https://zenodo.org/records/1154821) | [`city_osm.py`](datasets/city_osm.py) | [Notebook](examples/semantic_segmentation_example_city_osm.ipynb) | [train](splits/city_osm_train.csv) · [val](splits/city_osm_val.csv) · [test](splits/city_osm_test.csv) |
| [Cloud-38](https://www.kaggle.com/datasets/sorour/38cloud-cloud-segmentation-in-satellite-images) | [`cloud38.py`](datasets/cloud38.py) | [Notebook](examples/semantic_segmentation_example_cloud38.ipynb) | Predefined |
| [Cloud-95](https://www.kaggle.com/datasets/sorour/95cloud-cloud-segmentation-on-satellite-images) | [`cloud95.py`](datasets/cloud95.py) | [Notebook](examples/semantic_segmentation_example_cloud95.ipynb) | Predefined |
| [CloudSEN12](https://www.scidb.cn/en/detail?dataSetId=f72d622ff4ea4fa18070456a98222b1a) | [`cloudsen12.py`](datasets/cloudsen12.py) | [Notebook](examples/semantic_segmentation_example_cloudsen12.ipynb) | Predefined |
| [DeepGlobe](https://www.kaggle.com/datasets/balraj98/deepglobe-land-cover-classification-dataset) | [`deepglobe.py`](datasets/deepglobe.py) | [Notebook](examples/semantic_segmentation_example_deepglobe.ipynb) | Predefined |
| [DeepGlobe Road Extraction](https://www.kaggle.com/datasets/balraj98/deepglobe-road-extraction-dataset) | [`deepglobe_road.py`](datasets/deepglobe_road.py) | [Notebook](examples/semantic_segmentation_example_deepglobe_road.ipynb) | Predefined |
| [DFC21-DSE](https://ieee-dataport.org/competitions/2021-ieee-grss-data-fusion-contest-track-dse) | [`dfc21_dse.py`](datasets/dfc21_dse.py) | [Notebook](examples/semantic_segmentation_example_dfc21_dse.ipynb) | Predefined |
| [DLR SkyScapes](https://www.dlr.de/en/eoc/about-us/remote-sensing-technology-institute/photogrammetry-and-image-analysis/public-datasets/dlr-skyscapes) | [`dlr_skyscapes.py`](datasets/dlr_skyscapes.py) | [Notebook](examples/semantic_segmentation_example_dlr_skyscapes.ipynb) | Predefined |
| [DLRSD](https://sites.google.com/view/zhouwx/dataset#h.p_hQS2jYeaFpV0) | [`dlrsd.py`](datasets/dlrsd.py) | [Notebook](examples/semantic_segmentation_example_dlrsd.ipynb) | [train](splits/dlrsd_train.csv) · [val](splits/dlrsd_val.csv) · [test](splits/dlrsd_test.csv) |
| [DSTL Satellite Imagery Feature Detection](https://www.kaggle.com/competitions/dstl-satellite-imagery-feature-detection/overview) | [`dstl.py`](datasets/dstl.py) | [Notebook](examples/semantic_segmentation_example_dstl.ipynb) | Predefined |
| [DynamicEarthNet](https://mediatum.ub.tum.de/1650201) | [`dynamicearthnet.py`](datasets/dynamicearthnet.py) | [Notebook](examples/semantic_segmentation_example_dynamicearthnet.ipynb) | Predefined |
| [EnviroAtlas](https://zenodo.org/records/6268150) | [`enviroatlas.py`](datasets/enviroatlas.py) | [Notebook](examples/semantic_segmentation_example_enviroatlas.ipynb) | Predefined |
| [EORSSD](https://github.com/rmcong/EORSSD-dataset/tree/master) | [`eorssd.py`](datasets/eorssd.py) | [Notebook](examples/semantic_segmentation_example_eorssd.ipynb) | Predefined |
| [ETCI 2021 Floods](https://www.kaggle.com/datasets/aninda/etci-2021-competition-on-flood-detection/data) | [`etci2021floods.py`](datasets/etci2021floods.py) | [Notebook](examples/semantic_segmentation_example_etci2021floods.ipynb) | Predefined |
| [Field Delineation France](https://drive.google.com/drive/folders/1GDL1pvlDCcsyEwafe7N41WtS7doFlju3) | [`field_delineation_france.py`](datasets/field_delineation_france.py) | [Notebook](examples/semantic_segmentation_example_field_delineation_france.ipynb) | [train](splits/field_delineation_france_train.csv) · [val](splits/field_delineation_france_val.csv) · [test](splits/field_delineation_france_test.csv) |
| [Five Billion Pixels](https://drive.google.com/drive/folders/1924VnO08Gqo3Nv7Y4KirgJ9kqqCup7f0) | [`five_billion_pixels.py`](datasets/five_billion_pixels.py) | [Notebook](examples/semantic_segmentation_example_five_billion_pixels.ipynb) | [train](splits/five_billion_pixels_train.csv) · [val](splits/five_billion_pixels_val.csv) · [test](splits/five_billion_pixels_test.csv) |
| [FLAIR2](https://ignf.github.io/FLAIR/#FLAIR2) | [`flair2.py`](datasets/flair2.py) | [Notebook](examples/semantic_segmentation_example_flair2.ipynb) | Predefined |
| [Flood Area](https://www.kaggle.com/datasets/faizalkarim/flood-area-segmentation) | [`flood_area.py`](datasets/flood_area.py) | [Notebook](examples/semantic_segmentation_example_flood_area.ipynb) | [train](splits/flood_area_train.csv) · [val](splits/flood_area_val.csv) · [test](splits/flood_area_test.csv) |
| [FloodNet](https://drive.google.com/drive/folders/1sZZMJkbqJNbHgebKvHzcXYZHJd6ss4tH) | [`floodnet.py`](datasets/floodnet.py) | [Notebook](examples/semantic_segmentation_example_floodnet.ipynb) | Predefined |
| [ForestNet](https://stanfordmlgroup.github.io/projects/forestnet/) | [`forestnet.py`](datasets/forestnet.py) | [Notebook](examples/semantic_segmentation_example_forestnet.ipynb) | Predefined |
| [GeoLifeCLEF 2021](https://www.kaggle.com/competitions/geolifeclef-2021/data) | [`geolifeclef2021.py`](datasets/geolifeclef2021.py) | [Notebook](examples/semantic_segmentation_example_geolifeclef2021.ipynb) | Predefined |
| [GeoNRW](https://ieee-dataport.org/open-access/geonrw) | [`geonrw.py`](datasets/geonrw.py) | [Notebook](examples/semantic_segmentation_example_geonrw.ipynb) | [train](splits/geonrw_train.csv) · [val](splits/geonrw_val.csv) · [test](splits/geonrw_test.csv) |
| [GID-15](https://captain-whu.github.io/GID15/) | [`gid15.py`](datasets/gid15.py) | [Notebook](examples/semantic_segmentation_example_gid15.ipynb) | Predefined |
| [GLH-Water](https://drive.google.com/drive/folders/1VACDh3aGx72hDdz7Taf7FyUrpZS31Ksx) | [`glh_water.py`](datasets/glh_water.py) | [Notebook](examples/semantic_segmentation_example_glh_water.ipynb) | [train](splits/glh_water_train.csv) · [val](splits/glh_water_val.csv) · [test](splits/glh_water_test.csv) |
| [GRSS DFC2022](https://ieee-dataport.org/competitions/data-fusion-contest-2022-dfc2022) | [`grss_dfc_2022.py`](datasets/grss_dfc_2022.py) | [Notebook](examples/semantic_segmentation_example_grss_dfc_2022.ipynb) | Predefined |
| [GSFC](https://drive.google.com/drive/folders/1HaXiyJxm4CzvC__zLt2VQd3Qvb1WC9nF?usp=sharing) | [`gsfc.py`](datasets/gsfc.py) | [Notebook](examples/semantic_segmentation_example_gsfc.ipynb) | Predefined |
| [GSFC (AI4QC)](https://zenodo.org/records/11120856) | [`gsfc_ai4qc.py`](datasets/gsfc_ai4qc.py) | [Notebook](examples/semantic_segmentation_example_gsfc_ai4qc.ipynb) | Predefined |
| [GTA-V-SID](https://github.com/jiupinjia/gtav-sattellite-imagery-dataset?tab=readme-ov-file) | [`gta_v_sid.py`](datasets/gta_v_sid.py) | [Notebook](examples/semantic_segmentation_example_gta_v_sid.ipynb) | Predefined |
| [HDMaps](https://www.dlr.de/en/eoc/about-us/remote-sensing-technology-institute/photogrammetry-and-image-analysis/public-datasets/hd-maps) | [`hd_maps.py`](datasets/hd_maps.py) | [Notebook](examples/semantic_segmentation_example_hd_maps.ipynb) | Predefined |
| [Hephaestus](https://www.dropbox.com/scl/fi/qx99z7tlx6994r7zn4fgh/Hephaestus_Classification.zip?rlkey=lnnmakeu50au0r50yjbjonivx&e=1&dl=0) | [`hephaestus_sem_seg.py`](datasets/hephaestus_sem_seg.py) | [Notebook](examples/semantic_segmentation_example_hephaestus_sem_seg.ipynb) | [train](splits/hephaestus_sem_seg_train.csv) · [val](splits/hephaestus_sem_seg_val.csv) · [test](splits/hephaestus_sem_seg_test.csv) |
| [HLSBurnScars](https://huggingface.co/datasets/ibm-nasa-geospatial/hls_burn_scars/tree/main) | [`hlsburnscars.py`](datasets/hlsburnscars.py) | [Notebook](examples/semantic_segmentation_example_hlsburnscars.ipynb) | Predefined |
| [Hollstein](https://drive.google.com/drive/folders/1nT-Jr_0Qmr9BIU8kgHNY1e-HL3cdH4KS?usp=sharing) | [`hollstein.py`](datasets/hollstein.py) | [Notebook](examples/semantic_segmentation_example_hollstein.ipynb) | Predefined |
| [Hollstein (AI4QC)](https://zenodo.org/records/12585409) | [`hollstein_ai4qc.py`](datasets/hollstein_ai4qc.py) | [Notebook](examples/semantic_segmentation_example_hollstein_ai4qc.ipynb) | Predefined |
| [HRC-WHU](https://drive.google.com/file/d/1qqikjaX7tkfOONsF5EtR4vl6J7sToA6p/view) | [`hrc_whu.py`](datasets/hrc_whu.py) | [Notebook](examples/semantic_segmentation_example_hrc_whu.ipynb) | Predefined |
| [Inria](https://project.inria.fr/aerialimagelabeling/) | [`inria.py`](datasets/inria.py) | [Notebook](examples/semantic_segmentation_example_inria.ipynb) | Predefined |
| [Kaggle Semantic Segmentation](https://www.kaggle.com/datasets/humansintheloop/semantic-segmentation-of-aerial-imagery) | [`kaggle_semantic_segmentation.py`](datasets/kaggle_semantic_segmentation.py) | [Notebook](examples/semantic_segmentation_example_kaggle_semantic_segmentation.ipynb) | segmentation: [test](splits/kaggle_semamtic_segmentation_test.csv)<br>segmentation: [train](splits/kaggle_semantic_segmentation_train.csv) · [val](splits/kaggle_semantic_segmentation_val.csv) |
| [KappaSet](https://drive.google.com/drive/folders/1H18RIitlVvhmlY63_lj_BrtYPwzN8tgr?usp=sharing) | [`kappaset.py`](datasets/kappaset.py) | [Notebook](examples/semantic_segmentation_example_kappaset.ipynb) | Predefined |
| [KappaSet (AI4QC)](https://zenodo.org/records/11121065) | [`kappaset_ai4qc.py`](datasets/kappaset_ai4qc.py) | [Notebook](examples/semantic_segmentation_example_kappaset_ai4qc.ipynb) | Predefined |
| [KappaZeta](https://zenodo.org/records/5095024#.Y-DWMnbMIuU) | [`kappazeta.py`](datasets/kappazeta.py) | [Notebook](examples/semantic_segmentation_example_kappazeta.ipynb) | Predefined |
| [L7Irish](https://huggingface.co/datasets/torchgeo/l7irish/tree/main) | [`l7irish.py`](datasets/l7irish.py) | [Notebook](examples/semantic_segmentation_example_l7irish.ipynb) | [train](splits/l7irish_train.csv) · [val](splits/l7irish_val.csv) · [test](splits/l7irish_test.csv) |
| [Landcover AI](https://landcover.ai.linuxpolska.com/) | [`landcover_ai.py`](datasets/landcover_ai.py) | [Notebook](examples/semantic_segmentation_example_landcover_ai.ipynb) | Predefined |
| [LoveDA](https://zenodo.org/records/5706578) | [`loveda.py`](datasets/loveda.py) | [Notebook](examples/semantic_segmentation_example_loveda.ipynb) | Predefined |
| [m-pv4ger-seg](https://zenodo.org/records/8277054) | [`m_pv4ger_seg.py`](datasets/m_pv4ger_seg.py) | [Notebook](examples/semantic_segmentation_example_m_pv4ger_seg.ipynb) | Predefined |
| [MADOS](https://zenodo.org/records/10664073) | [`mados.py`](datasets/mados.py) | [Notebook](examples/semantic_segmentation_example_mados.ipynb) | Predefined |
| [Massachusetts Buildings](https://www.cs.toronto.edu/~vmnih/data/) | [`massachusetts_buildings.py`](datasets/massachusetts_buildings.py) | [Notebook](examples/semantic_segmentation_example_massachusetts_buildings.ipynb) | Predefined |
| [Massachusetts Roads](https://www.cs.toronto.edu/~vmnih/data/) | [`massachusetts_roads.py`](datasets/massachusetts_roads.py) | [Notebook](examples/semantic_segmentation_example_massachusetts_roads.ipynb) | Predefined |
| [Mid-Air](https://midair.ulg.ac.be/index.html) | [`mid_air.py`](datasets/mid_air.py) | [Notebook](examples/semantic_segmentation_example_mid_air.ipynb) | [train](splits/mid_air_train.csv) · [val](splits/mid_air_val.csv) · [test](splits/mid_air_test.csv) |
| [MSLCC](https://www.dlr.de/de/eoc/ueber-uns/institut-fuer-methodik-der-fernerkundung/photogrammetrie-und-bildanalyse/offentliche-datensatze/mslcc) | [`mslcc.py`](datasets/mslcc.py) | [Notebook](examples/semantic_segmentation_example_mslcc.ipynb) | Predefined |
| [Open Sentinel Map](https://github.com/VisionSystemsInc/open-sentinel-map?tab=readme-ov-file) | [`osm.py`](datasets/osm.py) | [Notebook](examples/semantic_segmentation_example_osm.ipynb) | Predefined |
| [Open-Canopy](https://huggingface.co/datasets/AI4Forest/Open-Canopy) | [`open_canopy.py`](datasets/open_canopy.py) | [Notebook](examples/semantic_segmentation_example_open_canopy.ipynb) | Predefined |
| [ORSSD](https://github.com/rmcong/ORSSD-dataset/) | [`orssd.py`](datasets/orssd.py) | [Notebook](examples/semantic_segmentation_example_orssd.ipynb) | Predefined |
| [PixBox (AI4QC)](https://zenodo.org/records/11121168) | [`pixbox_ai4qc.py`](datasets/pixbox_ai4qc.py) | [Notebook](examples/semantic_segmentation_example_pixbox_ai4qc.ipynb) | Predefined |
| [Potsdam](https://www.isprs.org/education/benchmarks/UrbanSemLab/default.aspx) | [`potsdam.py`](datasets/potsdam.py) | [Notebook](examples/semantic_segmentation_example_potsdam.ipynb) | Predefined |
| [Qingdao UAV-borne HSI (QUH)](https://zenodo.org/records/8223066) | [`quh.py`](datasets/quh.py) | [Notebook](examples/semantic_segmentation_example_quh.ipynb) | Predefined |
| [RIT-18](https://github.com/rmkemker/RIT-18) | [`rit18.py`](datasets/rit18.py) | [Notebook](examples/semantic_segmentation_example_rit18.ipynb) | Predefined |
| [RoadNet](https://drive.google.com/file/d/1GDHy7uwgOswuCDC49OamlNkAxjaITPBI/view) | [`roadnet.py`](datasets/roadnet.py) | [Notebook](examples/semantic_segmentation_example_roadnet.ipynb) | [train](splits/roadnet_train.csv) · [val](splits/roadnet_val.csv) · [test](splits/roadnet_test.csv) |
| [Rwanda Field Boundary](https://source.coop/repositories/nasa/rwanda-field-boundary-competition/access) | [`rwanda_field_boundary.py`](datasets/rwanda_field_boundary.py) | [Notebook](examples/semantic_segmentation_example_rwanda_field_boundary.ipynb) | Predefined |
| [S1S2-Water](https://zenodo.org/records/11278238) | [`s1s2_water.py`](datasets/s1s2_water.py) | [Notebook](examples/semantic_segmentation_example_s1s2_water.ipynb) | [train](splits/s1s2water_train.csv) · [val](splits/s1s2water_val.csv) · [test](splits/s1s2water_test.csv) |
| [Satellite Buildings](https://www.kaggle.com/datasets/hyyyrwang/buildings-dataset/data) | [`satellite_buildings.py`](datasets/satellite_buildings.py) | [Notebook](examples/semantic_segmentation_example_satellite_buildings.ipynb) | [train](splits/satellite_buildings_train.csv) · [val](splits/satellite_buildings_val.csv) · [test](splits/satellite_buildings_test.csv) |
| [SatlasPretrain](https://github.com/allenai/satlas/blob/main/SatlasPretrain.md) | [`satlaspretrain_semantic_segmentation.py`](datasets/satlaspretrain_semantic_segmentation.py) | [Notebook](examples/semantic_segmentation_example_satlaspretrain_semantic_segmentation.ipynb) | Predefined |
| [SeasoNet](https://zenodo.org/records/6979994) | [`seasonet.py`](datasets/seasonet.py) | [Notebook](examples/semantic_segmentation_example_seasonet.ipynb) | Predefined |
| [SEN12MS](https://dataserv.ub.tum.de/s/m1474000) | [`sen12ms.py`](datasets/sen12ms.py) | [Notebook](examples/semantic_segmentation_example_sen12ms.ipynb) | [train](splits/sen12ms_train.csv) · [val](splits/sen12ms_val.csv) · [test](splits/sen12ms_test.csv) |
| [Sen1Floods11](https://github.com/cloudtostreet/Sen1Floods11) | [`sen1floods11.py`](datasets/sen1floods11.py) | [Notebook](examples/semantic_segmentation_example_sen1floods11.ipynb) | Predefined |
| [Sentinel-2 Cloud Cover Segmentation](https://source.coop/repositories/radiantearth/cloud-cover-detection-challenge/access) | [`sentinel2_cloud_cover.py`](datasets/sentinel2_cloud_cover.py) | [Notebook](examples/semantic_segmentation_example_sentinel2_cloud_cover.ipynb) | [train](splits/sentinel_2_cloud_cover_segmentation_train.csv) · [val](splits/sentinel_2_cloud_cover_segmentation_val.csv) · [test](splits/sentinel_2_cloud_cover_segmentation_test.csv) |
| [Sentinel-2 Cloud Mask Catalogue](https://zenodo.org/records/4172871#.YIvbA44zbIV) | [`s2cmc.py`](datasets/s2cmc.py) | [Notebook](examples/semantic_segmentation_example_s2cmc.ipynb) | Predefined |
| [SkyScapes](https://www.dlr.de/en/eoc/about-us/remote-sensing-technology-institute/photogrammetry-and-image-analysis/public-datasets/dlr-skyscapes) | [`skyscapes.py`](datasets/skyscapes.py) | [Notebook](examples/semantic_segmentation_example_skyscapes.ipynb) | Predefined |
| [SpaceNet 1](https://registry.opendata.aws/spacenet/) | [`spacenet_1.py`](datasets/spacenet_1.py) | [Notebook](examples/semantic_segmentation_example_spacenet_1.ipynb) | Predefined |
| [SpaceNet 2](https://www.kaggle.com/datasets/ugorjiir/spacenet-2-paris-buildings) | [`spacenet_2.py`](datasets/spacenet_2.py) | [Notebook](examples/semantic_segmentation_example_spacenet_2.ipynb) | Predefined |
| [SpaceNet 3](https://medium.com/the-downlinq/introducing-the-spacenet-road-detection-and-routing-challenge-and-dataset-7604de39b779) | [`spacenet_3.py`](datasets/spacenet_3.py) | [Notebook](examples/semantic_segmentation_example_spacenet_3.ipynb) | Predefined |
| [SpaceNet 4](https://registry.opendata.aws/spacenet/) | [`spacenet_4.py`](datasets/spacenet_4.py) | [Notebook](examples/semantic_segmentation_example_spacenet_4.ipynb) | Predefined |
| [SpaceNet 5](https://medium.com/the-downlinq/spacenet-5-dataset-release-80bd82d7c528) | [`spacenet_5.py`](datasets/spacenet_5.py) | [Notebook](examples/semantic_segmentation_example_spacenet_5.ipynb) | Predefined |
| SpaceNet 6 | [`spacenet6.py`](datasets/spacenet6.py) | [Notebook](examples/semantic_segmentation_example_spacenet6.ipynb) | Predefined |
| [SpaceNet 7](https://medium.com/the-downlinq/announcing-spacenet-7-the-multi-temporal-urban-development-challenge-53f68e24b9fa) | [`spacenet_7.py`](datasets/spacenet_7.py) | [Notebook](examples/semantic_segmentation_example_spacenet_7.ipynb) | Predefined |
| [SpaceNet 8](https://medium.com/@SpaceNet_Project/announcing-spacenet-8-flood-detection-challenge-using-multiclass-segmentation-49bac6ae300) | [`spacenet_8.py`](datasets/spacenet_8.py) | [Notebook](examples/semantic_segmentation_example_spacenet_8.ipynb) | Predefined |
| [SPARCS](https://www.usgs.gov/landsat-missions/spatial-procedures-automated-removal-cloud-and-shadow-sparcs-validation-data) | [`sparcs.py`](datasets/sparcs.py) | [Notebook](examples/semantic_segmentation_example_sparcs.ipynb) | [train](splits/sparcs_train.csv) · [val](splits/sparcs_val.csv) · [test](splits/sparcs_test.csv) |
| [SSL4EO-L-Benchmark](https://huggingface.co/datasets/torchgeo/ssl4eo_l_benchmark/tree/main) | [`ssl4eo.py`](datasets/ssl4eo.py) | [Notebook](examples/semantic_segmentation_example_ssl4eo.ipynb) | [train](splits/ssl4eo_train.csv) · [val](splits/ssl4eo_val.csv) · [test](splits/ssl4eo_test.csv) |
| [Synthinel](https://drive.google.com/drive/folders/1T2fO-VLfyQoQdy5C4at_uHkP0KBRZkit) | [`synthinel.py`](datasets/synthinel.py) | [Notebook](examples/semantic_segmentation_example_synthinel.ipynb) | [train](splits/synthinel_train.csv) · [val](splits/synthinel_val.csv) · [test](splits/synthinel_test.csv) |
| [Transmission Towers and Powers Lines (TTPLA)](https://github.com/R3ab/ttpla_dataset?tab=readme-ov-file) | [`ttpla.py`](datasets/ttpla.py) | [Notebook](examples/semantic_segmentation_example_ttpla.ipynb) | [train](splits/ttpla_train.csv) · [val](splits/ttpla_val.csv) · [test](splits/ttpla_test.csv) |
| [UAVid](https://uavid.nl/) | [`uavid.py`](datasets/uavid.py) | [Notebook](examples/semantic_segmentation_example_uavid.ipynb) | Predefined |
| [UBC](https://drive.google.com/file/d/1XnKFKqjoa95PLXFw01HcXx4Az49Qw37i/view) | [`ubc.py`](datasets/ubc.py) | [Notebook](examples/semantic_segmentation_example_ubc.ipynb) | Predefined |
| [UDD-5](https://drive.google.com/file/d/1BNL8HNFRiNjSzdcQJo-uXiejZJ6DgunY/view) | [`udd5.py`](datasets/udd5.py) | [Notebook](examples/semantic_segmentation_example_udd5.ipynb) | Predefined |
| [UDD-6](https://drive.google.com/file/d/1BNL8HNFRiNjSzdcQJo-uXiejZJ6DgunY/view) | [`udd6.py`](datasets/udd6.py) | [Notebook](examples/semantic_segmentation_example_udd6.ipynb) | Predefined |
| [Vaihingen](https://www.isprs.org/education/benchmarks/UrbanSemLab/default.aspx) | [`vaihingen.py`](datasets/vaihingen.py) | [Notebook](examples/semantic_segmentation_example_vaihingen.ipynb) | [train](splits/vaihingen_train.csv) · [val](splits/vaihingen_val.csv) · [test](splits/vaihingen_test.csv) |
| [VALID](https://drive.google.com/drive/folders/1nKQqR92NqNEWj-KXvxAKekFdQikrotLj) | [`valid_semantic_segmentation.py`](datasets/valid_semantic_segmentation.py) | [Notebook](examples/semantic_segmentation_example_valid_semantic_segmentation.ipynb) | [train](splits/valid_semantic_segmentation_train.csv) · [val](splits/valid_semantic_segmentation_val.csv) · [test](splits/valid_semantic_segmentation_test.csv) |
| [Water Bodies](https://www.kaggle.com/datasets/franciscoescobar/satellite-images-of-water-bodies) | [`water_bodies.py`](datasets/water_bodies.py) | [Notebook](examples/semantic_segmentation_example_water_bodies.ipynb) | [train](splits/water_bodies_train.csv) · [val](splits/water_bodies_val.csv) · [test](splits/water_bodies_test.csv) |
| [Weakly Supervised Cropland Segmentation](https://sustainlab-group.github.io/sustainbench/docs/datasets/sdg2/weakly_supervised_cropland.html) | [`wscs_semantic_segmentation.py`](datasets/wscs_semantic_segmentation.py) | [Notebook](examples/semantic_segmentation_example_wscs_semantic_segmentation.ipynb) | Predefined |
| [WHDLD](https://sites.google.com/view/zhouwx/dataset) | [`whdld.py`](datasets/whdld.py) | [Notebook](examples/semantic_segmentation_example_whdld.ipynb) | [train](splits/whdld_train.csv) · [val](splits/whdld_val.csv) · [test](splits/whdld_test.csv) |
| [WHU-OHS](http://irsip.whu.edu.cn/resources/WHU_OHS_show.php) | [`whu_ohs.py`](datasets/whu_ohs.py) | [Notebook](examples/semantic_segmentation_example_whu_ohs.ipynb) | [train](splits/whu_ohs_train.csv) · [val](splits/whu_ohs_val.csv) · [test](splits/whu_ohs_test.csv) |
| [WHU-OPT-SAR](https://drive.google.com/drive/folders/1W3iMpkehng7ADXmhz9pPvmgFQBayq22t) | [`whu_opt_sar.py`](datasets/whu_opt_sar.py) | [Notebook](examples/semantic_segmentation_example_whu_opt_sar.ipynb) | [train](splits/whu_opt_sar_train.csv) · [val](splits/whu_opt_sar_val.csv) · [test](splits/whu_opt_sar_test.csv) |
| [WHUAerial](http://gpcv.whu.edu.cn/data/building_dataset.html) | [`whuaerial.py`](datasets/whuaerial.py) | [Notebook](examples/semantic_segmentation_example_whuaerial.ipynb) | Predefined |
| [WHUMix](http://gpcv.whu.edu.cn/data/whu-mix(raster)/whu_mix%20(raster).html) | [`whumix.py`](datasets/whumix.py) | [Notebook](examples/semantic_segmentation_example_whumix.ipynb) | Predefined |
| [WHUS2CD+](https://github.com/Neooolee/WHUS2-CD) | [`whus2cdplus.py`](datasets/whus2cdplus.py) | [Notebook](examples/semantic_segmentation_example_whus2cdplus.ipynb) | Predefined |
| [Zurich Summer](https://zenodo.org/records/5914759) | [`zurich_summer.py`](datasets/zurich_summer.py) | [Notebook](examples/semantic_segmentation_example_zurich_summer.ipynb) | [train](splits/zurich_summer_train.csv) · [val](splits/zurich_summer_val.csv) · [test](splits/zurich_summer_test.csv) |

</details>

### Object detection

<details>
<summary>Show the list</summary>

| Dataset | Loader | Notebook | Splits |
| --- | --- | --- | --- |
| [Aerial Maritime Drone](https://public.roboflow.com/object-detection/aerial-maritime/9/download/voc) | [`aerial_maritime_drone.py`](datasets/aerial_maritime_drone.py) | [Notebook](examples/object_detection_example_aerial_maritime_drone.ipynb) | Predefined |
| [AFO](https://www.kaggle.com/datasets/jangsienicajzkowy/afo-aerial-dataset-of-floating-objects) | [`afo.py`](datasets/afo.py) | [Notebook](examples/object_detection_example_afo.ipynb) | Predefined |
| [Airbus Aircraft Detection](https://www.kaggle.com/datasets/airbusgeo/airbus-aircrafts-sample-dataset) | [`airbus_aircraft.py`](datasets/airbus_aircraft.py) | [Notebook](examples/object_detection_example_airbus_aircraft.ipynb) | [train](splits/airbus_aircraft_train.csv) · [val](splits/airbus_aircraft_val.csv) · [test](splits/airbus_aircraft_test.csv) |
| [Airbus Oil Storage Detection](https://www.kaggle.com/datasets/airbusgeo/airbus-oil-storage-detection-dataset) | [`airbus_oil_storage.py`](datasets/airbus_oil_storage.py) | [Notebook](examples/object_detection_example_airbus_oil_storage.ipynb) | [train](splits/airbus_oil_storage_train.csv) · [val](splits/airbus_oil_storage_val.csv) · [test](splits/airbus_oil_storage_test.csv) |
| [ArtifiVe-Potsdam](https://rs.ipb.uni-bonn.de/data/artifive-potsdam/index.html) | [`artifive_potsdam.py`](datasets/artifive_potsdam.py) | [Notebook](examples/object_detection_example_artifive_potsdam.ipynb) | [train](splits/artifive_potsdam_train.csv) · [val](splits/artifive_potsdam_val.csv) · [test](splits/artifive_potsdam_test.csv) |
| [AU-AIR](https://bozcani.github.io/auairdataset) | [`auair.py`](datasets/auair.py) | [Notebook](examples/object_detection_example_auair.ipynb) | [train](splits/auair_train.csv) · [val](splits/auair_val.csv) · [test](splits/auair_test.csv) |
| CARPK | [`carpk.py`](datasets/carpk.py) | [Notebook](examples/object_detection_example_carpk.ipynb) | Predefined |
| [CGI Planes](https://www.kaggle.com/datasets/aceofspades914/cgi-planes-in-satellite-imagery-w-bboxes) | [`cgi_planes.py`](datasets/cgi_planes.py) | [Notebook](examples/object_detection_example_cgi_planes.ipynb) | Predefined |
| [COWC](https://gdo152.llnl.gov/cowc/download/) | [`cowc_object_detection.py`](datasets/cowc_object_detection.py) | [Notebook](examples/object_detection_example_cowc_object_detection.ipynb) | Predefined |
| [COWC-M](https://gdo152.llnl.gov/cowc/download/) | [`cowc_m_object_detection.py`](datasets/cowc_m_object_detection.py) | [Notebook](examples/object_detection_example_cowc_m_object_detection.ipynb) | [train](splits/cowc_m_object_detection_train.csv) · [val](splits/cowc_m_object_detection_val.csv) · [test](splits/cowc_m_object_detection_test.csv) |
| [DFC23](https://ieee-dataport.org/competitions/2023-ieee-grss-data-fusion-contest-large-scale-fine-grained-building-classification) | [`dfc23.py`](datasets/dfc23.py) | [Notebook](examples/object_detection_example_dfc23.ipynb) | Predefined |
| [DLR-MVDA](https://www.dlr.de/en/eoc/about-us/remote-sensing-technology-institute/photogrammetry-and-image-analysis/public-datasets/dlr-mvda) | [`dlr_mvda.py`](datasets/dlr_mvda.py) | [Notebook](examples/object_detection_example_dlr_mvda.ipynb) | Predefined |
| [EAGLE](https://www.dlr.de/en/eoc/about-us/remote-sensing-technology-institute/photogrammetry-and-image-analysis/public-datasets/eagle) | [`eagle.py`](datasets/eagle.py) | [Notebook](examples/object_detection_example_eagle.ipynb) | Predefined |
| [FAIR1M](https://www.gaofen-challenge.com/benchmark) | [`fair1m.py`](datasets/fair1m.py) | [Notebook](examples/object_detection_example_fair1m.ipynb) | [train](splits/fair1m_train.csv) · [val](splits/fair1m_val.csv) · [test](splits/fair1m_test.csv) |
| FMOW | [`fmow.py`](datasets/fmow.py) | [Notebook](examples/object_detection_example_fmow.ipynb) | Predefined |
| [HRPlanesv2](https://zenodo.org/records/7331974) | [`hrplanesv2.py`](datasets/hrplanesv2.py) | [Notebook](examples/object_detection_example_hrplanesv2.ipynb) | Predefined |
| HRRSD | Generic: [`ObjectDetectionPascalDataset`](https://github.com/biasvariancelabs/aitlas/blob/v2.0.2/aitlas/datasets/object_detection.py) | [Notebook](examples/object_detection_example_hrrsd.ipynb) | Predefined |
| [HRSID](https://github.com/chaozhong2010/HRSID) | [`hrsid.py`](datasets/hrsid.py) | [Notebook](examples/object_detection_example_hrsid.ipynb) | [train](splits/hrsid_train.csv) · [val](splits/hrsid_val.csv) · [test](splits/hrsid_test.csv) |
| [IDTReeS](https://zenodo.org/records/3934932) | [`idtrees.py`](datasets/idtrees.py) | [Notebook](examples/object_detection_example_idtrees.ipynb) | Predefined |
| [ISBDA](https://github.com/zgzxy001/MSNET?tab=readme-ov-file) | [`isbda.py`](datasets/isbda.py) | [Notebook](examples/object_detection_example_isbda.ipynb) | Predefined |
| [ITCVD](https://hyper.ai/cn/datasets/5419) | [`itcvd.py`](datasets/itcvd.py) | [Notebook](examples/object_detection_example_itcvd.ipynb) | Predefined |
| [Larch Casebearer - Tree Classification](https://lila.science/datasets/forest-damages-larch-casebearer/) | [`larch_casebearer_tree_classification.py`](datasets/larch_casebearer_tree_classification.py) | [Notebook](examples/object_detection_example_larch_casebearer_tree_classification.ipynb) | [train](splits/larch_casebearer_train.csv) · [val](splits/larch_casebearer_val.csv) · [test](splits/larch_casebearer_test.csv) |
| [Larch Casebearer - Tree Damage](https://lila.science/datasets/forest-damages-larch-casebearer/) | [`larch_casebearer_tree_damage.py`](datasets/larch_casebearer_tree_damage.py) | [Notebook](examples/object_detection_example_larch_casebearer_tree_damage.ipynb) | [train](splits/larch_casebearer_train.csv) · [val](splits/larch_casebearer_val.csv) · [test](splits/larch_casebearer_test.csv) |
| LEVIR | [`levir.py`](datasets/levir.py) | [Notebook](examples/object_detection_example_levir.ipynb) | [train](splits/levir_train.csv) · [val](splits/levir_val.csv) · [test](splits/levir_test.csv) |
| [MAR20](https://gcheng-nwpu.github.io/) | [`mar20.py`](datasets/mar20.py) | [Notebook](examples/object_detection_example_mar20.ipynb) | Predefined |
| [Marine Debris](https://source.coop/repositories/nasa/marine-debris) | [`marine_debris.py`](datasets/marine_debris.py) | [Notebook](examples/object_detection_example_marine_debris.ipynb) | [train](splits/marine_debris_train.csv) · [val](splits/marine_debris_val.csv) · [test](splits/marine_debris_test.csv) |
| [MASATI](https://docs.google.com/forms/d/e/1FAIpQLSe1qeo_XH2v_obix_huLVVqWpGL_1DAZlMTVlEeMuy-EHdH9w/viewform?c=0&w=1) | [`masati_object_detection.py`](datasets/masati_object_detection.py) | [Notebook](examples/object_detection_example_masati_object_detection.ipynb) | [train](splits/masati_object_detection_train.csv) · [val](splits/masati_object_detection_val.csv) · [test](splits/masati_object_detection_test.csv) |
| [NeonTree](https://zenodo.org/records/5914554) | [`neontree.py`](datasets/neontree.py) | [Notebook](examples/object_detection_example_neontree.ipynb) | Predefined |
| [New Zealand Cattle Detection](https://zenodo.org/records/5908869) | [`nz_cattle.py`](datasets/nz_cattle.py) | [Notebook](examples/object_detection_example_nz_cattle.ipynb) | [train](splits/nz_cattle_train.csv) · [val](splits/nz_cattle_val.csv) · [test](splits/nz_cattle_test.csv) |
| [Oil and Gas Storage Tank (OGST)](https://data.mendeley.com/datasets/bkxj8z84m9/3) | [`ogst.py`](datasets/ogst.py) | [Notebook](examples/object_detection_example_ogst.ipynb) | [train](splits/ogst_train.csv) · [val](splits/ogst_val.csv) · [test](splits/ogst_test.csv) |
| [OIRDS](https://sourceforge.net/projects/oirds/) | [`oirds.py`](datasets/oirds.py) | [Notebook](examples/object_detection_example_oirds.ipynb) | [train](splits/oirds_train.csv) · [val](splits/oirds_val.csv) · [test](splits/oirds_test.csv) |
| [Parallax (AI4QC)](https://zenodo.org/records/13903820) | [`parallax_ai4qc.py`](datasets/parallax_ai4qc.py) | [Notebook](examples/object_detection_example_parallax_ai4qc_rotated_bboxes.ipynb) | Predefined |
| [RarePlanes](https://www.iqt.org/library/the-rareplanes-dataset) | [`rare_planes.py`](datasets/rare_planes.py) | [Notebook](examples/object_detection_example_rare_planes.ipynb) | [train](splits/rare_planes_train.csv) · [val](splits/rare_planes_val.csv) · [test](splits/rare_planes_test.csv) |
| [RFI (AI4QC)](https://zenodo.org/records/13757181) | [`rfi_ai4qc.py`](datasets/rfi_ai4qc.py) | [Notebook](examples/object_detection_example_rfi_ai4qc_coco.ipynb) | Predefined |
| [SAR Ship Detection](https://drive.google.com/drive/folders/1wcyV2nzcibh3EoaTuJ67HoS6SqakeGB9?usp=drive_link) | [`sar_ship_detection.py`](datasets/sar_ship_detection.py) | [Notebook](examples/object_detection_example_sar_ship_detection.ipynb) | [train](splits/sar_ship_detection_train.csv) · [val](splits/sar_ship_detection_val.csv) · [test](splits/sar_ship_detection_test.csv) |
| [SeaDronesSee](https://seadronessee.cs.uni-tuebingen.de/home) | [`seadronessee.py`](datasets/seadronessee.py) | [Notebook](examples/object_detection_example_seadronessee.ipynb) | Predefined |
| [Search And Rescue](https://github.com/michaelthoreau/SearchAndRescueNet) | [`search_and_rescue.py`](datasets/search_and_rescue.py) | [Notebook](examples/object_detection_example_search_and_rescue.ipynb) | Predefined |
| [Ships in Google Earth](https://www.kaggle.com/datasets/tomluther/ships-in-google-earth) | [`ships_in_google_earth.py`](datasets/ships_in_google_earth.py) | [Notebook](examples/object_detection_example_ships_in_google_earth.ipynb) | Predefined |
| [SIMD](https://github.com/ihians/simd) | [`simd.py`](datasets/simd.py) | [Notebook](examples/object_detection_example_simd.ipynb) | Predefined |
| [SSDD](https://github.com/TianwenZhang0825/Official-SSDD?tab=readme-ov-file) | [`ssdd.py`](datasets/ssdd.py) | [Notebook](examples/object_detection_example_ssdd.ipynb) | Predefined |
| [SWIM](https://www.kaggle.com/datasets/lilitopia/swimship-wake-imagery-mass) | [`swim.py`](datasets/swim.py) | [Notebook](examples/object_detection_example_swim.ipynb) | Predefined |
| [Swimming Pool And Car Detection](https://www.kaggle.com/datasets/kbhartiya83/swimming-pool-and-car-detection/data) | [`swimming_pool_and_car_detection.py`](datasets/swimming_pool_and_car_detection.py) | [Notebook](examples/object_detection_example_swimming_pool_and_car_detection.ipynb) | Predefined |
| [Things and Stuff](https://ai.stanford.edu/~gaheitz/Research/TAS/) | [`things_and_stuff.py`](datasets/things_and_stuff.py) | [Notebook](examples/object_detection_example_things_and_stuff.ipynb) | Predefined |
| [UAVOD-10](https://drive.google.com/file/d/1emLAe7002_syWNxsTO0MgVg4knokFVlQ/view) | [`uavod10.py`](datasets/uavod10.py) | [Notebook](examples/object_detection_example_uavod10.ipynb) | [train](splits/uavod10_train.csv) · [val](splits/uavod10_val.csv) · [test](splits/uavod10_test.csv) |
| [UCAS-AOD](https://hyper.ai/cn/datasets/5419) | [`ucas_aod.py`](datasets/ucas_aod.py) | [Notebook](examples/object_detection_example_ucas_aod.ipynb) | [train](splits/ucas_aod_train.csv) · [val](splits/ucas_aod_val.csv) · [test](splits/ucas_aod_test.csv) |
| [VALID](https://drive.google.com/drive/folders/1nKQqR92NqNEWj-KXvxAKekFdQikrotLj) | [`valid_object_detection.py`](datasets/valid_object_detection.py) | [Notebook](examples/object_detection_example_valid_object_detection.ipynb) | [train](splits/valid_object_detection_train.csv) · [val](splits/valid_object_detection_val.csv) · [test](splits/valid_object_detection_test.csv) |
| [Vehicle Detection in Aerial Imagery (VEDAI)](https://downloads.greyc.fr/vedai/) | [`vedai.py`](datasets/vedai.py) | [Notebook](examples/object_detection_example_vedai.ipynb) | [train](splits/vedai_train.csv) · [val](splits/vedai_val.csv) · [test](splits/vedai_test.csv) |
| [VHR-10](https://drive.google.com/file/d/1--foZ3dV5OCsqXQXT84UeKtrAqc5CkAE/view) | [`vhr10.py`](datasets/vhr10.py) | [Notebook](examples/object_detection_example_vhr10.ipynb) | [train](splits/vhr10_train.csv) · [val](splits/vhr10_val.csv) · [test](splits/vhr10_test.csv) |
| VisDrone | [`visdrone.py`](datasets/visdrone.py) | [Notebook](examples/object_detection_example_visdrone.ipynb) | Predefined |
| [xView](https://challenge.xviewdataset.org/challenge-description) | [`xview.py`](datasets/xview.py) | [Notebook](examples/object_detection_example_xview.ipynb) | Predefined |

</details>

### Change detection

<details>
<summary>Show the list</summary>

| Dataset | Loader | Notebook | Splits |
| --- | --- | --- | --- |
| [3DCD](https://sites.google.com/uniroma1.it/3dchangedetection/home-page) | [`cd3d.py`](datasets/cd3d.py) | [Notebook](examples/change_detection_example_cd3d.ipynb) | Predefined |
| [AICD](https://www.kaggle.com/datasets/kmader/aerial-change-detection-in-video-games) | [`aicd.py`](datasets/aicd.py) | [Notebook](examples/change_detection_example_aicd.ipynb) | [train](splits/aicd_train.csv) · [val](splits/aicd_val.csv) · [test](splits/aicd_test.csv) |
| [Bandon](https://github.com/fitzpchao/BANDON) | [`bandon.py`](datasets/bandon.py) | [Notebook](examples/change_detection_example_bandon.ipynb) | [train](splits/bandon_train.csv) · [val](splits/bandon_val.csv) · [test](splits/bandon_test.csv) |
| [California Flood](https://sites.google.com/view/luppino/data) | [`california_flood.py`](datasets/california_flood.py) | [Notebook](examples/change_detection_example_california_flood.ipynb) | Predefined |
| [ChaBuD](https://huggingface.co/datasets/blanchon/ChaBuD) | [`chabud.py`](datasets/chabud.py) | [Notebook](examples/change_detection_example_chabud.ipynb) | Predefined |
| [CLCD](https://github.com/liumency/CropLand-CD) | [`clcd.py`](datasets/clcd.py) | [Notebook](examples/change_detection_example_clcd.ipynb) | [train](splits/clcd_train.csv) · [val](splits/clcd_val.csv) · [test](splits/clcd_test.csv) |
| [CNAM-CD](https://github.com/Silvestezhou/CNAM-CD) | [`cnam_cd.py`](datasets/cnam_cd.py) | [Notebook](examples/change_detection_example_cnam_cd.ipynb) | [train](splits/cnam_cd_train.csv) · [val](splits/cnam_cd_val.csv) · [test](splits/cnam_cd_test.csv) |
| [DSIFN](https://github.com/GeoZcx/A-deeply-supervised-image-fusion-network-for-change-detection-in-remote-sensing-images/tree/master/dataset) | [`dsifn.py`](datasets/dsifn.py) | [Notebook](examples/change_detection_example_dsifn.ipynb) | [train](splits/dsifn_train.csv) · [val](splits/dsifn_val.csv) · [test](splits/dsifn_test.csv) |
| [EGY-BCD](https://github.com/oshholail/EGY-BCD) | [`egy_bcd.py`](datasets/egy_bcd.py) | [Notebook](examples/change_detection_example_egy_bcd.ipynb) | [train](splits/egy_bcd_train.csv) · [val](splits/egy_bcd_val.csv) · [test](splits/egy_bcd_test.csv) |
| [GVLM](https://github.com/zxk688/GVLM) | [`gvlm.py`](datasets/gvlm.py) | [Notebook](examples/change_detection_example_gvlm.ipynb) | Predefined |
| [Hi-UCD](https://github.com/Daisy-7/Hi-UCD-S) | [`hi_ucd.py`](datasets/hi_ucd.py) | [Notebook](examples/change_detection_example_hi_ucd.ipynb) | Predefined |
| [HRSCD](https://ieee-dataport.org/open-access/hrscd-high-resolution-semantic-change-detection-dataset) | [`hrscd.py`](datasets/hrscd.py) | [Notebook](examples/change_detection_example_hrscd.ipynb) | Predefined |
| [Hyperspectral Change Detection](https://gitlab.citius.usc.es/hiperespectral/ChangeDetectionDataset) | [`hyperspectral_change_detection.py`](datasets/hyperspectral_change_detection.py) | [Notebook](examples/change_detection_example_hyperspectral_change_detection.ipynb) | Predefined |
| [LEVIR-CD](https://justchenhao.github.io/LEVIR/) | [`levir_cd.py`](datasets/levir_cd.py) | [Notebook](examples/change_detection_example_levir_cd.ipynb) | Predefined |
| [LEVIR-CD+](https://drive.google.com/drive/folders/1lJ5GteQy5Z2LVayrKYZOn7_QPImlYP8C) | [`levir_cdplus.py`](datasets/levir_cdplus.py) | [Notebook](examples/change_detection_example_levir_cdplus.ipynb) | [train](splits/levir_cdplus_train.csv) · [val](splits/levir_cdplus_val.csv) · [test](splits/levir_cdplus_test.csv) |
| [MSBC](https://github.com/Lihy256/MSCDUnet) | [`msbc.py`](datasets/msbc.py) | [Notebook](examples/change_detection_example_msbc.ipynb) | [train](splits/msbc_train.csv) · [val](splits/msbc_val.csv) · [test](splits/msbc_test.csv) |
| [MSOSCD](https://github.com/Lihy256/MSCDUnet) | [`msoscd.py`](datasets/msoscd.py) | [Notebook](examples/change_detection_example_msoscd.ipynb) | [train](splits/msoscd_train.csv) · [val](splits/msoscd_val.csv) · [test](splits/msoscd_test.csv) |
| [MUMUCD](https://zenodo.org/records/10674011) | [`mumucd.py`](datasets/mumucd.py) | [Notebook](examples/change_detection_example_mumucd.ipynb) | Predefined |
| [OMBRIA](https://github.com/geodrak/OMBRIA) | [`ombria.py`](datasets/ombria.py) | [Notebook](examples/change_detection_example_ombria.ipynb) | [train](splits/ombria_train.csv) · [val](splits/ombria_val.csv) · [test](splits/ombria_test.csv) |
| [OSCD](https://ieee-dataport.org/open-access/oscd-onera-satellite-change-detection) | [`oscd.py`](datasets/oscd.py) | [Notebook](examples/change_detection_example_oscd.ipynb) | Predefined |
| [QuakeSet](https://huggingface.co/datasets/DarthReca/quakeset) | [`quakeset.py`](datasets/quakeset.py) | [Notebook](examples/change_detection_example_quakeset.ipynb) | Predefined |
| [S2Looking](https://github.com/S2Looking/Dataset) | [`s2looking.py`](datasets/s2looking.py) | [Notebook](examples/change_detection_example_s2looking.ipynb) | Predefined |
| [Season-Varying CDD](https://drive.google.com/file/d/1GX656JqqOyBi_Ef0w65kDGVto-nHrNs9) | [`season_varying_cdd.py`](datasets/season_varying_cdd.py) | [Notebook](examples/change_detection_example_season_varying_cdd.ipynb) | [train](splits/season_varying_cdd_train.csv) · [val](splits/season_varying_cdd_val.csv) · [test](splits/season_varying_cdd_test.csv) |
| [SECOND](https://captain-whu.github.io/SCD/) | [`second.py`](datasets/second.py) | [Notebook](examples/change_detection_example_second.ipynb) | Predefined |
| [SMARS](https://www.dlr.de/de/eoc/ueber-uns/institut-fuer-methodik-der-fernerkundung/photogrammetrie-und-bildanalyse/offentliche-datensatze/smars) | [`smars.py`](datasets/smars.py) | [Notebook](examples/change_detection_example_smars.ipynb) | Predefined |
| [SYSU-CD](https://github.com/liumency/SYSU-CD) | [`sysu_cd.py`](datasets/sysu_cd.py) | [Notebook](examples/change_detection_example_sysu_cd.ipynb) | [train](splits/sysu_cd_train.csv) · [val](splits/sysu_cd_val.csv) · [test](splits/sysu_cd_test.csv) |
| [SZTAKI AirChange](http://web.eee.sztaki.hu/remotesensing/airchange_benchmark.html) | [`sztaki_airchange.py`](datasets/sztaki_airchange.py) | [Notebook](examples/change_detection_example_sztaki_airchange.ipynb) | Predefined |
| [xView2](https://xview2.org/dataset) | [`xview2.py`](datasets/xview2.py) | [Notebook](examples/change_detection_example_xview2.ipynb) | Predefined |
| [ZY3](https://drive.google.com/file/d/1yoVP5xc4dPA2sDwYIgVkMEjwCz7q9AdC/view) | [`zy3.py`](datasets/zy3.py) | [Notebook](examples/change_detection_example_zy3.ipynb) | Predefined |

</details>

### Time series

Crop types and other tasks on satellite image time series.

<details>
<summary>Show the list</summary>

| Dataset | Loader | Notebook | Splits |
| --- | --- | --- | --- |
| [BioMassters](https://nascetti-a.github.io/BioMasster/) | [`biomassters.py`](datasets/biomassters.py) | [Notebook](examples/multitemporal_semantic_segmentation_example_biomassters.ipynb) | Predefined |
| [BreizhCrops](https://github.com/dl4sits/BreizhCrops) | [`breizhcrops.py`](datasets/breizhcrops.py) | [Notebook](examples/breizhcrops_evaluation_aitlas.ipynb) | Predefined |
| [Cashew Plantation](https://staging.source.coop/technoserve/cashews-benin) | [`cashew_plantation.py`](datasets/cashew_plantation.py) | [Notebook](examples/multitemporal_semantic_segmentation_example_cashew_plantation.ipynb) | Predefined |
| [Crop Harvest](https://zenodo.org/records/10251170) | [`cropharvest.py`](datasets/cropharvest.py) | [Notebook](examples/multitemporal_semantic_segmentation_example_cropharvest.ipynb) | Predefined |
| [Crop Type Mapping (Ghana and South Sudan)](https://drive.google.com/drive/folders/1WhVObtFOzYFiXBsbbrEGy1DUtv7ov7wF) | [`crop_type_mapping_ghana_south_sudan.py`](datasets/crop_type_mapping_ghana_south_sudan.py) | [Notebook](examples/multitemporal_semantic_segmentation_example_crop_type_mapping_ghana_south_sudan.ipynb) | Predefined |
| [Crop Type Mapping in Kenya](https://drive.google.com/drive/folders/1Rq1F-ys-rkjftAUnbdGJ1T6udsZWnEyo) | [`crop_type_kenya.py`](datasets/crop_type_kenya.py) | [Notebook](examples/multitemporal_image_classification_example_crop_type_kenya.ipynb) | Predefined |
| [CV4A Kenya Crop Type](https://source.coop/radiantearth/african-crops-kenya-02) | [`cv4a_kenya_crop_type.py`](datasets/cv4a_kenya_crop_type.py) | [Notebook](examples/multitemporal_semantic_segmentation_example_cv4a_kenya_crop_type.ipynb) | Predefined |
| EOPatch Crops | [`eopatch_crops.py`](datasets/eopatch_crops.py) | [Notebook](examples/crop_type_prediction_example_netherlands.ipynb) | Predefined |
| [MTS12](https://gpcv.whu.edu.cn/data/dataset12/dataset12.html) | [`mts12.py`](datasets/mts12.py) | [Notebook](examples/multitemporal_semantic_segmentation_example_mts12.ipynb) | Predefined |
| [Out-of-Domain Land Cover](https://drive.google.com/drive/folders/138EeHCXxYJZ_OdNqgcn4olY2-CCxEnVa) | [`ood_land_cover.py`](datasets/ood_land_cover.py) | [Notebook](examples/multitemporal_image_classification_example_ood_land_cover.ipynb) | Predefined |
| [PASTIS](https://github.com/VSainteuf/pastis-benchmark) | [`pastis.py`](datasets/pastis.py) | [Notebook](examples/multitemporal_semantic_segmentation_example_pastis.ipynb) | Predefined |
| [PASTIS HD](https://huggingface.co/datasets/IGNF/PASTIS-HD) | [`pastis_hd.py`](datasets/pastis_hd.py) | [Notebook](examples/multitemporal_semantic_segmentation_example_pastis_hd.ipynb) | [train](splits/pastis_hd_train.csv) · [val](splits/pastis_hd_val.csv) · [test](splits/pastis_hd_test.csv) |
| [SA Crop Type](https://source.coop/repositories/radiantearth/south-africa-crops-competition/access) | [`sa_crop_type.py`](datasets/sa_crop_type.py) | [Notebook](examples/multitemporal_semantic_segmentation_example_sa_crop_type.ipynb) | Predefined |
| [Sen4AgriNet](https://github.com/Orion-AI-Lab/S4A) | [`sen4agrinet.py`](datasets/sen4agrinet.py) | [Notebook](examples/multitemporal_semantic_segmentation_example_sen4agrinet.ipynb) | Predefined |
| [TreeSatAI-TS](https://huggingface.co/datasets/IGNF/TreeSatAI-Time-Series) | [`treesatai_ts.py`](datasets/treesatai_ts.py) | [Notebook](examples/multitemporal_image_classification_example_treesatai_ts.ipynb) | Predefined |

</details>

### Anomaly detection

Unsupervised detection of anomalies in Sentinel-1 and Sentinel-2 products (AI4QC).

<details>
<summary>Show the list</summary>

| Dataset | Loader | Notebook | Splits |
| --- | --- | --- | --- |
| [S1 New Anomalies](https://zenodo.org/records/13789234) | [`s1anomalies_ai4qc.py`](datasets/s1anomalies_ai4qc.py) | [Notebook](examples/unsupervised_example_s1_new_anomalies_ai4qc.ipynb) | Predefined |
| [S2 New Anomalies](https://zenodo.org/records/13790109) | [`s2anomalies_ai4qc.py`](datasets/s2anomalies_ai4qc.py) | [Notebook](examples/unsupervised_example_s2_new_anomalies_ai4qc.ipynb) | Predefined |

</details>

### Other notebooks

**Foundation models.** Each of the 16 foundation models in AiTLAS has a notebook that extracts feature embeddings: [AnySat](examples/feature_embedding_example_anysat_sen12ms_faireo.ipynb), [CACo](examples/feature_embedding_example_caco_fmow_faireo.ipynb), [Copernicus-FM](examples/feature_embedding_example_copernicusfm_sacroptype_faireo.ipynb), [CROMA](examples/feature_embedding_example_croma_sacroptype_faireo.ipynb), [DOFA](examples/feature_embedding_example_dofa_sen12ms_faireo.ipynb), [Galileo](examples/feature_embedding_example_galileo_sacroptype_faireo.ipynb), [GASSL](examples/feature_embedding_example_gassl_fmow_faireo.ipynb), [Panopticon](examples/feature_embedding_example_panopticon_sacroptype_faireo.ipynb), [Presto](examples/feature_embedding_example_presto_sacroptype_faireo.ipynb), [Prithvi](examples/feature_embedding_example_prithvi_sacroptype_faireo.ipynb), [SatMAE](examples/feature_embedding_example_satmae_fmow_faireo.ipynb), [SatMAE++](examples/feature_embedding_example_satmae_plusplus_fmow_faireo.ipynb), [Scale-MAE](examples/feature_embedding_example_scalemae_fmow_faireo.ipynb), [SeCo](examples/feature_embedding_example_seco_fmow_faireo.ipynb), [TerraFM](examples/feature_embedding_example_terrafm_sacroptype_faireo.ipynb) and [TerraMind](examples/feature_embedding_example_terramind_sacroptype_faireo.ipynb). TerraMind also has notebooks for [semantic segmentation](examples/semantic_segmentation_example_terramind_dlrsd.ipynb) and [feature extraction](examples/feature_extraction_example_terramind_dlrsd.ipynb) on DLRSD, and for [image generation](examples/image_generation_example_five_billion_pixels_terramind_faireo.ipynb) and [Thinking in Modalities](examples/thinking_in_modalities_example_five_billion_pixels_terramind_faireo.ipynb) on Five Billion Pixels. Two more train foundation models on a task: [Galileo on BrazilDAM](examples/multiclass_classification_example_galileo_brazildam.ipynb) and [Copernicus-FM on DLRSD](examples/multilabel_classification_example_copernicusfm_dlrsd_multilabel.ipynb).

**Configuration library.** Ready-made `CompositeModel` configurations with foundation model backbones, for multi-class classification ([1](examples/config_library_faireo_config_mcc_1.ipynb), [2](examples/config_library_faireo_config_mcc_2.ipynb)), multi-label classification ([1](examples/config_library_faireo_config_mlc_1.ipynb), [2](examples/config_library_faireo_config_mlc_2.ipynb)) and segmentation ([1](examples/config_library_faireo_config_seg_1.ipynb), [2](examples/config_library_faireo_config_seg_2.ipynb)).

**More.** Statistics of the [multi-class](examples/multiclass_datasets_statistics.ipynb), [multi-label](examples/multilabel_datasets_statistics.ipynb) and [segmentation](examples/semantic_segmentation_datasets_statistics.ipynb) datasets; [Grad-CAM explanations](examples/grad_cam_classification_example.ipynb) for classification; a [land use classification benchmark](examples/land_use_classification_benchmark.ipynb); a [data analysis of BreizhCrops and EOPatch files](examples/breizhcrops_eopatch_data_analysis.ipynb); and other formats of two AI4QC datasets: RFI in [Pascal VOC](examples/object_detection_example_rfi_ai4qc_pascal.ipynb) and [YOLO](examples/object_detection_example_rfi_ai4qc_yolo.ipynb) format, and Parallax with [straight boxes](examples/object_detection_example_parallax_ai4qc_straight_bboxes.ipynb).

## Data splits

Where the original authors did not define train, validation and test splits, we made our own. They are in [`splits/`](splits), one CSV file per split, named `<dataset>_train.csv`, `<dataset>_val.csv` and `<dataset>_test.csv`. Each file lists the samples in its split, in the format that the dataset's loader reads.

## Related resources

- [AiTLAS](https://github.com/biasvariancelabs/aitlas): the toolbox these loaders build on, with more than 60 model architectures and 16 geospatial foundation models.
- [FAIR-EO Hub](https://eodata.bvlabs.ai/ai4eo/#/data): metadata for every dataset, including its license and provenance.
- [FAIR-EO change detection benchmark](https://github.com/biasvariancelabs/FAIR-EO-CD-benchmark): ten change detection models compared on ten of these datasets, with fixed splits and metrics.
- [AiTLAS Benchmark Arena](https://github.com/biasvariancelabs/aitlas-arena): more than 500 trained image classification models on 22 datasets.

## Citation

If you use FAIR-EO in your research, please cite AiTLAS, as well as the papers of the datasets you use:

```bibtex
@article{aitlas2023,
AUTHOR = {Dimitrovski, Ivica and Kitanovski, Ivan and Panov, Panče and Kostovska, Ana and Simidjievski, Nikola and Kocev, Dragi},
TITLE = {AiTLAS: Artificial Intelligence Toolbox for Earth Observation},
JOURNAL = {Remote Sensing},
VOLUME = {15},
YEAR = {2023},
NUMBER = {9},
ARTICLE-NUMBER = {2343},
ISSN = {2072-4292},
DOI = {10.3390/rs15092343}
}
```

## License

The code in this repository is released under the [MIT License](LICENSE). The datasets are not part of this repository: each keeps the license set by its authors, which you can find in the [FAIR-EO Hub](https://eodata.bvlabs.ai/ai4eo/#/data).
