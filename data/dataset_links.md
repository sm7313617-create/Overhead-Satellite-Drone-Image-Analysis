# Dataset Links

## SpaceNet-1

Official:
https://spacenet.ai/spacenet-buildings-dataset-v1/

AWS:
s3://spacenet-dataset/spacenet/SN1_buildings/tarballs/

Downloaded:
- SN1_buildings_train_AOI_1_Rio_3band.tar.gz
- SN1_buildings_train_AOI_1_Rio_8band.tar.gz
- SN1_buildings_train_AOI_1_Rio_geojson_buildings.tar.gz
- SN1_buildings_train_AOI_1_Rio_metadata.tar.gz
- SN1_buildings_test_AOI_1_Rio_3band.tar.gz
- SN1_buildings_test_AOI_1_Rio_8band.tar.gz

Storage:
Dataset stored externally in Google Drive.

## Svamitva Drone Aerial Images

Official:
https://www.kaggle.com/datasets/utkarshsaxenadn/svamitva-drone-aerial-images

About Dataset:
This special dataset was curated for the Smart India Hackathon 2024 and the Indus Hackathon 2025, which included problem statements on extracting features from drone aerial images. The data, provided by the Smart India Hackathon, initially came as ECW files and were converted to TIF files using QGIS and GDAL. These images were then split into patches of size 1024 by 1024 pixels and manually annotated using polygons in Label Studio.

Originally, 482 images were labeled, and data augmentation increased the dataset to around 1300 files. The original images are from patch 1 to patch 482, while the rest are augmented versions.

The dataset includes various versions:

Filtered data: Contains only images with buildings.
Binary version: For building footprint extraction using binary segmentation tasks.
Original SIH dataset: Includes the TIF images supplied by the Smart India Hackathon 2024.
Additionally, script files used to create custom datasets are available. These scripts help with tasks such as data generation, data augmentation, and converting annotations.

Two models are provided: one that takes 1024 by 1024 pixel inputs and another that takes 512 by 512 pixel inputs. These models are available at different training epochs (e.g., 50, 100, 150, 200 epochs). While the models show satisfactory results, improvements can still be made. The annotations are not perfect but are sufficient for training a good model. We are continuously working on improving the annotations.

Thank you for using the dataset. It is recommended to review the data structure for better understanding and usage.

