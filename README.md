# Mangrove Classification Pipeline

Multi-classifier (Random Forest, SVM-RBF, XGBoost) land cover classification using Sentinel-2 and Landsat 9 feature stacks.

## Data

### Training Data
Training polygons are included in this repo under `Training/`. The shapefile contains four classes:
1. BuiltUp / Bareland
2. Other Vegetation
3. Water
4. Mangroves

### Feature Stacks
Feature stacks are generated from Google Earth Engine. To create them:
1. Open [Google Earth Engine Code Editor](https://code.earthengine.google.com/)
2. Create a new script and paste the contents of `Export_All_Feature_Stacks_2024.js`
3. Define your `ROI` geometry in the code editor (draw a polygon over your study area)
4. Run the script — this queues 4 export tasks to Google Drive:
   - `S2_SR_Feature_Stack_2024.tif` (Sentinel-2 Surface Reflectance, 22 bands)
   - `S2_TOA_Feature_Stack_2024.tif` (Sentinel-2 TOA, 22 bands)
   - `L9_SR_Feature_Stack_2024.tif` (Landsat 9 Surface Reflectance, 18 bands)
   - `L9_TOA_Feature_Stack_2024.tif` (Landsat 9 TOA, 18 bands)
5. In the Tasks tab, click Run on each export
6. Download the exported TIFs from Google Drive

### Folder Structure
Organize your local data folder like this before running:

```
my_data/
├── Training/
│   ├── Training 2024.shp
│   ├── Training 2024.shx
│   ├── Training 2024.dbf
│   ├── Training 2024.prj
│   └── Training 2024.cpg
└── Images/
    ├── L9_SR_Feature_Stack_2024.tif
    ├── L9_TOA_Feature_Stack_2024.tif
    ├── S2_SR_Feature_Stack_2024.tif
    └── S2_TOA_Feature_Stack_2024.tif
```

## Running with Docker

### Build
```bash
docker build -t mg-classification .
```

### Run
```bash
docker run --rm -v /path/to/my_data:/data mg-classification
```

Replace `/path/to/my_data` with the actual path to your data folder.

This runs all 4 feature stacks × 3 classifiers = 12 classification runs. Outputs are written to `/path/to/my_data/output/`.

### Outputs (per feature stack)
```
output/L9_SR_2024/
├── classified_RF.tif
├── classified_SVM.tif
├── classified_XGB.tif
├── classified_postprocessed_RF.tif
├── classified_postprocessed_SVM.tif
├── classified_postprocessed_XGB.tif
├── feature_importance_RF.png
├── feature_importance_SVM.png
├── feature_importance_XGB.png
├── accuracy_report_RF.json
├── accuracy_report_SVM.json
└── accuracy_report_XGB.json
```

## Running without Docker
```bash
pip install numpy geopandas rasterio scipy scikit-learn matplotlib xgboost
```
Update the file paths in `Script.py` to point to your local data, then:
```bash
python Script.py
```

## Note
SVM prediction on full rasters is significantly slower than RF or XGBoost. Expect minutes to hours per raster depending on size.
