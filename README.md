# Mangrove Classification Pipeline

Multi-classifier land cover classification of mangrove ecosystems in Greater Accra, Ghana using Sentinel-2 and Landsat 9 feature stacks.

Compares Random Forest, SVM-RBF, and XGBoost on four sensor/processing combinations (L9 SR, L9 TOA, S2 SR, S2 TOA) with polygon-based stratified splitting, post-processing, and full experiment tracking via Weights & Biases.

## Architecture

```
┌──────────────┐     ┌──────────────┐     ┌──────────────────┐
│  GEE Export  │────▶│  Feature     │────▶│   Classification │
│  (JS script) │     │  Stacks      │     │   (RF/SVM/XGB)   │
└──────────────┘     │  .tif files  │     └────────┬─────────┘
                     └──────────────┘              │
┌──────────────┐                          ┌────────▼─────────┐
│  Training    │─────────────────────────▶│  Post-processing │
│  Polygons    │  polygon-based split     │  small patch     │
│  (.shp)      │  per-polygon sampling    │  removal + focal │
└──────────────┘                          │  mode smoothing  │
                                          └────────┬─────────┘
                     ┌──────────────┐              │
                     │  W&B         │◀─────────────┤
                     │  Dashboard   │     ┌────────▼─────────┐
                     └──────────────┘     │  Outputs         │
                                          │  • GeoTIFFs      │
                                          │  • JSON reports  │
                                          │  • Importance    │
                                          │    plots         │
                                          └──────────────────┘
```

## Quick Start

```bash
# Clone
git clone https://github.com/reginaldotoo/mangrove-classification.git
cd mangrove-classification

# Install
pip install -r requirements.txt

# Run all datasets and classifiers
python Script.py --config config.yaml

# Run a single dataset
python Script.py --config config.yaml --run L9_SR

# Run without W&B tracking
python Script.py --config config.yaml --no-wandb

# Docker
docker build -t mangrove-classification .
docker run -v /path/to/data:/data mangrove-classification \
    python Script.py --config config.yaml
```

## Configuration

All parameters are in `config.yaml` — no hardcoded paths or magic numbers in the code. Hyperparameters, file paths, post-processing settings, and W&B config are all in one place.

## Experiment Tracking

Training runs log to [Weights & Biases](https://wandb.ai):

- Overall accuracy and kappa per classifier per dataset
- Per-class F1, precision, recall
- Feature importance plots
- Full config for reproducibility

```bash
wandb login
python Script.py --config config.yaml
```


## Classes

| ID | Class | Color |
|----|-------|-------|
| 1  | Built-Up / Bareland | Grey |
| 2  | Other Vegetation | Green |
| 3  | Water | Blue |
| 4  | Mangroves | Red |


| File | Description |
|------|-------------|
| `Script.py` | Full classification pipeline |
| `config.yaml` | All parameters and file paths |
| `Export_All_Feature_Stacks_2024.js` | Google Earth Engine export script |
| `Training.zip` | Training polygon shapefiles |
| `Dockerfile` | Containerized execution |
| `.github/workflows/` | CI for build verification |


