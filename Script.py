"""
Mangrove Land Cover Classification Pipeline

Multi-classifier (Random Forest, SVM-RBF, XGBoost) land cover classification
using Sentinel-2 and Landsat 9 feature stacks, with polygon-based stratified
splitting, post-processing, and accuracy assessment.

Usage:
    python Script.py --config config.yaml
    python Script.py --config config.yaml --run L9_SR
    python Script.py --config config.yaml --no-wandb
"""

import argparse
import json
import logging
import time
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
import yaml
from rasterio.features import rasterize
from scipy.ndimage import label, generic_filter, binary_dilation
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    cohen_kappa_score,
    f1_score,
    jaccard_score,
)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False

# ── Logging ──────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────────────────────

BAND_NAMES_L9_SR = [
    'SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B6', 'SR_B7',
    'NDVI', 'EVI', 'NDWI', 'NDBI', 'MVI', 'SAVI', 'LSWI',
    'contrast', 'homogeneity', 'entropy',
]

BAND_NAMES_L9_TOA = [
    'B2', 'B3', 'B4', 'B5', 'B6', 'B7',
    'NDVI', 'EVI', 'NDWI', 'NDBI', 'MVI', 'SAVI', 'LSWI',
    'contrast', 'homogeneity', 'entropy',
]

BAND_NAMES_S2 = [
    'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B11', 'B12',
    'NDVI', 'EVI', 'NDWI', 'NDBI', 'MVI', 'SAVI', 'LSWI',
    'contrast', 'homogeneity', 'entropy',
]

CLASS_NAMES = {1: 'BuiltUp Bareland', 2: 'Other Vegetation', 3: 'Water', 4: 'Mangroves'}
CLASS_COLORS = {1: '#8b8b8b', 2: '#00ff27', 3: '#1700ff', 4: '#ff0101'}
CLASS_LABELS = sorted(CLASS_NAMES.keys())
NODATA = 255

BAND_NAMES = None


# ── Config ───────────────────────────────────────────────────────────────────

def load_config(path):
    """Load YAML config file."""
    with open(path) as f:
        cfg = yaml.safe_load(f)
    logger.info(f"Config loaded from {path}")
    return cfg


# ── Classifiers ──────────────────────────────────────────────────────────────

def get_classifiers(cfg):
    """Return dict of classifier name -> (instance, needs_scaling) from config."""
    seed = cfg['classifiers']['seed']
    n_est = cfg['classifiers']['n_estimators']
    rf_cfg = cfg['classifiers']['rf']
    svm_cfg = cfg['classifiers']['svm']
    xgb_cfg = cfg['classifiers']['xgb']

    classifiers = {
        'RF': (
            RandomForestClassifier(
                n_estimators=n_est, class_weight=rf_cfg['class_weight'],
                random_state=seed, n_jobs=rf_cfg['n_jobs'],
            ),
            False,
        ),
        'SVM': (
            SVC(
                kernel=svm_cfg['kernel'], C=svm_cfg['C'], gamma=svm_cfg['gamma'],
                class_weight=svm_cfg['class_weight'], random_state=seed,
                cache_size=svm_cfg['cache_size'],
            ),
            True,
        ),
    }

    if HAS_XGBOOST:
        classifiers['XGB'] = (
            XGBClassifier(
                n_estimators=n_est, max_depth=xgb_cfg['max_depth'],
                learning_rate=xgb_cfg['learning_rate'],
                subsample=xgb_cfg['subsample'],
                colsample_bytree=xgb_cfg['colsample_bytree'],
                eval_metric='mlogloss', random_state=seed, n_jobs=-1,
            ),
            False,
        )
    else:
        logger.warning("xgboost not installed — using sklearn GradientBoosting")
        classifiers['XGB'] = (
            GradientBoostingClassifier(
                n_estimators=n_est, max_depth=xgb_cfg['max_depth'],
                learning_rate=xgb_cfg['learning_rate'],
                subsample=xgb_cfg['subsample'], random_state=seed,
            ),
            False,
        )

    return classifiers


# ── Data Loading ─────────────────────────────────────────────────────────────

def load_feature_stack(path):
    """Load feature stack and auto-detect sensor from band count."""
    global BAND_NAMES
    logger.info(f"Loading feature stack: {path}")

    with rasterio.open(path) as src:
        data = src.read()
        profile = src.profile.copy()
        transform = src.transform
        crs = src.crs

    n_bands = data.shape[0]
    if n_bands == 20:
        BAND_NAMES = BAND_NAMES_S2
        logger.info(f"Detected Sentinel-2 stack ({n_bands} bands)")
    elif n_bands == 16:
        BAND_NAMES = BAND_NAMES_L9_SR if 'SR' in str(path).upper() else BAND_NAMES_L9_TOA
        logger.info(f"Detected Landsat 9 stack ({n_bands} bands)")
    else:
        BAND_NAMES = [f'Band_{i+1}' for i in range(n_bands)]
        logger.warning(f"Unexpected band count ({n_bands}) — using generic names")

    logger.info(f"Shape: {data.shape}")
    return data, profile, transform, crs


def load_training_polygons(shp_path, raster_crs):
    """Load training polygons and reproject to match raster CRS."""
    logger.info(f"Loading training polygons: {shp_path}")
    gdf = gpd.read_file(shp_path)

    if 'Class' not in gdf.columns:
        raise ValueError("Shapefile must have a 'Class' field with values 1-4.")

    gdf['Class'] = gdf['Class'].astype(int)
    invalid = gdf[~gdf['Class'].isin(CLASS_LABELS)]
    if len(invalid) > 0:
        logger.warning(f"{len(invalid)} polygons with Class outside 1-4, skipping")
        gdf = gdf[gdf['Class'].isin(CLASS_LABELS)].copy()

    if gdf.crs != raster_crs:
        logger.info(f"Reprojecting polygons from {gdf.crs} to {raster_crs}")
        gdf = gdf.to_crs(raster_crs)

    gdf['polygon_id'] = range(len(gdf))
    logger.info(f"Total polygons: {len(gdf)}")
    for cls_val, cls_name in CLASS_NAMES.items():
        logger.info(f"  Class {cls_val} ({cls_name}): {(gdf['Class'] == cls_val).sum()}")

    return gdf


# ── Pixel Extraction ─────────────────────────────────────────────────────────

def extract_samples(data, gdf, transform, cfg):
    """Extract pixel samples from polygons, capping per-polygon contribution."""
    max_per_poly = cfg['data']['max_pixels_per_polygon']
    seed = cfg['classifiers']['seed']
    logger.info(f"Extracting samples (max {max_per_poly} per polygon)")

    rows, cols = data.shape[1], data.shape[2]
    rng = np.random.RandomState(seed)

    class_raster = rasterize(
        [(geom, cls) for geom, cls in zip(gdf.geometry, gdf['Class'])],
        out_shape=(rows, cols), transform=transform, fill=0, dtype='uint8',
    )
    pid_raster = rasterize(
        [(geom, pid + 1) for geom, pid in zip(gdf.geometry, gdf['polygon_id'])],
        out_shape=(rows, cols), transform=transform, fill=0, dtype='int32',
    )

    labelled_mask = class_raster > 0
    pixel_locs = np.where(labelled_mask)
    all_y = class_raster[pixel_locs]
    all_groups = pid_raster[pixel_locs] - 1
    all_X = data[:, pixel_locs[0], pixel_locs[1]].T

    valid_mask = np.all(np.isfinite(all_X), axis=1)
    all_X, all_y, all_groups = all_X[valid_mask], all_y[valid_mask], all_groups[valid_mask]

    keep_indices, trimmed = [], 0
    for pid in np.unique(all_groups):
        pid_idx = np.where(all_groups == pid)[0]
        if len(pid_idx) <= max_per_poly:
            keep_indices.append(pid_idx)
        else:
            keep_indices.append(rng.choice(pid_idx, max_per_poly, replace=False))
            trimmed += 1

    keep_indices = np.concatenate(keep_indices)
    X, y, groups = all_X[keep_indices], all_y[keep_indices], all_groups[keep_indices]

    logger.info(f"Samples: {len(y)} ({trimmed} polygons trimmed)")
    return X, y, groups


# ── Train / Validate Split ───────────────────────────────────────────────────

def split_by_polygon(X, y, groups, cfg):
    """Stratified polygon-level train/validation split."""
    test_size = cfg['data']['test_size']
    seed = cfg['classifiers']['seed']
    rng = np.random.RandomState(seed)

    train_mask = np.zeros(len(y), dtype=bool)
    val_mask = np.zeros(len(y), dtype=bool)

    for cls_val in CLASS_LABELS:
        cls_mask = y == cls_val
        cls_pids = np.unique(groups[cls_mask])
        rng.shuffle(cls_pids)
        n_val = max(1, int(len(cls_pids) * test_size))
        val_pids = set(cls_pids[:n_val].tolist())

        cls_indices = np.where(cls_mask)[0]
        is_val = np.isin(groups[cls_indices], list(val_pids))
        val_mask[cls_indices[is_val]] = True
        train_mask[cls_indices[~is_val]] = True

    X_train, X_val = X[train_mask], X[val_mask]
    y_train, y_val = y[train_mask], y[val_mask]
    logger.info(f"Split: {len(y_train)} train / {len(y_val)} val")
    return X_train, X_val, y_train, y_val


# ── Classification ───────────────────────────────────────────────────────────

def train_classifier(clf, X_train, y_train, clf_name, needs_scaling):
    """Train classifier with optional feature scaling."""
    scaler = None
    X_fit = X_train

    if needs_scaling:
        scaler = StandardScaler()
        X_fit = scaler.fit_transform(X_train)

    logger.info(f"Training {clf_name}...")
    t0 = time.time()
    if clf_name == 'XGB' and HAS_XGBOOST:
        clf.fit(X_fit, y_train - 1)
    else:
        clf.fit(X_fit, y_train)

    logger.info(f"{clf_name} trained in {time.time() - t0:.1f}s")
    return clf, scaler


def classify_raster(data, clf, clf_name, scaler=None):
    """Apply classifier to full raster."""
    logger.info(f"Classifying raster with {clf_name}")
    n_bands, rows, cols = data.shape

    flat = data.reshape(n_bands, -1).T
    valid_mask = np.all(np.isfinite(flat), axis=1)
    logger.info(f"Valid pixels: {valid_mask.sum():,} / {len(valid_mask):,}")

    X_predict = flat[valid_mask]
    if scaler is not None:
        X_predict = scaler.transform(X_predict)

    result = np.full(len(valid_mask), NODATA, dtype=np.uint8)
    if valid_mask.sum() > 0:
        t0 = time.time()
        preds = clf.predict(X_predict)
        if clf_name == 'XGB' and HAS_XGBOOST:
            preds = preds + 1
        result[valid_mask] = preds.astype(np.uint8)
        logger.info(f"Prediction done in {time.time() - t0:.1f}s")

    return result.reshape(rows, cols)


# ── Post-processing ──────────────────────────────────────────────────────────

def postprocess_map(classified, cfg):
    """Remove small patches and apply focal mode smoothing."""
    min_conn = cfg['postprocessing']['min_connected']
    focal = cfg['postprocessing']['focal_size']
    logger.info(f"Post-processing (min_connected={min_conn}, focal={focal})")

    result = classified.copy()
    original_nodata = classified == NODATA
    result = reassign_small_patches(result, original_nodata, min_conn, focal)
    result = apply_focal_mode(result, original_nodata, focal)
    return result


def reassign_small_patches(classified, original_nodata, min_size, focal_size):
    """Reassign small connected components to majority neighbor class."""
    result = classified.copy()
    reassigned = 0

    for cls in [c for c in np.unique(result) if c != NODATA]:
        labeled_arr, n_comp = label(result == cls)
        for comp_id in range(1, n_comp + 1):
            comp_mask = labeled_arr == comp_id
            if comp_mask.sum() >= min_size:
                continue

            dilated = binary_dilation(comp_mask, iterations=focal_size)
            neighbor_mask = dilated & ~comp_mask & ~original_nodata
            if neighbor_mask.sum() > 0:
                nvals = result[neighbor_mask]
                nvals = nvals[(nvals != NODATA) & (nvals >= 1) & (nvals <= 4)]
                if len(nvals) > 0:
                    counts = np.bincount(nvals.astype(int), minlength=5)
                    result[comp_mask] = np.argmax(counts[1:]) + 1
                    reassigned += comp_mask.sum()
                    continue

            result[comp_mask] = NODATA
            reassigned += comp_mask.sum()

    logger.info(f"Reassigned {reassigned} pixels from small patches")
    return result


def apply_focal_mode(classified, original_nodata, size):
    """Apply focal mode filter to smooth class boundaries."""
    def mode_filter(values):
        valid = values[(values != NODATA) & (values >= 1) & (values <= 4)]
        if len(valid) == 0:
            return NODATA
        return np.argmax(np.bincount(valid.astype(int), minlength=5)[1:]) + 1

    result = generic_filter(
        classified.astype(float), mode_filter, size=size,
        mode='constant', cval=NODATA,
    ).astype(np.uint8)
    result[original_nodata] = NODATA
    return result


# ── Accuracy ─────────────────────────────────────────────────────────────────

def assess_accuracy(clf, X_val, y_val, clf_name, scaler=None):
    """Compute classification metrics."""
    X_eval = scaler.transform(X_val) if scaler else X_val
    y_pred = clf.predict(X_eval)
    if clf_name == 'XGB' and HAS_XGBOOST:
        y_pred = y_pred + 1

    cm = confusion_matrix(y_val, y_pred, labels=CLASS_LABELS)
    report = classification_report(
        y_val, y_pred, labels=CLASS_LABELS,
        target_names=[CLASS_NAMES[i] for i in CLASS_LABELS],
        output_dict=True, zero_division=0,
    )
    kappa = cohen_kappa_score(y_val, y_pred)
    f1_cls = f1_score(y_val, y_pred, labels=CLASS_LABELS, average=None, zero_division=0)
    iou_cls = jaccard_score(y_val, y_pred, labels=CLASS_LABELS, average=None, zero_division=0)
    oa = np.trace(cm) / cm.sum()

    logger.info(f"  OA={oa:.4f}  Kappa={kappa:.4f}")
    for idx, cls_val in enumerate(CLASS_LABELS):
        cn = CLASS_NAMES[cls_val]
        logger.info(f"  {cn:<20} F1={f1_cls[idx]:.4f}  IoU={iou_cls[idx]:.4f}")

    results = {
        'classifier': clf_name, 'overall_accuracy': float(oa),
        'kappa': float(kappa), 'confusion_matrix': cm.tolist(), 'per_class': {},
    }
    for idx, cls_val in enumerate(CLASS_LABELS):
        cn = CLASS_NAMES[cls_val]
        results['per_class'][cn] = {
            'f1': float(f1_cls[idx]), 'iou': float(iou_cls[idx]),
            'precision': float(report[cn]['precision']),
            'recall': float(report[cn]['recall']),
        }
    return results


# ── Feature Importance ───────────────────────────────────────────────────────

def report_feature_importance(clf, clf_name, X_val, y_val, scaler, output_dir, seed):
    """Compute and plot feature importance."""
    if hasattr(clf, 'feature_importances_'):
        importances = clf.feature_importances_
    else:
        X_eval = scaler.transform(X_val) if scaler else X_val
        y_eval = y_val - 1 if (clf_name == 'XGB' and HAS_XGBOOST) else y_val
        perm = permutation_importance(clf, X_eval, y_eval, n_repeats=10,
                                      random_state=seed, n_jobs=-1)
        importances = perm.importances_mean

    # Plot
    sorted_idx = np.argsort(importances)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(range(len(BAND_NAMES)), importances[sorted_idx], color='#2196F3')
    ax.set_yticks(range(len(BAND_NAMES)))
    ax.set_yticklabels([BAND_NAMES[i] for i in sorted_idx])
    ax.set_xlabel('Importance')
    ax.set_title(f'{clf_name} Feature Importance')
    plt.tight_layout()
    plot_path = Path(output_dir) / f'feature_importance_{clf_name}.png'
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)

    return {BAND_NAMES[i]: float(importances[i]) for i in range(len(BAND_NAMES))}


# ── Area Calculation ─────────────────────────────────────────────────────────

def calculate_areas(classified, transform, crs):
    """Calculate area per class in km²."""
    if crs and crs.is_geographic:
        center_lat = transform[5] + (classified.shape[0] // 2) * transform[4]
        pw = abs(transform[0]) * 111320.0 * np.cos(np.radians(abs(center_lat)))
        ph = abs(transform[4]) * 111320.0
        pixel_area_km2 = (pw * ph) / 1e6
    else:
        pixel_area_km2 = abs(transform[0] * transform[4]) / 1e6

    areas = {}
    for cls_val, cls_name in CLASS_NAMES.items():
        n = (classified == cls_val).sum()
        areas[cls_name] = float(n * pixel_area_km2)
    return areas


# ── Output ───────────────────────────────────────────────────────────────────

def save_raster(array, profile, path):
    """Save a 2D array as single-band GeoTIFF."""
    out = profile.copy()
    out.update(dtype='uint8', count=1, nodata=NODATA, compress='lzw')
    with rasterio.open(path, 'w', **out) as dst:
        dst.write(array.astype(np.uint8), 1)
    logger.info(f"Saved: {path}")


# ── W&B ──────────────────────────────────────────────────────────────────────

def init_wandb(cfg, run_name):
    """Initialize W&B run if enabled and available."""
    wcfg = cfg.get('wandb', {})
    if not wcfg.get('enabled', False) or not HAS_WANDB:
        if wcfg.get('enabled') and not HAS_WANDB:
            logger.warning("wandb not installed — running without tracking")
        return None

    run = wandb.init(
        project=wcfg['project'], entity=wcfg.get('entity'),
        name=run_name, tags=wcfg.get('tags', []),
        config={'classifiers': cfg['classifiers'], 'data': cfg['data']},
    )
    logger.info(f"W&B run: {run.url}")
    return run


def log_to_wandb(run, clf_name, run_name, results, output_dir):
    """Log metrics and plots to W&B."""
    if not run:
        return

    prefix = f"{run_name}/{clf_name}"
    wandb.log({
        f"{prefix}/overall_accuracy": results['overall_accuracy'],
        f"{prefix}/kappa": results['kappa'],
    })
    for cls_name, m in results['per_class'].items():
        wandb.log({
            f"{prefix}/{cls_name}/f1": m['f1'],
            f"{prefix}/{cls_name}/precision": m['precision'],
            f"{prefix}/{cls_name}/recall": m['recall'],
        })

    plot_path = Path(output_dir) / f'feature_importance_{clf_name}.png'
    if plot_path.exists():
        wandb.log({f"{prefix}/feature_importance": wandb.Image(str(plot_path))})


# ── Main ─────────────────────────────────────────────────────────────────────

def run_classification(cfg, run_info, wandb_run=None):
    """Full pipeline for one feature stack, all classifiers."""
    run_name = run_info['name']
    output_dir = Path(run_info['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    seed = cfg['classifiers']['seed']

    logger.info(f"{'=' * 50}")
    logger.info(f"RUNNING: {run_name}")
    logger.info(f"{'=' * 50}")

    data, profile, transform, crs = load_feature_stack(run_info['feature_stack'])
    gdf = load_training_polygons(cfg['data']['training_polygons'], crs)
    X, y, groups = extract_samples(data, gdf, transform, cfg)
    X_train, X_val, y_train, y_val = split_by_polygon(X, y, groups, cfg)

    classifiers = get_classifiers(cfg)
    run_results = {}

    for clf_name, (clf, needs_scaling) in classifiers.items():
        logger.info(f"--- {clf_name} ---")

        clf, scaler = train_classifier(clf, X_train, y_train, clf_name, needs_scaling)
        classified = classify_raster(data, clf, clf_name, scaler)
        postprocessed = postprocess_map(classified, cfg)

        save_raster(classified, profile, output_dir / f'classified_{clf_name}.tif')
        save_raster(postprocessed, profile, output_dir / f'classified_postprocessed_{clf_name}.tif')

        acc = assess_accuracy(clf, X_val, y_val, clf_name, scaler)
        acc['feature_importance'] = report_feature_importance(
            clf, clf_name, X_val, y_val, scaler, output_dir, seed,
        )
        acc['areas_km2'] = calculate_areas(classified, transform, crs)
        acc['areas_km2_postprocessed'] = calculate_areas(postprocessed, transform, crs)

        with open(output_dir / f'accuracy_report_{clf_name}.json', 'w') as f:
            json.dump(acc, f, indent=2)

        log_to_wandb(wandb_run, clf_name, run_name, acc, output_dir)
        run_results[clf_name] = acc

    return run_results


def main():
    parser = argparse.ArgumentParser(description="Mangrove Classification Pipeline")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument("--run", default=None, help="Run specific dataset only")
    parser.add_argument("--no-wandb", action="store_true", help="Disable W&B")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.no_wandb:
        cfg.setdefault('wandb', {})['enabled'] = False

    wandb_run = init_wandb(cfg, run_name=args.run or "all_runs")

    runs = cfg['runs']
    if args.run:
        runs = [r for r in runs if r['name'] == args.run]
        if not runs:
            logger.error(f"Run '{args.run}' not found in config")
            return

    all_results = {}
    for run_info in runs:
        all_results[run_info['name']] = run_classification(cfg, run_info, wandb_run)

    # Summary
    logger.info("=" * 50)
    logger.info("SUMMARY")
    for run_name, run_results in all_results.items():
        for clf_name, r in run_results.items():
            logger.info(f"  {run_name:<10} {clf_name:<5} OA={r['overall_accuracy']:.4f} K={r['kappa']:.4f}")

    if wandb_run:
        wandb_run.finish()
    logger.info("DONE")


if __name__ == '__main__':
    main()
