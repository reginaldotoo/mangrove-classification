import json
import sys
import time
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
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

# Optional: XGBoost (falls back to sklearn GradientBoosting if unavailable)
try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    print("WARNING: xgboost not installed — falling back to sklearn GradientBoostingClassifier.")

#  CONFIGURATION
# Band names must match GEE export order exactly

BAND_NAMES_L9_SR = [
    'SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B6', 'SR_B7',
    'NDVI', 'EVI', 'NDWI', 'NDBI', 'MVI', 'SAVI', 'LSWI',
    'contrast', 'homogeneity', 'entropy',
]  # 16 bands

BAND_NAMES_L9_TOA = [
    'B2', 'B3', 'B4', 'B5', 'B6', 'B7',
    'NDVI', 'EVI', 'NDWI', 'NDBI', 'MVI', 'SAVI', 'LSWI',
    'contrast', 'homogeneity', 'entropy',
]  # 16 bands

BAND_NAMES_S2 = [
    'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B11', 'B12',
    'NDVI', 'EVI', 'NDWI', 'NDBI', 'MVI', 'SAVI', 'LSWI',
    'contrast', 'homogeneity', 'entropy',
]  # 20 bands

CLASS_NAMES = {
    1: 'BuiltUp Bareland',
    2: 'Other Vegetation',
    3: 'Water',
    4: 'Mangroves',
}

CLASS_COLORS = {
    1: '#8b8b8b',
    2: '#00ff27',
    3: '#1700ff',
    4: '#ff0101',
}

CLASS_LABELS = sorted(CLASS_NAMES.keys())  # [1, 2, 3, 4]
NODATA = 255

#  FILE PATHS (mapped via Docker: -v /your/local/data:/data)

TRAINING_POLYGONS = '/data/Training/Training 2024.shp'

RUNS = [
    {
        'name': 'L9_SR',
        'feature_stack': '/data/Images/L9_SR_Feature_Stack_2024.tif',
        'output_dir': '/data/output/L9_SR_2024',
    },
    {
        'name': 'L9_TOA',
        'feature_stack': '/data/Images/L9_TOA_Feature_Stack_2024.tif',
        'output_dir': '/data/output/L9_TOA_2024',
    },
    {
        'name': 'S2_SR',
        'feature_stack': '/data/Images/S2_SR_Feature_Stack_2024.tif',
        'output_dir': '/data/output/S2_SR_2024',
    },
    {
        'name': 'S2_TOA',
        'feature_stack': '/data/Images/S2_TOA_Feature_Stack_2024.tif',
        'output_dir': '/data/output/S2_TOA_2024',
    },
]

# Auto-detected at runtime based on feature stack band count
BAND_NAMES = None

#  PARAMETERS
N_TREES = 200
TEST_SIZE = 0.2
MAX_PIXELS_PER_POLYGON = 50
MIN_CONNECTED = 4
FOCAL_SIZE = 3
SEED = 42

#  CLASSIFIER DEFINITIONS

def get_classifiers(seed):
    """
    Return a dict of classifier name -> (classifier_instance, needs_scaling).
    SVM-RBF requires feature scaling; tree-based methods do not.
    """
    classifiers = {
        'RF': (
            RandomForestClassifier(
                n_estimators=N_TREES,
                class_weight='balanced',
                random_state=seed,
                n_jobs=-1,
            ),
            False,  # no scaling needed
        ),
        'SVM': (
            SVC(
                kernel='rbf',
                C=10,
                gamma='scale',
                class_weight='balanced',
                random_state=seed,
                cache_size=1000,  # MB — speeds up large datasets
            ),
            True,  # scaling required
        ),
    }

    # XGBoost or fallback
    if HAS_XGBOOST:
        classifiers['XGB'] = (
            XGBClassifier(
                n_estimators=N_TREES,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                eval_metric='mlogloss',
                random_state=seed,
                n_jobs=-1,
            ),
            False,
        )
    else:
        classifiers['XGB'] = (
            GradientBoostingClassifier(
                n_estimators=N_TREES,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                random_state=seed,
            ),
            False,
        )

    return classifiers


#  DATA LOADING

def load_feature_stack(path):
    """Load the feature stack and auto-detect sensor from band count."""
    global BAND_NAMES
    print(f'Loading feature stack: {path}')
    with rasterio.open(path) as src:
        data = src.read()  # shape: (bands, rows, cols)
        profile = src.profile.copy()
        transform = src.transform
        crs = src.crs

    n_bands = data.shape[0]

    if n_bands == 20:
        BAND_NAMES = BAND_NAMES_S2
        print(f'  Detected Sentinel-2 stack ({n_bands} bands)')
    elif n_bands == 16:
        if 'SR' in str(path).upper():
            BAND_NAMES = BAND_NAMES_L9_SR
            print(f'  Detected Landsat 9 SR stack ({n_bands} bands)')
        else:
            BAND_NAMES = BAND_NAMES_L9_TOA
            print(f'  Detected Landsat 9 TOA stack ({n_bands} bands)')
    else:
        BAND_NAMES = [f'Band_{i+1}' for i in range(n_bands)]
        print(f'  WARNING: unexpected band count ({n_bands}) — using generic names')
        print(f'  Expected 20 (S2) or 16 (L9). Check your GEE export.')

    print(f'  Shape: {data.shape} ({n_bands} bands, {data.shape[1]} rows, {data.shape[2]} cols)')
    return data, profile, transform, crs


def load_training_polygons(shp_path, raster_crs):
    """Load training polygons and reproject to match raster CRS if needed."""
    print(f'Loading training polygons: {shp_path}')
    gdf = gpd.read_file(shp_path)

    if 'Class' not in gdf.columns:
        sys.exit("ERROR: Shapefile must contain a 'Class' field with integer values 1-4.")

    gdf['Class'] = gdf['Class'].astype(int)

    # Validate class values
    invalid = gdf[~gdf['Class'].isin(CLASS_LABELS)]
    if len(invalid) > 0:
        print(f'  WARNING: {len(invalid)} polygons have Class values outside 1-4, skipping them')
        gdf = gdf[gdf['Class'].isin(CLASS_LABELS)].copy()

    # Reproject if CRS doesn't match
    if gdf.crs != raster_crs:
        print(f'  Reprojecting polygons from {gdf.crs} to {raster_crs}')
        gdf = gdf.to_crs(raster_crs)

    # Assign unique polygon IDs for group-based splitting
    gdf['polygon_id'] = range(len(gdf))

    print(f'  Total polygons: {len(gdf)}')
    for cls_val, cls_name in CLASS_NAMES.items():
        count = (gdf['Class'] == cls_val).sum()
        print(f'    Class {cls_val} ({cls_name}): {count} polygons')

    return gdf


#  PIXEL EXTRACTION

def extract_samples(data, gdf, transform, max_per_polygon=MAX_PIXELS_PER_POLYGON):
    """
    Extract pixel samples from polygons, capping per-polygon contribution
    to prevent large polygons from dominating any class.
    Returns feature array X, labels y, and polygon IDs for group splitting.
    """
    print(f'Extracting pixel samples (max {max_per_polygon} per polygon)...')
    rows, cols = data.shape[1], data.shape[2]
    rng = np.random.RandomState(SEED)

    # Rasterize class labels (all polygons at once)
    class_raster = rasterize(
        [(geom, cls) for geom, cls in zip(gdf.geometry, gdf['Class'])],
        out_shape=(rows, cols),
        transform=transform,
        fill=0,
        dtype='uint8',
    )

    # Rasterize polygon IDs (for group-based splitting)
    # Add 1 so that polygon_id=0 doesn't collide with fill=0
    pid_raster = rasterize(
        [(geom, pid + 1) for geom, pid in zip(gdf.geometry, gdf['polygon_id'])],
        out_shape=(rows, cols),
        transform=transform,
        fill=0,
        dtype='int32',
    )

    # Extract all labelled pixels
    labelled_mask = class_raster > 0
    pixel_locs = np.where(labelled_mask)

    all_y = class_raster[pixel_locs]
    all_groups = pid_raster[pixel_locs] - 1  # restore original polygon_id
    all_X = data[:, pixel_locs[0], pixel_locs[1]].T  # shape: (n_pixels, n_bands)

    # Filter out nodata / NaN pixels
    valid_mask = np.all(np.isfinite(all_X), axis=1)
    all_X = all_X[valid_mask]
    all_y = all_y[valid_mask]
    all_groups = all_groups[valid_mask]

    # Subsample per polygon to cap contribution
    keep_indices = []
    trimmed = 0
    for pid in np.unique(all_groups):
        pid_indices = np.where(all_groups == pid)[0]
        if len(pid_indices) <= max_per_polygon:
            keep_indices.append(pid_indices)
        else:
            keep_indices.append(rng.choice(pid_indices, max_per_polygon, replace=False))
            trimmed += 1

    keep_indices = np.concatenate(keep_indices)
    X = all_X[keep_indices]
    y = all_y[keep_indices]
    groups = all_groups[keep_indices]

    print(f'  Total samples extracted: {len(y)} ({trimmed} polygons trimmed)')
    for cls_val, cls_name in CLASS_NAMES.items():
        count = (y == cls_val).sum()
        print(f'    Class {cls_val} ({cls_name}): {count} pixels')

    return X, y, groups


#  TRAIN / VALIDATE SPLIT

def split_by_polygon(X, y, groups, test_size, seed):
    """
    Stratified polygon-level train/validation split.
    Splits within each class so no class gets starved in validation.
    """
    print(f'Stratified polygon split (test_size={test_size}, seed={seed})...')
    rng = np.random.RandomState(seed)

    train_mask = np.zeros(len(y), dtype=bool)
    val_mask = np.zeros(len(y), dtype=bool)

    for cls_val in CLASS_LABELS:
        cls_mask = y == cls_val
        cls_polygon_ids = np.unique(groups[cls_mask])

        rng.shuffle(cls_polygon_ids)
        n_val = max(1, int(len(cls_polygon_ids) * test_size))
        val_pids = set(cls_polygon_ids[:n_val].tolist())

        # Vectorized assignment
        cls_indices = np.where(cls_mask)[0]
        cls_groups = groups[cls_indices]
        is_val = np.isin(cls_groups, list(val_pids))
        val_mask[cls_indices[is_val]] = True
        train_mask[cls_indices[~is_val]] = True

    X_train, X_val = X[train_mask], X[val_mask]
    y_train, y_val = y[train_mask], y[val_mask]

    print(f'  Training samples: {len(y_train)}')
    print(f'  Validation samples: {len(y_val)}')
    for cls_val, cls_name in CLASS_NAMES.items():
        n_train = (y_train == cls_val).sum()
        n_val = (y_val == cls_val).sum()
        print(f'    {cls_name}: {n_train} train / {n_val} val')

    return X_train, X_val, y_train, y_val


#  CLASSIFICATION

def train_classifier(clf, X_train, y_train, clf_name, needs_scaling):
    """
    Train a classifier. If needs_scaling, fit a StandardScaler first.
    Returns (trained_clf, scaler_or_None).
    """
    scaler = None
    X_fit = X_train

    if needs_scaling:
        print(f'  Scaling features for {clf_name}...')
        scaler = StandardScaler()
        X_fit = scaler.fit_transform(X_train)

    print(f'Training {clf_name}...')
    t0 = time.time()

    # XGBoost requires labels starting from 0
    if clf_name == 'XGB' and HAS_XGBOOST:
        y_shifted = y_train - 1  # shift 1-4 -> 0-3
        clf.fit(X_fit, y_shifted)
    else:
        clf.fit(X_fit, y_train)

    elapsed = time.time() - t0
    print(f'  {clf_name} training complete in {elapsed:.1f}s.')
    return clf, scaler


def classify_raster(data, clf, clf_name, scaler=None):
    """Apply classifier to full raster. Returns 2D classified array."""
    print(f'Classifying full raster with {clf_name}...')
    n_bands, rows, cols = data.shape

    if clf_name == 'SVM':
        print(f'  NOTE: SVM pixel-wise prediction is slow on large rasters.')

    # Reshape to (n_pixels, n_bands)
    flat = data.reshape(n_bands, -1).T  # shape: (rows*cols, n_bands)

    # Identify valid (non-NaN) pixels
    valid_mask = np.all(np.isfinite(flat), axis=1)
    n_valid = valid_mask.sum()
    print(f'  Valid pixels: {n_valid:,} / {len(valid_mask):,}')

    # Scale if needed
    X_predict = flat[valid_mask]
    if scaler is not None:
        X_predict = scaler.transform(X_predict)

    # Predict only valid pixels
    result = np.full(len(valid_mask), NODATA, dtype=np.uint8)

    if n_valid > 0:
        t0 = time.time()
        preds = clf.predict(X_predict)

        # Shift XGBoost labels back: 0-3 -> 1-4
        if clf_name == 'XGB' and HAS_XGBOOST:
            preds = preds + 1

        result[valid_mask] = preds.astype(np.uint8)
        elapsed = time.time() - t0
        print(f'  Prediction complete in {elapsed:.1f}s.')

    classified = result.reshape(rows, cols)
    return classified


#  POST-PROCESSING

def postprocess(classified, min_connected, focal_size):
    """
    Post-processing:
    1. Reassign small isolated patches to the majority class of their neighbors
    2. Apply focal mode smoothing to clean boundaries
    """
    print(f'Post-processing (min_connected={min_connected}, focal_size={focal_size})...')
    result = classified.copy()

    # Track original nodata so we never fill pixels that were never classified
    original_nodata = classified == NODATA

    # --- Reassign small patches to neighbor majority ---
    result = reassign_small_patches(result, original_nodata, min_connected, focal_size)

    # --- Apply focal mode smoothing ---
    result = apply_focal_mode(result, original_nodata, focal_size)

    print('  Post-processing complete.')
    return result


def reassign_small_patches(classified, original_nodata, min_size, focal_size):
    """Reassign small connected components to the majority class of their neighbors."""
    result = classified.copy()
    unique_classes = [c for c in np.unique(result) if c != NODATA]
    reassigned = 0

    for cls in unique_classes:
        class_mask = result == cls
        labeled, n_components = label(class_mask)
        for comp_id in range(1, n_components + 1):
            comp_mask = labeled == comp_id
            if comp_mask.sum() < min_size:
                # Find neighboring pixels (dilate the patch mask)
                dilated = binary_dilation(comp_mask, iterations=focal_size)
                neighbor_mask = dilated & ~comp_mask & ~original_nodata

                if neighbor_mask.sum() > 0:
                    neighbor_vals = result[neighbor_mask]
                    neighbor_vals = neighbor_vals[(neighbor_vals != NODATA) &
                                                  (neighbor_vals >= 1) &
                                                  (neighbor_vals <= 4)]
                    if len(neighbor_vals) > 0:
                        counts = np.bincount(neighbor_vals.astype(int), minlength=5)
                        majority_class = np.argmax(counts[1:]) + 1
                        result[comp_mask] = majority_class
                        reassigned += comp_mask.sum()
                        continue

                # Fallback: no valid neighbors, set to NODATA
                result[comp_mask] = NODATA
                reassigned += comp_mask.sum()

    print(f'    Reassigned {reassigned} pixels from small patches')
    return result


def apply_focal_mode(classified, original_nodata, size):
    """Apply focal mode (majority) filter to smooth boundaries."""
    def mode_filter(values):
        valid_vals = values[(values != NODATA) & (values >= 1) & (values <= 4)]
        if len(valid_vals) == 0:
            return NODATA
        counts = np.bincount(valid_vals.astype(int), minlength=5)
        return np.argmax(counts[1:]) + 1

    result = generic_filter(
        classified.astype(float),
        mode_filter,
        size=size,
        mode='constant',
        cval=NODATA,
    ).astype(np.uint8)

    # Only preserve ORIGINAL nodata — don't re-punch holes from small patch removal
    result[original_nodata] = NODATA
    return result


#  ACCURACY ASSESSMENT

def assess_accuracy(clf, X_val, y_val, clf_name, scaler=None):
    """Compute and return accuracy metrics."""
    print(f'Computing accuracy metrics for {clf_name}...')

    X_eval = X_val
    if scaler is not None:
        X_eval = scaler.transform(X_val)

    y_pred = clf.predict(X_eval)

    # Shift XGBoost labels back: 0-3 -> 1-4
    if clf_name == 'XGB' and HAS_XGBOOST:
        y_pred = y_pred + 1

    cm = confusion_matrix(y_val, y_pred, labels=CLASS_LABELS)
    report = classification_report(
        y_val, y_pred,
        labels=CLASS_LABELS,
        target_names=[CLASS_NAMES[i] for i in CLASS_LABELS],
        output_dict=True,
        zero_division=0,
    )
    kappa = cohen_kappa_score(y_val, y_pred)
    f1_per_class = f1_score(y_val, y_pred, labels=CLASS_LABELS, average=None, zero_division=0)
    iou_per_class = jaccard_score(y_val, y_pred, labels=CLASS_LABELS, average=None, zero_division=0)
    overall_accuracy = np.trace(cm) / cm.sum()

    # Print results
    print(f'\n ACCURACY ASSESSMENT — {clf_name} ')
    print(f'Overall Accuracy: {overall_accuracy:.4f}')
    print(f'Kappa: {kappa:.4f}')
    print(f'\nConfusion Matrix:')
    print(cm)
    print(f'\nPer-Class Metrics:')
    print(f'  {"Class":<25} {"F1":>8} {"IoU":>8} {"Precision":>10} {"Recall":>8}')
    print(f'  {"-"*60}')
    for idx, cls_val in enumerate(CLASS_LABELS):
        cls_name_ = CLASS_NAMES[cls_val]
        print(f'  {cls_name_:<25} {f1_per_class[idx]:>8.4f} {iou_per_class[idx]:>8.4f} '
              f'{report[cls_name_]["precision"]:>10.4f} {report[cls_name_]["recall"]:>8.4f}')

    # Build results dict
    results = {
        'classifier': clf_name,
        'overall_accuracy': float(overall_accuracy),
        'kappa': float(kappa),
        'confusion_matrix': cm.tolist(),
        'per_class': {},
    }
    for idx, cls_val in enumerate(CLASS_LABELS):
        cls_name_ = CLASS_NAMES[cls_val]
        results['per_class'][cls_name_] = {
            'f1': float(f1_per_class[idx]),
            'iou': float(iou_per_class[idx]),
            'precision': float(report[cls_name_]['precision']),
            'recall': float(report[cls_name_]['recall']),
        }

    return results


#  FEATURE IMPORTANCE

def report_feature_importance(clf, clf_name, X_val, y_val, scaler, output_dir):
    """
    Print and plot feature importance.
    - RF / XGB: use native .feature_importances_
    - SVM: use sklearn permutation_importance on validation set
    """
    print(f'\n FEATURE IMPORTANCE — {clf_name}')

    if hasattr(clf, 'feature_importances_'):
        importances = clf.feature_importances_
        method = 'native'
    else:
        # Permutation importance for SVM
        print(f'  Computing permutation importance (this may take a moment)...')
        X_eval = scaler.transform(X_val) if scaler is not None else X_val
        y_eval = y_val
        # For XGBoost with shifted labels
        if clf_name == 'XGB' and HAS_XGBOOST:
            y_eval = y_val - 1
        perm_result = permutation_importance(
            clf, X_eval, y_eval, n_repeats=10, random_state=SEED, n_jobs=-1,
        )
        importances = perm_result.importances_mean
        method = 'permutation'

    indices = np.argsort(importances)[::-1]
    for rank, idx in enumerate(indices):
        print(f'  {rank+1:>2}. {BAND_NAMES[idx]:<20} {importances[idx]:.4f}')

    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    sorted_idx = np.argsort(importances)
    ax.barh(range(len(BAND_NAMES)), importances[sorted_idx], color='#2196F3')
    ax.set_yticks(range(len(BAND_NAMES)))
    ax.set_yticklabels([BAND_NAMES[i] for i in sorted_idx])
    ax.set_xlabel('Importance')
    method_label = '(permutation)' if method == 'permutation' else ''
    ax.set_title(f'{clf_name} Feature Importance {method_label}')
    plt.tight_layout()
    plot_path = Path(output_dir) / f'feature_importance_{clf_name}.png'
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    print(f'  Saved: {plot_path}')

    return {BAND_NAMES[i]: float(importances[i]) for i in range(len(BAND_NAMES))}


#  AREA CALCULATION

def calculate_areas(classified, transform, crs):
    """Calculate area per class in square kilometers."""
    print('\n CLASS AREAS ')

    if crs and crs.is_geographic:
        center_row = classified.shape[0] // 2
        center_lat = transform[5] + center_row * transform[4]
        lat_rad = np.radians(abs(center_lat))
        m_per_deg_lat = 111320.0
        m_per_deg_lon = 111320.0 * np.cos(lat_rad)
        pixel_width_m = abs(transform[0]) * m_per_deg_lon
        pixel_height_m = abs(transform[4]) * m_per_deg_lat
        pixel_area_km2 = (pixel_width_m * pixel_height_m) / 1e6
        print(f'  CRS is geographic — approx pixel size: {pixel_width_m:.1f} x {pixel_height_m:.1f} m')
    else:
        pixel_area_km2 = abs(transform[0] * transform[4]) / 1e6

    areas = {}
    for cls_val, cls_name in CLASS_NAMES.items():
        n_pixels = (classified == cls_val).sum()
        area_km2 = n_pixels * pixel_area_km2
        areas[cls_name] = float(area_km2)
        print(f'  {cls_name}: {area_km2:.3f} sq km ({n_pixels:,} pixels)')

    return areas


#  OUTPUT

def save_raster(array, profile, path, nodata=NODATA):
    """Save a 2D array as a single-band GeoTIFF with LZW compression."""
    out_profile = profile.copy()
    out_profile.update(
        dtype='uint8',
        count=1,
        nodata=nodata,
        compress='lzw',
    )
    with rasterio.open(path, 'w', **out_profile) as dst:
        dst.write(array.astype(np.uint8), 1)
    print(f'  Saved: {path}')


#  MAIN

def run_classification(feature_stack_path, training_polygons_path, output_dir_path, run_name):
    """Run the full classification pipeline for one feature stack, across all classifiers."""
    print(f'\n{"="*70}')
    print(f'  RUNNING: {run_name}')
    print(f'{"="*70}\n')

    output_dir = Path(output_dir_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    data, profile, transform, crs = load_feature_stack(feature_stack_path)
    gdf = load_training_polygons(training_polygons_path, crs)

    # Extract samples and split (shared across classifiers)
    X, y, groups = extract_samples(data, gdf, transform)
    X_train, X_val, y_train, y_val = split_by_polygon(X, y, groups, TEST_SIZE, SEED)

    # Get all classifiers
    classifiers = get_classifiers(SEED)
    run_results = {}

    for clf_name, (clf, needs_scaling) in classifiers.items():
        print(f'\n{"-"*50}')
        print(f'  Classifier: {clf_name}')
        print(f'{"-"*50}')

        # Train
        clf, scaler = train_classifier(clf, X_train, y_train, clf_name, needs_scaling)

        # Classify full raster
        classified = classify_raster(data, clf, clf_name, scaler)

        # Post-process
        postprocessed = postprocess(classified, MIN_CONNECTED, FOCAL_SIZE)

        # Save outputs (separate files per classifier)
        print(f'\nSaving outputs for {clf_name}...')
        save_raster(classified, profile, output_dir / f'classified_{clf_name}.tif')
        save_raster(postprocessed, profile, output_dir / f'classified_postprocessed_{clf_name}.tif')

        # Accuracy
        accuracy_results = assess_accuracy(clf, X_val, y_val, clf_name, scaler)

        # Feature importance
        importance_dict = report_feature_importance(
            clf, clf_name, X_val, y_val, scaler, output_dir
        )
        accuracy_results['feature_importance'] = importance_dict

        # Areas
        accuracy_results['areas_km2'] = calculate_areas(classified, transform, crs)
        accuracy_results['areas_km2_postprocessed'] = calculate_areas(postprocessed, transform, crs)

        # Save per-classifier report
        report_path = output_dir / f'accuracy_report_{clf_name}.json'
        with open(report_path, 'w') as f:
            json.dump(accuracy_results, f, indent=2)
        print(f'\n  Saved accuracy report: {report_path}')

        run_results[clf_name] = accuracy_results

    print(f'\n {run_name} COMPLETE (all classifiers)')
    return run_results


def main():
    all_results = {}
    for run in RUNS:
        results = run_classification(
            feature_stack_path=run['feature_stack'],
            training_polygons_path=TRAINING_POLYGONS,
            output_dir_path=run['output_dir'],
            run_name=run['name'],
        )
        all_results[run['name']] = results

    # Print comparison summary
    print(f'\n{"="*70}')
    print('  COMPARISON SUMMARY')
    print(f'{"="*70}')

    for clf_name in ['RF', 'SVM', 'XGB']:
        print(f'\n  --- {clf_name} ---')
        print(f'  {"Run":<12} {"OA":>8} {"Kappa":>8}', end='')
        for cls_name in CLASS_NAMES.values():
            print(f'  {cls_name[:10]:>10}', end='')
        print()
        print(f'  {"-"*70}')

        for run_name, run_results in all_results.items():
            if clf_name in run_results:
                r = run_results[clf_name]
                print(f'  {run_name:<12} {r["overall_accuracy"]:>8.4f} {r["kappa"]:>8.4f}', end='')
                for cls_name in CLASS_NAMES.values():
                    f1 = r['per_class'][cls_name]['f1']
                    print(f'  {f1:>10.4f}', end='')
                print()

    print(f'\n ALL RUNS COMPLETE')


if __name__ == '__main__':
    main()
