//EXPORT ALL FEATURE STACKS (2024) 

//  DEM AND SLOPE 
var dem = ee.Image('USGS/SRTMGL1_003').clip(ROI);
var slope = ee.Terrain.slope(dem).rename('slope');

//  CLOUD MASKING FUNCTIONS 

function maskS2SR(image) {
  var scl = image.select('SCL');
  var mask = scl.neq(3).and(scl.neq(7)).and(scl.neq(8)).and(scl.neq(9)).and(scl.neq(10));
  return image.updateMask(mask).divide(10000);
}

function maskS2TOA(image) {
  var qa = image.select('QA60');
  var mask = qa.bitwiseAnd(1 << 10).eq(0).and(qa.bitwiseAnd(1 << 11).eq(0));
  return image.updateMask(mask).divide(10000);
}

function maskLandsat(image) {
  var qa = image.select('QA_PIXEL');
  var mask = qa.bitwiseAnd(1 << 3).eq(0).and(qa.bitwiseAnd(1 << 5).eq(0));
  return image.updateMask(mask);
}

function applyScaleFactors(image) {
  var scaled = image.select(['SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B6', 'SR_B7'])
                     .multiply(0.0000275).add(-0.2).clamp(0, 1);
  return image.addBands(scaled, null, true);
}

//  IMAGE COMPOSITES 

var s2sr = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
            .filterDate('2024-01-01', '2024-12-31').filterBounds(ROI)
            .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20))
            .map(maskS2SR).median().clip(ROI);

var s2toa = ee.ImageCollection('COPERNICUS/S2_HARMONIZED')
             .filterDate('2024-01-01', '2024-12-31').filterBounds(ROI)
             .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20))
             .map(maskS2TOA).median().clip(ROI);

var l9sr = ee.ImageCollection('LANDSAT/LC09/C02/T1_L2')
            .filterDate('2024-01-01', '2024-12-31').filterBounds(ROI)
            .map(maskLandsat).map(applyScaleFactors).median().clip(ROI);

var l9toa = ee.ImageCollection('LANDSAT/LC09/C02/T1_TOA')
             .filterDate('2024-01-01', '2024-12-31').filterBounds(ROI)
             .map(maskLandsat).median().clip(ROI);

             
//  HELPER: BUILD S2 FEATURE STACK 

function buildS2Stack(img) {
  var NDVI = img.normalizedDifference(['B8', 'B4']).rename('NDVI');
  var EVI = img.expression('2.5 * ((NIR - RED) / (NIR + 6 * RED - 7.5 * BLUE + 1))',
    {'NIR': img.select('B8'), 'RED': img.select('B4'), 'BLUE': img.select('B2')}).rename('EVI');
  var NDWI = img.normalizedDifference(['B3', 'B8']).rename('NDWI');
  var NDBI = img.normalizedDifference(['B11', 'B8']).rename('NDBI');
  var MVI = img.select('B8').subtract(img.select('B3'))
              .divide(img.select('B11').subtract(img.select('B3'))).rename('MVI');
  var SAVI = img.expression('((NIR - RED) / (NIR + RED + L)) * (1 + L)',
    {'NIR': img.select('B8'), 'RED': img.select('B4'), 'L': 0.5}).rename('SAVI');
  var LSWI = img.normalizedDifference(['B8', 'B11']).rename('LSWI');

  var nirInt = img.select('B8').multiply(10000).toInt();
  var glcm = nirInt.glcmTexture({size: 3});

  return img.select(['B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B11', 'B12'])
            .addBands([NDVI, EVI, NDWI, NDBI, MVI, SAVI, LSWI])
            .addBands(glcm.select('B8_contrast').rename('contrast'))
            .addBands(glcm.select('B8_idm').rename('homogeneity'))
            .addBands(glcm.select('B8_ent').rename('entropy'))
            .addBands(dem.rename('elevation'))
            .addBands(slope);
}

//  HELPER: BUILD L9 SR FEATURE STACK 

function buildL9SRStack(img) {
  var NDVI = img.normalizedDifference(['SR_B5', 'SR_B4']).rename('NDVI');
  var EVI = img.expression('2.5 * ((NIR - RED) / (NIR + 6 * RED - 7.5 * BLUE + 1))',
    {'NIR': img.select('SR_B5'), 'RED': img.select('SR_B4'), 'BLUE': img.select('SR_B2')}).rename('EVI');
  var NDWI = img.normalizedDifference(['SR_B3', 'SR_B5']).rename('NDWI');
  var NDBI = img.normalizedDifference(['SR_B6', 'SR_B5']).rename('NDBI');
  var MVI = img.select('SR_B5').subtract(img.select('SR_B3'))
              .divide(img.select('SR_B6').subtract(img.select('SR_B3'))).rename('MVI');
  var SAVI = img.expression('((NIR - RED) / (NIR + RED + L)) * (1 + L)',
    {'NIR': img.select('SR_B5'), 'RED': img.select('SR_B4'), 'L': 0.5}).rename('SAVI');
  var LSWI = img.normalizedDifference(['SR_B5', 'SR_B6']).rename('LSWI');

  var nirInt = img.select('SR_B5').multiply(1000).toInt();
  var glcm = nirInt.glcmTexture({size: 3});

  return img.select(['SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B6', 'SR_B7'])
            .addBands([NDVI, EVI, NDWI, NDBI, MVI, SAVI, LSWI])
            .addBands(glcm.select('SR_B5_contrast').rename('contrast'))
            .addBands(glcm.select('SR_B5_idm').rename('homogeneity'))
            .addBands(glcm.select('SR_B5_ent').rename('entropy'))
            .addBands(dem.rename('elevation'))
            .addBands(slope);
}

//  HELPER: BUILD L9 TOA FEATURE STACK 

function buildL9TOAStack(img) {
  var NDVI = img.normalizedDifference(['B5', 'B4']).rename('NDVI');
  var EVI = img.expression('2.5 * ((NIR - RED) / (NIR + 6 * RED - 7.5 * BLUE + 1))',
    {'NIR': img.select('B5'), 'RED': img.select('B4'), 'BLUE': img.select('B2')}).rename('EVI');
  var NDWI = img.normalizedDifference(['B3', 'B5']).rename('NDWI');
  var NDBI = img.normalizedDifference(['B6', 'B5']).rename('NDBI');
  var MVI = img.select('B5').subtract(img.select('B3'))
              .divide(img.select('B6').subtract(img.select('B3'))).rename('MVI');
  var SAVI = img.expression('((NIR - RED) / (NIR + RED + L)) * (1 + L)',
    {'NIR': img.select('B5'), 'RED': img.select('B4'), 'L': 0.5}).rename('SAVI');
  var LSWI = img.normalizedDifference(['B5', 'B6']).rename('LSWI');

  var nirInt = img.select('B5').multiply(1000).toInt();
  var glcm = nirInt.glcmTexture({size: 3});

  return img.select(['B2', 'B3', 'B4', 'B5', 'B6', 'B7'])
            .addBands([NDVI, EVI, NDWI, NDBI, MVI, SAVI, LSWI])
            .addBands(glcm.select('B5_contrast').rename('contrast'))
            .addBands(glcm.select('B5_idm').rename('homogeneity'))
            .addBands(glcm.select('B5_ent').rename('entropy'))
            .addBands(dem.rename('elevation'))
            .addBands(slope);
}

//  BUILD STACKS 

var s2srStack = buildS2Stack(s2sr);
var s2toaStack = buildS2Stack(s2toa);
var l9srStack = buildL9SRStack(l9sr);
var l9toaStack = buildL9TOAStack(l9toa);

print('S2 bands:', s2srStack.bandNames().size());   // 22
print('L9 bands:', l9srStack.bandNames().size());    // 18

//  VISUALIZATION 

Map.centerObject(ROI, 10);
Map.addLayer(s2sr, {bands: ['B8', 'B11', 'B3'], min: 0, max: 0.4}, 'S2 SR', false);
Map.addLayer(s2toa, {bands: ['B8', 'B11', 'B3'], min: 0, max: 0.4}, 'S2 TOA', false);
Map.addLayer(l9sr, {bands: ['SR_B5', 'SR_B6', 'SR_B3'], min: 0, max: 0.4}, 'L9 SR', false);
Map.addLayer(l9toa, {bands: ['B5', 'B6', 'B3'], min: 0, max: 0.4}, 'L9 TOA', false);

//  EXPORT ALL 

Export.image.toDrive({
  image: s2srStack.toFloat(),
  description: 'S2_SR_Feature_Stack_2024',
  region: ROI, scale: 10, crs: 'EPSG:4326', maxPixels: 1e13, fileFormat: 'GeoTIFF'
});

Export.image.toDrive({
  image: s2toaStack.toFloat(),
  description: 'S2_TOA_Feature_Stack_2024',
  region: ROI, scale: 10, crs: 'EPSG:4326', maxPixels: 1e13, fileFormat: 'GeoTIFF'
});

Export.image.toDrive({
  image: l9srStack.toFloat(),
  description: 'L9_SR_Feature_Stack_2024',
  region: ROI, scale: 30, crs: 'EPSG:4326', maxPixels: 1e13, fileFormat: 'GeoTIFF'
});

Export.image.toDrive({
  image: l9toaStack.toFloat(),
  description: 'L9_TOA_Feature_Stack_2024',
  region: ROI, scale: 30, crs: 'EPSG:4326', maxPixels: 1e13, fileFormat: 'GeoTIFF'
});
