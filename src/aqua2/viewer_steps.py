"""Stage parameter definitions and validation, independent of Tk widgets."""
import math

PREPROCESS_FIELDS=[('regMaskGap','Border Crop (pixels)',int,0),('smoXY','Spatial Smoothing σ',float,0),('movAvgWin','Baseline Window (frames)',int,1),
    ('cut','Baseline Segment (frames)',int,1),('medSmo','Median Radius',int,0),('frameRate','Frame Interval (s)',float,0),('spatialRes','Spatial Resolution (μm)',float,0),
    ('registrateCorrect','Registration Mode 1/2/3',int,1),('bleachCorrect','Bleaching Mode 1/2/3',int,1)]
DETECTION_FIELDS=[('thrARScl','Activity Threshold (σ)',float,0),('minSize','Min Size (pixels)',int,1),('minDur','Min Duration (frames)',int,1),
    ('sigThr','Significance Threshold',float,0),('maxDelay','Merge Delay',float,0),('seedSzRatio','Seed Size Ratio',float,0)]
DETECTION_FIELDS += [('maxSize','Max Size (pixels)',float,1),('circularityThr','Circularity',float,0),('spaMergeDist','Spatial Merge Distance',float,0),
    ('sourceSzRatio','Source Size Ratio',float,0),('sourceSensitivity','Source Sensitivity',int,1),('gloDur','Global min duration (frames)',int,1),
    ('gtwSmo','Propagation Smoothing',float,0),('minShow1','Display Threshold',float,0),('propthrmin','Propagation min',float,0),
    ('propthrstep','Propagation step',float,0),('propthrmax','Propagation max',float,0),('compress','Footprint Duration Ratio',float,0),
    ('gapExt','Segmentation context (frames)',int,0),('TPatch','Temporal downsampling',int,1)]
PREPROCESS_FIELDS += [('northx','North Vector X',float,-float('inf')),('northy','North Vector Y',float,-float('inf'))]
DETECTION_FLAGS=[('detectGlo','Global Events',False),('propMetric','Propagation Features',False),('networkFeatures','Network Features',False)]
FIXED_DETECTION_FLAGS=dict(needTemp=True,needSpa=True,needRefine=False,needGrow=False,ignoreTau=True,whetherExtend=True)
PRIMARY_PREPROCESS=('smoXY','medSmo','registrateCorrect','bleachCorrect')
PRIMARY_DETECTION=('thrARScl','minSize','minDur')
PREPROCESS_KEYS=tuple(field[0] for field in PREPROCESS_FIELDS)


def validated(values,fields):
    result={}
    for key,label,cast,minimum in fields:
        value=float(values[key])
        if (not math.isfinite(value) and not (key=='maxSize' and value==float('inf'))) or value<minimum or (cast is int and value!=int(value)):raise ValueError(label+' is invalid')
        if key in ('frameRate','spatialRes','thrARScl') and value<=0:raise ValueError(label+' must be positive')
        if key in ('registrateCorrect','bleachCorrect') and value not in (1,2,3):raise ValueError(label+' must be 1, 2 or 3')
        if key=='seedSzRatio' and value>1:raise ValueError('Seed size ratio must be 0–1')
        if key=='sourceSensitivity' and value>10:raise ValueError('Source sensitivity must be 1–10')
        result[key]=cast(value)
    return result
