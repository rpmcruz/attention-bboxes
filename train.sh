#!/bin/bash

DATASETS="Birds StanfordCars StanfordDogs"
DETECTIONS="OneStage DETR"
HEATMAPS="GaussHeatmap LogisticHeatmap"
OCCLUSIONS="encoder image"
PENALTIES="0 0.1 1"

for DATASET in $DATASETS; do
OUT="model-$DATASET-none.pth"
if [ ! -f $OUT ]; then
    echo $OUT
    python3 train.py $OUT $DATASET
fi

for DETECTION in $DETECTIONS; do
for HEATMAP in $HEATMAPS; do
for OCCLUSION in $OCCLUSIONS; do
for PENALTY in $PENALTIES; do
OUT="model-$DATASET-$OCCLUSION-$DETECTION-$HEATMAP-$PENALTY.pth"
if [ ! -f $OUT ]; then
    echo $OUT
    python3 train.py $OUT $DATASET --occlusion $OCCLUSION --detection $DETECTION --heatmap $HEATMAP --penalty $PENALTY
fi
OUT="model-$DATASET-$OCCLUSION-$DETECTION-$HEATMAP-$PENALTY-adversarial.pth"
if [ ! -f $OUT ]; then
    echo $OUT
    python3 train.py $OUT $DATASET --occlusion $OCCLUSION --detection $DETECTION --heatmap $HEATMAP --penalty $PENALTY --adversarial
fi

done; done; done; done; done