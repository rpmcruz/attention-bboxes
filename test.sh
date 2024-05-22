#!/bin/bash

#DATASETS="Birds StanfordCars StanfordDogs"
DATASETS="Birds"
#DETECTIONS="OneStage DETR"
DETECTIONS="OneStage"
HEATMAPS="GaussHeatmap LogisticHeatmap"
OCCLUSIONS="encoder image"
PENALTIES="0 0.1 1"

for DATASET in $DATASETS; do
MODEL="model-$DATASET-none.pth"
python3 test.py $MODEL $DATASET

for DETECTION in $DETECTIONS; do
for HEATMAP in $HEATMAPS; do
for PENALTY in $PENALTIES; do
OCCLUSION="encoder"
MODEL="model-$DATASET-$OCCLUSION-$DETECTION-$HEATMAP-$PENALTY.pth"
python3 test.py $MODEL $DATASET

for OCCLUSION in $OCCLUSIONS; do
MODEL="model-$DATASET-$OCCLUSION-$DETECTION-$HEATMAP-$PENALTY-adversarial.pth"
python3 test.py $MODEL $DATASET
done

done; done; done; done