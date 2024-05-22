#!/bin/bash

#DATASETS="Birds StanfordCars StanfordDogs"
DATASETS="Birds"
DETECTIONS="OneStage DETR"
HEATMAPS="GaussHeatmap LogisticHeatmap"
OCCLUSIONS="encoder image"
PENALTIES="0 0.001"

for DATASET in $DATASETS; do
MODEL="model-$DATASET-none.pth"
if [ ! -f $MODEL ]; then
    echo $MODEL
    python3 train.py $MODEL $DATASET
fi

for DETECTION in $DETECTIONS; do
for HEATMAP in $HEATMAPS; do
for PENALTY in $PENALTIES; do
OCCLUSION="encoder"
MODEL="model-$DATASET-$OCCLUSION-$DETECTION-$HEATMAP-$PENALTY.pth"
if [ ! -f $MODEL ]; then
    echo $MODEL
    python3 train.py $MODEL $DATASET --occlusion $OCCLUSION --detection $DETECTION --heatmap $HEATMAP --penalty $PENALTY
fi

for OCCLUSION in $OCCLUSIONS; do
MODEL="model-$DATASET-$OCCLUSION-$DETECTION-$HEATMAP-$PENALTY-adversarial.pth"
if [ ! -f $MODEL ]; then
    echo $MODEL
    python3 train.py $MODEL $DATASET --occlusion $OCCLUSION --detection $DETECTION --heatmap $HEATMAP --penalty $PENALTY --adversarial
fi
done

done; done; done; done
