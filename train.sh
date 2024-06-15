#!/bin/bash

DATASET=Birds
MODELS="OnlyClass Heatmap SimpleDet"
for MODEL in $MODELS; do
echo "model-$DATASET-$MODEL"
python3 train.py model-$DATASET-$MODEL.pth Birds $MODEL
done