#!/bin/bash

DATASET=Birds
#MODELS="OnlyClass Heatmap"
MODELS=""
for MODEL in $MODELS; do
echo "model-$DATASET-$MODEL"
python3 train.py model-$DATASET-$MODEL.pth Birds $MODEL
done

MODELS="SimpleDet"
PENALTIES="0 0.001 0.1 1 10"
for MODEL in $MODELS; do
for PENALTY in $PENALTIES; do
echo "model-$DATASET-$MODEL-$PENALTY"
python3 train.py model-$DATASET-$MODEL-$PENALTY.pth Birds $MODEL --penalty $PENALTY
done
done