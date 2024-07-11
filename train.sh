#!/bin/bash

DATASETS="Birds StanfordCars StanfordDogs"
PENALTIES="0 0.001 0.1 1 10"

for DATASET in $DATASETS; do

MODELS="OnlyClass ProtoPNet ViT"
for MODEL in $MODELS; do
echo "model-$DATASET-$MODEL"
python3 train.py model-$DATASET-$MODEL.pth $DATASET $MODEL
done

MODEL=Heatmap
for PENALTY in $PENALTIES; do
echo "model-$DATASET-$MODEL-l1-$PENALTY-sigmoid"
python3 train.py model-$DATASET-$MODEL-l1-$PENALTY-sigmoid.pth $DATASET $MODEL --penalty-l1 $PENALTY --sigmoid
done

MODELS="SimpleDet FasterRCNN FCOS DETR"
HEATMAPS="GaussHeatmap LogisticHeatmap"
for HEATMAP in $HEATMAPS; do
for MODEL in $MODELS; do
for PENALTY in $PENALTIES; do
echo "model-$DATASET-$MODEL-l1-$PENALTY-heatmap-$HEATMAP-sigmoid.pth"
python3 train.py model-$DATASET-$MODEL-l1-$PENALTY-heatmap-$HEATMAP-sigmoid.pth $DATASET $MODEL --penalty-l1 $PENALTY --heatmap $HEATMAP --sigmoid
done
done
done

done
