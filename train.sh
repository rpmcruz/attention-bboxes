#!/bin/bash

DATASETS="Birds StanfordCars StanfordDogs"
PENALTIES="0 0.001 0.1 1 10"

for DATASET in $DATASETS; do

#MODELS="OnlyClass ProtoPNet ViT"
MODELS="OnlyClass"
for MODEL in $MODELS; do
NAME="model-$DATASET-$MODEL.pth"
if [ ! -f $NAME ]; then
echo $NAME
sbatch --short python3 train.py model-$DATASET-$MODEL.pth $DATASET $MODEL
fi
done

MODEL=Heatmap
for PENALTY in $PENALTIES; do
NAME="model-$DATASET-$MODEL-l1-$PENALTY-sigmoid.pth"
if [ ! -f $NAME ]; then
echo $NAME
sbatch --short python3 train.py $NAME $DATASET $MODEL --penalty-l1 $PENALTY --sigmoid
fi
done

MODELS="SimpleDet FasterRCNN FCOS DETR"
HEATMAPS="GaussHeatmap LogisticHeatmap"
for HEATMAP in $HEATMAPS; do
for MODEL in $MODELS; do
for PENALTY in $PENALTIES; do
NAME="model-$DATASET-$MODEL-l1-$PENALTY-heatmap-$HEATMAP-sigmoid.pth"
if [ ! -f $NAME ]; then
echo $NAME
sbatch --short python3 train.py $NAME $DATASET $MODEL --penalty-l1 $PENALTY --heatmap $HEATMAP --sigmoid
fi
done
done
done

done
