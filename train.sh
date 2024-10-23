#!/bin/bash
# train different model types

#DATASETS="Birds StanfordCars StanfordDogs"
DATASETS="$1"

for DATASET in $DATASETS; do

MODELS="OnlyClass"
for MODEL in $MODELS; do
NAME="model-$DATASET-$MODEL.pth"
if [ ! -f $NAME ]; then
echo $NAME
sbatch python train.py model-$DATASET-$MODEL.pth $DATASET $MODEL
fi
done

MODEL=Heatmap
#OCCLUSIONS="image encoder"
OCCLUSIONS="encoder"
for OCCLUSION in $OCCLUSIONS; do
    sbatch ./train2.sh $DATASET $MODEL none $OCCLUSION
done

MODELS="SimpleDet FasterRCNN FCOS DETR"
HEATMAPS="GaussHeatmap LogisticHeatmap"

for HEATMAP in $HEATMAPS; do
for MODEL in $MODELS; do
for OCCLUSION in $OCCLUSIONS; do
    sbatch ./train2.sh $DATASET $MODEL $HEATMAP $OCCLUSION
done
done
done

done
