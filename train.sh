#!/bin/bash
# train different model types

DATASETS="Birds StanfordCars StanfordDogs"

for DATASET in $DATASETS; do
if [ $DATASET = Birds ]; then NCLASSES="10 50 100 200"; fi
if [ $DATASET = StanfordCars ]; then NCLASSES="10 50 100 196"; fi
if [ $DATASET = StanfordDogs ]; then NCLASSES="10 50 80 120"; fi
for NCLASS in $NCLASSES; do

MODELS="OnlyClass"
for MODEL in $MODELS; do
NAME="model-$DATASET$NCLASS-$MODEL.pth"
if [ ! -f $NAME ]; then
    echo $NAME
    sbatch python train.py $NAME "$DATASET$NCLASS" $MODEL --epochs 1000
fi
done

# ViT: train for longer
#MODELS="ViTb ViTl ViTr"
#for MODEL in $MODELS; do
#NAME="model-$DATASET$NCLASS-$MODEL.pth"
#if [ ! -f $NAME ]; then
#    echo $NAME
#    sbatch python train.py $NAME "$DATASET$NCLASS" $MODEL --epochs 1000
#fi
#done

MODELS="Heatmap"
OCCLUSIONS="encoder"
for MODEL in $MODELS; do
for OCCLUSION in $OCCLUSIONS; do
    sbatch ./train2.sh "$DATASET$NCLASS" Heatmap none $OCCLUSION
done
done

HEATMAPS="GaussHeatmap LogisticHeatmap"
MODELS="SimpleDet FasterRCNN FCOS DETR"
OCCLUSIONS="encoder"
for HEATMAP in $HEATMAPS; do
for MODEL in $MODELS; do
for OCCLUSION in $OCCLUSIONS; do
    sbatch ./train2.sh "$DATASET$NCLASS" $MODEL $HEATMAP $OCCLUSION
done
done
done

done
done
