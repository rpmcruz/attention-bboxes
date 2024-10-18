#!/bin/bash

DATASETS="Birds StanfordCars StanfordDogs"
#PENALTIES="0 0.001 0.1 1 10"
PENALTIES="0 0.001 0.1 1"

for DATASET in $DATASETS; do

#MODELS="OnlyClass ProtoPNet ViT"
MODELS="OnlyClass"
for MODEL in $MODELS; do
NAME="model-$DATASET-$MODEL.pth"
if [ ! -f $NAME ]; then
echo $NAME
sbatch python train.py model-$DATASET-$MODEL.pth $DATASET $MODEL
fi
done

MODEL=Heatmap
OCCLUSIONS="image encoder"
for OCCLUSION in $OCCLUSIONS; do
for PENALTY in $PENALTIES; do
NAME="model-$DATASET-$MODEL-l1-$PENALTY-occlusion-$OCCLUSION-sigmoid.pth"
if [ ! -f $NAME ]; then
echo $NAME
sbatch python train.py $NAME $DATASET $MODEL --penalty-l1 $PENALTY --occlusion $OCCLUSION --sigmoid
fi
NAME="model-$DATASET-$MODEL-l1-$PENALTY-occlusion-$OCCLUSION-sigmoid-adversarial.pth"
if [ ! -f $NAME ]; then
echo $NAME
sbatch python train.py $NAME $DATASET $MODEL --penalty-l1 $PENALTY --occlusion $OCCLUSION --sigmoid --adversarial
fi
done
done

MODELS="SimpleDet FasterRCNN FCOS DETR"
HEATMAPS="GaussHeatmap LogisticHeatmap"

for HEATMAP in $HEATMAPS; do
for MODEL in $MODELS; do
for PENALTY in $PENALTIES; do
for OCCLUSION in $OCCLUSIONS; do
NAME="model-$DATASET-$MODEL-l1-$PENALTY-heatmap-$HEATMAP-occlusion-$OCCLUSION-sigmoid.pth"
if [ ! -f $NAME ]; then
echo $NAME
sbatch python train.py $NAME $DATASET $MODEL --penalty-l1 $PENALTY --heatmap $HEATMAP --occlusion $OCCLUSION --sigmoid
fi
NAME="model-$DATASET-$MODEL-l1-$PENALTY-heatmap-$HEATMAP-occlusion-$OCCLUSION-sigmoid-adversarial.pth"
if [ ! -f $NAME ]; then
echo $NAME
sbatch python train.py $NAME $DATASET $MODEL --penalty-l1 $PENALTY --heatmap $HEATMAP --occlusion $OCCLUSION --sigmoid --adversarial
fi
done
done
done
done

done
