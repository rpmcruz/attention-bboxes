#!/bin/bash

DATASETS="Birds StanfordCars StanfordDogs"
PENALTIES="0 0.001 0.1 1 10"

for DATASET in $DATASETS; do
MODEL=OnlyClass
echo "model-$DATASET-$MODEL"
python3 train.py model-$DATASET-$MODEL.pth $DATASET $MODEL

#MODEL=ProtoPNet
#echo "model-$DATASET-$MODEL"
#python3 train.py model-$DATASET-$MODEL.pth $DATASET $MODEL

MODEL=Heatmap
for PENALTY in $PENALTIES; do
#echo "model-$DATASET-$MODEL-l1-$PENALTY-softmax"
#python3 train.py model-$DATASET-$MODEL-l1-$PENALTY-softmax.pth $DATASET $MODEL --penalty-l1 $PENALTY
echo "model-$DATASET-$MODEL-l1-$PENALTY-sigmoid"
python3 train.py model-$DATASET-$MODEL-l1-$PENALTY-sigmoid.pth $DATASET $MODEL --penalty-l1 $PENALTY --sigmoid
done

MODELS="SimpleDet FasterRCNN FCOS DETR"
for MODEL in $MODELS; do
for PENALTY in $PENALTIES; do
#echo "model-$DATASET-$MODEL-l1-$PENALTY-softmax"
#python3 train.py model-$DATASET-$MODEL-l1-$PENALTY-softmax.pth $DATASET $MODEL --penalty-l1 $PENALTY
echo "model-$DATASET-$MODEL-l1-$PENALTY-sigmoid"
python3 train.py model-$DATASET-$MODEL-l1-$PENALTY-sigmoid.pth $DATASET $MODEL --penalty-l1 $PENALTY --sigmoid
done
done

done
