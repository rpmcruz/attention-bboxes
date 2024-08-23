#!/bin/bash

DATASETS="Birds StanfordCars StanfordDogs"
PENALTIES="0 0.001 0.1 1 10"
CAPTUM="IntegratedGradients GuidedGradCAM"
MODELS="SimpleDet FasterRCNN FCOS DETR"
HEATMAPS="GaussHeatmap LogisticHeatmap"

echo "model,dataset,acc,pg,degscore,sparsity"
for dataset in $DATASETS; do
# baseline: no heatmap (only class)
model="model-$dataset-OnlyClass.pth"
python3 test.py $model $dataset

# baseline: captum
for captum in $CAPTUM; do
python3 test.py $model $dataset --captum $captum
done

# baseline: protopnet
model="model-protopnet-$dataset.pth"
python3 test.py $model $dataset --protopnet

# proposal (with ablation)
for penalty in $PENALTIES; do
    for heatmap in $HEATMAPS; do
        model="model-$dataset-Heatmap-l1-$penalty-sigmoid.pth"
        python3 test.py $model $dataset
        for objdet in $MODELS; do
            model="model-$dataset-$objdet-l1-$penalty-heatmap-$heatmap-sigmoid.pth"
            python3 test.py $model $dataset
        done
    done
done
done