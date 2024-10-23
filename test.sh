#!/bin/bash

#DATASETS="Birds StanfordCars StanfordDogs"
DATASETS=$1
#PENALTIES="0 0.001 0.1 1 10"
PENALTIES="0 0.001 0.01 0.1 1"
XAI="CAM GradCAM DeepLIFT Occlusion"
MODELS="SimpleDet FasterRCNN FCOS DETR"
HEATMAPS="GaussHeatmap LogisticHeatmap"
OCCLUSIONS="encoder"
#OCCLUSIONS="encoder image"

echo "model,dataset,xai,crop,acc,pg,degscore,sparsity"

for dataset in $DATASETS; do
# baseline: no heatmap (only class)
model="model-$dataset-OnlyClass.pth"
python test.py $model $dataset

# baseline: xai
for xai in $XAI; do
python test.py $model $dataset --xai $xai
done

# baseline: protopnet
model="model-$dataset-ProtoPNet.pth"
python test.py $model $dataset --protopnet
python test.py $model $dataset --protopnet --crop

# baseline: protopnet crop
model="model-$dataset-ProtoPNet-crop.pth"
python test.py $model $dataset --protopnet
python test.py $model $dataset --protopnet --crop

# proposal (with ablation)
for occlusion in $OCCLUSIONS; do
    for penalty in $PENALTIES; do
        model="model-$dataset-Heatmap-l1-$penalty-heatmap-none-occlusion-$occlusion-sigmoid.pth"
        python test.py $model $dataset
        model="model-$dataset-Heatmap-l1-$penalty-heatmap-none-occlusion-$occlusion-sigmoid-adversarial.pth"
        python test.py $model $dataset
        for objdet in $MODELS; do
            for heatmap in $HEATMAPS; do
                model="model-$dataset-$objdet-l1-$penalty-heatmap-$heatmap-occlusion-$occlusion-sigmoid.pth"
                python test.py $model $dataset
                model="model-$dataset-$objdet-l1-$penalty-heatmap-$heatmap-occlusion-$occlusion-sigmoid-adversarial.pth"
                python test.py $model $dataset
            done
        done
    done
done
done  # $dataset
