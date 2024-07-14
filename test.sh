#!/bin/bash

DATASETS="Birds StanfordCars StanfordDogs"
DATASETS="StanfordCars"
for dataset in $DATASETS; do
for model in model-$dataset-*.pth; do
python3 test.py $model $dataset --visualize
done
done
