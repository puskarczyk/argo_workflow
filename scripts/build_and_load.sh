#!/bin/bash
set -e

build_and_load() {
    local img=$1
    local file=$2
    
    echo "Building image: $img from $file"
    docker build -t "$img" -f "$file" .
    
    echo "Loading image: $img to Argo cluster"
    kind load docker-image "$img" --name argo-project
}

build_and_load "ml-validation:v1" "Dockerfile.validation"
build_and_load "ml-preprocessing:v1" "Dockerfile.preprocessing"
build_and_load "ml-imbalance:v1" "Dockerfile.imbalance"
build_and_load "ml-train-lr:v1" "Dockerfile.train_lr"
build_and_load "ml-train-xgb:v1" "Dockerfile.train_xgb"
build_and_load "ml-plot:v1" "Dockerfile.plot"

echo "All images loaded successfully"