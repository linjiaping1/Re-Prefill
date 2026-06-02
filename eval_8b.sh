#!/bin/bash

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export MODEL_PATH="/data/jiapinglin/models/Qwen3-VL-8B-Instruct"
#export MODEL_PATH="/data/jiapinglin/models/GUI-Owl-1.5-8B"
#export MODEL_PATH="/data/jiapinglin/models/MAI-UI-8B"


echo "Running UI-Vision Basic..."
torchrun --nproc_per_node=8 rp_qwen3vl_wZoomin.py \
    --json_file_dir "/data/jiapinglin/datasets/agent/ui-vision/annotations/element_grounding/element_grounding_basic.json" \
    --model_path "$MODEL_PATH" \
    --benchmark "uivision"

echo "Running UI-Vision Functional..."
torchrun --nproc_per_node=8 rp_qwen3vl_wZoomin.py \
    --json_file_dir "/data/jiapinglin/datasets/agent/ui-vision/annotations/element_grounding//element_grounding_functional.json" \
    --model_path "$MODEL_PATH" \
    --benchmark "uivision"

echo "Running UI-Vision Spatial..."
torchrun --nproc_per_node=8 rp_qwen3vl_wZoomin.py \
    --json_file_dir "/data/jiapinglin/datasets/agent/ui-vision/annotations/element_grounding/element_grounding_spatial.json" \
    --model_path "$MODEL_PATH" \
    --benchmark "uivision"

# ======================================================================================================================
# ======================================================================================================================

echo "Running ScreenSpot-Pro..."
torchrun --nproc_per_node=8 rp_qwen3vl_wZoomin.py \
    --model_path "$MODEL_PATH" \
    --benchmark "sspro"

# ======================================================================================================================
# ======================================================================================================================

echo "Running ScreenSpot-V2..."
torchrun --nproc_per_node=8 rp_qwen3vl.py \
    --model_path "$MODEL_PATH" \
    --benchmark "ssv2"

# ======================================================================================================================
# ======================================================================================================================

echo "Running OSWorld-G..."
torchrun --nproc_per_node=8 rp_qwen3vl_wZoomin.py \
    --json_file_dir "/data/jiapinglin/datasets/agent/OSWorld-G/benchmark/classification_result.json" \
    --model_path "$MODEL_PATH" \
    --benchmark "osworld"

torchrun --nproc_per_node=8 rp_qwen3vl_wZoomin.py \
    --json_file_dir "/data/jiapinglin/datasets/agent/OSWorld-G/benchmark/OSWorld-G_refined_classified.json" \
    --model_path "$MODEL_PATH" \
    --benchmark "osworld"

# ======================================================================================================================
# ======================================================================================================================

echo "Running MMBench-GUI..."
torchrun --nproc_per_node=8 rp_qwen3vl_wZoomin.py \
    --model_path "$MODEL_PATH" \
    --benchmark "mmbench"

echo "Done"