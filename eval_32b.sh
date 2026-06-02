#!/bin/bash

export CUDA_VISIBLE_DEVICES=0,1,2,3
export MODEL_PATH="/root/autodl-tmp/models/Qwen3-VL-32B-Instruct"

echo "Running UI-Vision Basic..."
torchrun --nproc_per_node=4 rp_qwen3vl_wZoomin.py \
    --json_file_dir "/data/jiapinglin/datasets/agent/ui-vision/annotations/element_grounding/element_grounding_basic.json" \
    --model_path "$MODEL_PATH" \
    --benchmark "uivision" \
    --continuity_layer 6

echo "Running UI-Vision Functional..."
torchrun --nproc_per_node=4 rp_qwen3vl_wZoomin.py \
    --json_file_dir "/data/jiapinglin/datasets/agent/ui-vision/annotations/element_grounding//element_grounding_functional.json" \
    --model_path "$MODEL_PATH" \
    --benchmark "uivision" \
    --continuity_layer 6

echo "Running UI-Vision Spatial..."
torchrun --nproc_per_node=4 rp_qwen3vl_wZoomin.py \
    --json_file_dir "/data/jiapinglin/datasets/agent/ui-vision/annotations/element_grounding/element_grounding_spatial.json" \
    --model_path "$MODEL_PATH" \
    --benchmark "uivision" \
    --continuity_layer 6

# ======================================================================================================================
# ======================================================================================================================

echo "Running ScreenSpot-Pro..."
torchrun --nproc_per_node=4 rp_qwen3vl_wZoomin.py \
    --model_path "$MODEL_PATH" \
    --benchmark "sspro" \
    --continuity_layer 6

# ======================================================================================================================
# ======================================================================================================================

echo "Running ScreenSpot-V2..."
torchrun --nproc_per_node=4 rp_qwen3vl.py \
    --model_path "$MODEL_PATH" \
    --benchmark "ssv2" \
    --continuity_layer 6

# ======================================================================================================================
# ======================================================================================================================

echo "Running OSWorld-G..."
torchrun --nproc_per_node=4 rp_qwen3vl_wZoomin.py \
    --json_file_dir "/data/jiapinglin/datasets/agent/OSWorld-G/benchmark/classification_result.json" \
    --model_path "$MODEL_PATH" \
    --benchmark "osworld" \
    --continuity_layer 4

torchrun --nproc_per_node=4 rp_qwen3vl_wZoomin.py \
    --json_file_dir "/data/jiapinglin/datasets/agent/OSWorld-G/benchmark/OSWorld-G_refined_classified.json" \
    --model_path "$MODEL_PATH" \
    --benchmark "osworld" \
    --continuity_layer 4

# ======================================================================================================================
# ======================================================================================================================

echo "Running MMBench-GUI..."
torchrun --nproc_per_node=4 rp_qwen3vl_wZoomin.py \
    --model_path "$MODEL_PATH" \
    --benchmark "mmbench" \
    --continuity_layer 4

echo "Done"