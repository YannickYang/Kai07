#!/bin/bash
set -e
cd "$(dirname "$0")"
export PYTHONPATH="$(pwd)/src:${PYTHONPATH:-}"

# ====================== 公共路径 ======================
DATA_DIR="YOUR_DATA_DIR"
WEIGHTS_DIR="YOUR_WEIGHTS_DIR"
BAGEL_PATH="YOUR_WM_PATH"
export BAGEL_DIR="WM_PRETRAIN_MODEL_PATH"

#############################################################

# # Step 1: 微调 BAGEL 世界模型
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --standalone --nnodes=1 --nproc_per_node=8 \
    scripts/finetune_bagel_world_model.py \
    --model_path $BAGEL_PATH \
    --data_dir $DATA_DIR \
    --subtask_annotations $DATA_DIR/subtask_annotations.json \
    --output_dir checkpoints/bagel_wm \
    --total_steps 80000 \
    --save_every 2000 \
    --lr 5e-5 \
    --warmup_steps 200 \
    --gradient_accumulation_steps 4 \
    --freeze_vit

#############################################################

# # Step 2: 用微调后的 BAGEL 生成 subgoal 图像
# CUDA_VISIBLE_DEVICES=0 python3 scripts/precompute_subgoals.py \
#     --data-dir $DATA_DIR \
#     --bagel-path checkpoints/bagel_wm/final

#############################################################

# # Step 3: 计算归一化统计量 (VLA 训练前必须)
# python3 scripts/compute_norm_states_fast.py --config-name pi07_fold_clothes \
#     --base-dir $DATA_DIR

#############################################################

# # Step 4: VLA 训练 (需先完成 Step 1 & 2 & 3)
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --standalone --nnodes=1 --nproc_per_node=8 \
#     scripts/train_pytorch.py pi07_fold_clothes \
#     --exp_name fold_v1 \
#     --num_train_steps 100000 \
#     --keep_period 10000 \
#     --batch_size 32 \
#     --num_workers 8

#############################################################

# # 其他常用命令 (按需取消注释)
#
# # 检查环境
# # bash start_train.sh check
#
# # 计算归一化统计量
# # python3 scripts/compute_norm_stats.py --config-name pi07_fold_clothes
#
# # 断点续训: 在对应命令末尾加 --auto_resume (bagel) 或 --resume (VLA)
