# Kai0.7 Training Guide

## 环境配置

### 系统要求

- Linux (Ubuntu 20.04+)
- NVIDIA GPU: VLA 训练最少 1×A100 40GB，BAGEL 微调需 2+×A100 80GB
- CUDA 12.0+
- Conda

### 安装

```bash
conda create -n mini_lerobot python=3.10 -y
conda activate mini_lerobot

# PyTorch (按自己的 CUDA 版本调整)
pip install torch==2.9.1 torchvision==0.24.1 --index-url https://download.pytorch.org/whl/cu128

# 核心依赖
pip install transformers==4.51.3 accelerate==1.10.0 safetensors==0.7.0
pip install einops==0.8.1 scipy pillow tqdm wandb tyro sentencepiece
pip install av huggingface-hub

# LeRobot
pip install git+https://github.com/huggingface/lerobot.git@0cf864870cf29f4738d3ade893e6fd13fbd7cdb5

# 本项目
cd kai07 && pip install -e .

# BAGEL 微调额外依赖
pip install flash-attn --no-build-isolation
git clone https://github.com/ByteDance-Seed/Bagel.git /path/to/Bagel
```

### 下载权重

```bash
export HF_ENDPOINT=https://hf-mirror.com  # 国内镜像加速

# pi0.5 base (~5GB)
huggingface-cli download lerobot/pi05_base --local-dir /path/to/weights/pi05_base

# BAGEL-7B-MoT (~30GB)
huggingface-cli download ByteDance-Seed/BAGEL-7B-MoT --local-dir /path/to/weights/BAGEL-7B-MoT
```

---

## 训练流程

修改 `start_train.sh` 顶部 3 个路径：

```bash
DATA_DIR="数据集路径"
WEIGHTS_DIR="pi0.5 权重路径"
BAGEL_PATH="BAGEL-7B-MoT 权重路径"
```

然后按顺序执行：

```bash
# Step 1: 微调 BAGEL 世界模型
bash start_train.sh bagel

# Step 2: 用微调后的 BAGEL 生成 subgoal 图像
BAGEL_PATH=checkpoints/bagel_wm/final bash start_train.sh subgoals-bagel

# Step 3: 训练 VLA
bash start_train.sh
```

断点续训任意步骤加 `RESUME=1`。

---

## 可修改参数

### VLA 训练 (环境变量)

```bash
TOTAL_STEPS=100000 KEEP_PERIOD=10000 BATCH_SIZE=32 NUM_WORKERS=8 bash start_train.sh
```

| 变量 | 默认 | 说明 |
|------|------|------|
| `TOTAL_STEPS` | 100000 | 总训练步数 |
| `KEEP_PERIOD` | 10000 | checkpoint 保留间隔 |
| `BATCH_SIZE` | 32 | 全局 batch size |
| `NUM_WORKERS` | 8 | dataloader workers |
| `NUM_GPUS` | auto | GPU 数量 |
| `EXP_NAME` | fold_v1 | 实验名 |

### BAGEL 微调 (命令行参数)

`start_train.sh bagel` 默认参数即可。需要覆盖时直接调脚本：

```bash
BAGEL_DIR=/path/to/Bagel torchrun --nproc_per_node=4 \
    scripts/finetune_bagel_world_model.py \
    --model_path $BAGEL_PATH \
    --data_dir $DATA_DIR \
    --output_dir checkpoints/bagel_wm \
    --total_steps 5000 --save_every 500 --lr 5e-5
```

| 参数 | 默认 | 说明 |
|------|------|------|
| `--total_steps` | 5000 | 总步数 |
| `--save_every` | 500 | checkpoint 保存间隔 |
| `--lr` | 5e-5 | 学习率 |
| `--warmup_steps` | 200 | warmup 步数 |
| `--gradient_accumulation_steps` | 4 | 梯度累积 |
| `--ema_decay` | 0.9999 | EMA 衰减率 |
| `--freeze_vit` / `--no_freeze_vit` | freeze | 是否冻结 ViT |
| `--freeze_llm` | False | 是否冻结 LLM |
| `--wandb` | off | 开启 WandB 日志 |

---

## 所有命令

| 命令 | 说明 |
|------|------|
| `bash start_train.sh` | 训练 VLA |
| `bash start_train.sh bagel` | 微调 BAGEL 世界模型 |
| `bash start_train.sh subgoals-bagel` | 用 BAGEL 生成 subgoal |
| `bash start_train.sh hlp` | 训练高层策略 |
| `bash start_train.sh norm` | 计算归一化统计量 |
| `bash start_train.sh check` | 检查环境 |
