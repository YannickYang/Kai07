# Kai07

在 [χ₀ (Kai0)](https://github.com/OpenDriveLab/kai0) 架构基础上，实现 **π₀.7 (PI07)** 相关功能的扩展。

## 相对于 Kai0 的核心改进

Kai0 基于 π₀.5 构建了一套资源高效的机器人操作框架（Model Arithmetic + Stage Advantage + Train-Deploy Alignment）。Kai07 在此基础上引入 π₀.7 架构级别的增强：

| 特性 | Kai0 (π₀.5) | Kai07 (π₀.7) |
|------|-------------|--------------|
| 图像分辨率 | 224×224 | **448×448** |
| Action Expert | Gemma 300M | **Gemma 860M** |
| 状态输入 | 离散 token 化 | **线性投影** (continuous) |
| 历史帧编码 | 无 | **MEM History Encoder** (6帧, 30帧步长) |
| 注意力机制 | 标准 attention | **Block-Causal Attention** |
| 子目标图像 | 不支持 | **BAGEL 世界模型生成 subgoal** |
| 推理增强 | 无 | **Classifier-Free Guidance (CFG, β=1.7)** |
| 推理架构 | 单线程同步 | **三线程异步** (VLA + World Model + HLP) |
| 训练鲁棒性 | 无 | **观测延迟模拟** (max 12帧) |
| VLM 能力保持 | 无 | **Knowledge Insulation** (梯度解耦) |

### 关键模块说明

**1. MEM History Encoder**
- 将多帧多视角历史观测压缩为紧凑的 token 序列
- 每帧通过 SigLIP 编码后，空间压缩 1024→64 tokens/view
- 训练时以 30% 概率随机丢弃整个历史（数据增强）

**2. Block-Causal Attention**
- 6 个语义块：`history_obs → current_obs → subgoal → text → state → action`
- 块级下三角注意力（每个块可关注自身及前面所有块）
- text 块内部使用 causal attention，其余块内部双向

**3. BAGEL 世界模型**
- 基于 ByteDance BAGEL-7B-MoT 进行微调
- 输入当前观测 + 子任务文本，生成子目标图像
- 训练时离线预计算，推理时异步生成

**4. 三线程异步推理 (Pi07Policy)**
- Thread 1 (VLA): 高频动作块生成
- Thread 2 (World Model): 异步生成子目标图像 (每 4s 刷新)
- Thread 3 (High-Level Policy): 异步生成子任务文本 (每 8s 刷新)

**5. Knowledge Insulation (KI)**
- 训练时将 action loss 和 VLM loss 分离
- Action loss: 前缀 detach，不影响 VLM 参数
- VLM loss: 语言建模损失保持 VLM 能力不退化

## 项目结构

```
Kai07/
├── kai07/                              # 主代码目录
│   ├── src/openpi/
│   │   ├── models/pi0_config.py        # Pi07Config 配置定义
│   │   ├── models_pytorch/
│   │   │   ├── pi07_pytorch.py         # π₀.7 核心模型实现
│   │   │   ├── history_encoder.py      # MEM 历史编码器
│   │   │   └── high_level_policy.py    # 高层策略模型
│   │   ├── policies/
│   │   │   └── pi07_policy.py          # 三线程异步推理策略
│   │   ├── world_model/
│   │   │   └── bagel_wrapper.py        # BAGEL 世界模型封装
│   │   └── training/
│   │       └── config.py               # 训练配置 (含 pi07_fold_clothes)
│   ├── scripts/
│   │   ├── train_pytorch.py            # PyTorch 训练脚本
│   │   ├── finetune_bagel_world_model.py  # BAGEL 微调
│   │   ├── precompute_subgoals.py      # 子目标图像预计算
│   │   └── compute_norm_states_fast.py # 归一化统计量计算
│   ├── model_arithmetic/               # 检查点混合 (继承自 Kai0)
│   ├── stage_advantage/                # 阶段优势估计 (继承自 Kai0)
│   ├── train_deploy_alignment/         # 训练-部署对齐 (继承自 Kai0)
│   └── start_train.sh                  # 一键训练编排脚本
├── Bagel/                              # BAGEL 世界模型源码
├── lerobot/                            # LeRobot 数据集工具
└── ABPolicy-code/                      # 额外策略实现
```

## 安装

```bash
git clone --recurse-submodules <repo_url>
cd Kai07/kai07

# 使用 uv 安装依赖
GIT_LFS_SKIP_SMUDGE=1 uv sync
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .
uv pip install safetensors
```

## 使用方法

### 快速开始：完整训练流程

编辑 `kai07/start_train.sh`，设置三个路径：

```bash
DATA_DIR="path/to/dataset"          # LeRobot 格式数据集
WEIGHTS_DIR="path/to/pi05_base"     # π₀.5 预训练权重
BAGEL_PATH="path/to/BAGEL-7B-MoT"  # BAGEL 模型权重
```

**Step 1: 微调 BAGEL 世界模型**

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --standalone --nnodes=1 --nproc_per_node=8 \
    scripts/finetune_bagel_world_model.py \
    --model_path $BAGEL_PATH \
    --data_dir $DATA_DIR \
    --subtask_annotations $DATA_DIR/subtask_annotations.json \
    --output_dir checkpoints/bagel_wm \
    --total_steps 80000 \
    --lr 5e-5 --freeze_vit
```

**Step 2: 预计算子目标图像**

```bash
python3 scripts/precompute_subgoals.py \
    --data-dir $DATA_DIR \
    --bagel-path checkpoints/bagel_wm/final
```

**Step 3: 计算归一化统计量**

```bash
python3 scripts/compute_norm_states_fast.py --config-name pi07_fold_clothes \
    --base-dir $DATA_DIR
```

**Step 4: 训练 VLA (π₀.7)**

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --standalone --nnodes=1 --nproc_per_node=8 \
    scripts/train_pytorch.py pi07_fold_clothes \
    --exp_name fold_v1 \
    --num_train_steps 100000 \
    --keep_period 10000 \
    --batch_size 32 \
    --num_workers 8
```

### 推理部署

```python
from openpi.policies.pi07_policy import Pi07Policy
from openpi.world_model.bagel_wrapper import BagelWorldModel

# 加载模型
policy = Pi07Policy(
    vla_model=vla_model,
    world_model=BagelWorldModel.from_pretrained("path/to/bagel"),
    high_level_policy=hlp_model,  # 可选
    cfg_beta=1.7,
    num_denoising_steps=10,
)

# 异步推理
policy.start(task_instruction="Fold the cloth")
while not done:
    obs = get_observation()
    action = policy.step(obs)
    robot.execute(action)
policy.stop()
```

### 继续使用 Kai0 模块

Kai07 完全兼容 Kai0 的三大模块，详见 `kai07/README.md`：
- **Model Arithmetic**: 多检查点权重混合 → `model_arithmetic/`
- **Stage Advantage**: 阶段优势加权行为克隆 → `stage_advantage/`
- **Train-Deploy Alignment**: 数据增强/DAgger/时序平滑 → `train_deploy_alignment/`

## 计算需求

| 模式 | 显存 | 参考 GPU |
|------|------|----------|
| 推理 | >8 GB | RTX 4090 |
| 微调 (LoRA) | >22.5 GB | RTX 4090 |
| 微调 (Full) | >70 GB | A100 80GB / H100 |
| BAGEL 世界模型 | ~40 GB | A100 40GB+ |

## 致谢

- [openpi](https://github.com/Physical-Intelligence/openpi) — Physical Intelligence 的基座模型和训练框架
- [χ₀ (Kai0)](https://github.com/OpenDriveLab/kai0) — 分布一致性调优的机器人操作框架
- [BAGEL](https://github.com/ByteDance-Seed/BAGEL) — ByteDance 的多模态生成模型

## 引用

```bibtex
@article{sima2026kai0,
  title={$\chi_{0}$: Resource-Aware Robust Manipulation via Taming Distributional Inconsistencies},
  author={Yu, Checheng and Sima, Chonghao and Jiang, Gangcheng and Zhang, Hai and Mai, Haoguang and Li, Hongyang and Wang, Huijie and Chen, Jin and Wu, Kaiyang and Chen, Li and Zhao, Lirui and Shi, Modi and Luo, Ping and Bu, Qingwen and Peng, Shijia and Li, Tianyu and Yuan, Yibo},
  journal={arXiv preprint arXiv:2602.09021},
  year={2026}
}
```
