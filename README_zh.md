# PixelPilot

**Scalable Vision-Language-Action Models for End-to-End Autonomous Driving (ECCV 2026)**

[论文](https://arxiv.org/abs/2607.04637) · [项目主页](https://pixelpilotvla.github.io/) · [English](README.md)

PixelPilot 将图像空间规划与几何提升解耦：模型在图像平面内预测感知结果、驾驶动作和未来
轨迹点，推理时再根据相机标定将轨迹提升到米制空间。训练结合多任务监督微调、筛选后的
视觉推理数据和 GRPO。

本仓库提供 **nuScenes / Qwen2.5-VL-7B** 实现，监督训练基于 **LLaMA-Factory**，
强化学习基于所提供的 **Seg-Zero / EasyR1 / veRL** 代码。当前 GRPO 配置保留了历史
奖励实现；模型权重的可用情况及与论文的差异见[复现说明](docs/reproduction.md)。

## 🛠️ 安装

GPU 训练使用 Linux 和 Python 3.10。请为 SFT 和 GRPO 分别建立环境，并分别安装
与硬件匹配的 PyTorch/CUDA。以下命令均在仓库根目录执行。

```bash
# 数据准备环境
python -m pip install -r requirements/data.txt
python -m pip install --no-deps -e .

# SFT 环境
python -m pip install -r requirements/sft.txt
python -m pip install --no-deps -e ./LLaMAFactory
python -m pip install --no-deps -e .

# GRPO / 推理环境
python -m pip install -r requirements/train.txt
python -m pip install flash-attn==2.7.4.post1 --no-build-isolation
python -m pip install --no-deps -e .
```

下载 [Qwen2.5-VL-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct)
作为基础模型。环境建立、依赖版本和可选评测工具见[安装说明](docs/installation.md)。

## 📂 数据准备

准备 [nuScenes](https://www.nuscenes.org/nuscenes) 相机图像、标定表和地图，以及
**OmniDrive 格式的规划 info 文件**和**原始米制对话 JSON**：

```text
data/
├── nuscenes/                  # 相机图像、元数据和地图
├── infos/{train,val}.pkl      # 规划 info 文件
└── raw/nuscenes_{train,val}.json
```

```bash
python -m pixelpilot.generate \
  --train-json data/raw/nuscenes_train.json --train-infos data/infos/train.pkl \
  --val-json data/raw/nuscenes_val.json --val-infos data/infos/val.pkl \
  --nusc-root data/nuscenes --output-dir data/annotations --seed 42

python -m pixelpilot.collage --annotations data/annotations/train_integrated.json \
  --nusc-root data/nuscenes --output-dir data/nusc_fb_train --native-resolution
python -m pixelpilot.collage --annotations data/annotations/val_integrated.json \
  --nusc-root data/nuscenes --output-dir data/nusc_fb_val --native-resolution
```

输出包括多任务监督数据、联合任务、独立米制真值和前后视图拼接图。上游从原始 nuScenes
生成 info 文件的转换器不在本仓库中。输入字段、坐标约定和输出格式见[数据说明](docs/data.md)。

## ⚙️ 训练

### 阶段一：多任务 SFT

转换标注，然后使用仓库内附的 LLaMA-Factory 训练 7B 模型：

```bash
python -m pixelpilot.prepare --format sft \
  --train-json data/annotations/train_sft.json \
  --val-json data/annotations/val_integrated.json \
  --image-root data --output-dir data/nuscenes_sft

bash scripts/train_sft.sh --num-gpus 8
```

默认 [SFT 配置](configs/sft_7b.yaml) 使用全量微调、ZeRO-3、2 个 epoch，学习率为
`5e-6`。可以通过 `--model`、`--dataset-dir` 和 `--output-dir` 修改模型、数据和输出路径。

### 阶段二：视觉推理

针对训练集失败样本生成教师解释，执行有、无解释的配对推理，筛选出能够纠正动作且降低
米制轨迹 L2 的解释。通过 `pixelpilot.prepare --format sft --extra-train-json ...`
合并筛选结果，再继续 SFT。完整流程见[推理数据说明](docs/reasoning.md)。

### 阶段三：GRPO

切换到 GRPO 环境，准备联合任务数据并训练：

```bash
python -m pixelpilot.prepare \
  --train-json data/annotations/train_integrated.json \
  --image-root data --output-dir data/nuscenes_grpo

export MODEL_PATH="$PWD/checkpoints/pixelpilot-7b-sft"
export NUSCENES_ROOT="$PWD/data/nuscenes"
bash scripts/train_grpo.sh
```

默认 [GRPO 配置](configs/grpo_7b.yaml) 使用 4 张 GPU，每个提示生成 8 个候选。
配置细节、启动预览和检查点处理见[训练说明](docs/training.md)。

## 🔑 推理与评测

使用本地全量 SFT 权重或已合并的 GRPO 权重生成预测，再将图像空间轨迹提升到米制空间：

```bash
python -m pixelpilot.infer --model checkpoints/pixelpilot-7b-sft \
  --annotations data/annotations/val_integrated.json --image-root data \
  --output outputs/predictions.jsonl

python -m pixelpilot.lift --predictions outputs/predictions.jsonl \
  --nusc-root data/nuscenes --ground-z -1.0 --frame PLANNING \
  --output outputs/trajectories.json
```

检查点合并和文本诊断见[推理说明](docs/inference.md)，L2、碰撞及边界指标见
[评测说明](docs/evaluation.md)。本地已验证范围为 CPU 检查，GPU 训练和真实地图评测
仍未验证，具体记录见[验证说明](docs/validation.md)。

## 🗂️ 仓库结构

```text
PixelPilot/
├── LLaMAFactory/       # 内附 SFT 框架，内部版本 0.9.4.dev0
├── pixelpilot/         # 数据、视觉推理、模型推理、轨迹提升与评测
├── verl/               # GRPO 框架及奖励实现
├── configs/            # SFT / GRPO 配置和历史实验配置
├── scripts/            # 训练启动、检查点合并和源码导出
├── requirements/       # 各环境的依赖
├── examples/           # 小型标注和预测示例
├── tests/              # CPU 回归测试
└── docs/               # 使用文档、复现说明和验证记录
```

所有指南见[文档索引](docs/README.md)，本地检查和源码打包见[开发说明](docs/development.md)。

## 📝 引用

```bibtex
@article{tang2026pixelpilot,
  title={PixelPilot: Scalable Vision-Language-Action Models for End-to-End Autonomous Driving},
  author={Tang, Pin and Wang, Guoqing and Ren, Xiangxuan and Wang, Zhongdao and
          Zhao, Guodongfang and Feng, Bailan and Ma, Chao},
  journal={arXiv preprint arXiv:2607.04637},
  year={2026}
}
```

## 🙏 致谢

本实现基于 [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory)、
[Seg-Zero](https://github.com/dvlab-research/Seg-Zero)、
[EasyR1](https://github.com/hiyouga/EasyR1)、
[veRL](https://github.com/volcengine/verl)、
[Qwen2.5-VL](https://github.com/QwenLM/Qwen2.5-VL)、
[OmniDrive](https://github.com/NVlabs/OmniDrive) 和
[nuScenes](https://www.nuscenes.org/nuscenes)。

代码采用 [Apache-2.0](LICENSE) 许可证。第三方声明见 [NOTICE](NOTICE) 和
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。数据集和预训练模型遵循其各自条款。
