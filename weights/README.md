# Fine-tuned SAM3 LoRA Weights

本目录存放基于 SAM3 微调的 LoRA 适配器权重，用于云边协同水尺水位智能测量系统的云端分割推理阶段。

## 权重文件说明

| 文件 | 大小 | 用途 | 训练数据集 |
|------|------|------|-----------|
| `SAM3-LoRA-水尺主体_best.pt` | 17 MB | 水尺主体分割（检测水尺边界，提取 ROI 区域） | water_gauge.v3i（1257张实地水尺图像，COCO格式，2类：water_gauge / reflection） |
| `SAM3-LoRA-水位线V3clean_epoch05.pt` | 3.8 MB | 水位线精细定位（区分水面上下区域，输出水位线坐标） | WaterLine.v1i（483张精标图像，3类：Gauge_Air / Gauge_Water / Background） |

## 基础模型

- **来源**：[SAM 3 (Segment Anything Model 3)](https://github.com/facebookresearch/sam2) — Meta FAIR
- **版本**：SAM3（即 SAM 2.1，`sam3.pt` 检查点）
- **微调方法**：LoRA（Low-Rank Adaptation），秩 r=4，仅微调 image encoder 的 attention 层

> 注：基础模型 `sam3.pt`（3.3 GB）未包含在本仓库中，请自行从 Meta 官方渠道下载后放置于 `cloud_server/checkpoints/sam3.pt`。

## 加载方式

```python
import torch
from sam3_lora import LoRASAM3  # see SAM3_LoRA-main/

# 水尺主体分割
model = LoRASAM3(base_checkpoint="checkpoints/sam3.pt", lora_rank=4)
model.load_lora_weights("weights/SAM3-LoRA-水尺主体_best.pt")

# 水位线精细定位
model_wl = LoRASAM3(base_checkpoint="checkpoints/sam3.pt", lora_rank=4)
model_wl.load_lora_weights("weights/SAM3-LoRA-水位线V3clean_epoch05.pt")
```

详细推理流程见 [`SAM3_LoRA-main/inference_lora.py`](../SAM3_LoRA-main/inference_lora.py) 和 [`cloud_server/local_server.py`](../cloud_server/local_server.py)。

## 训练环境

- GPU：NVIDIA RTX 4090（AutoDL 云端）
- Framework：PyTorch 2.x
- 训练脚本：`SAM3_LoRA-main/train_sam3_lora_native.py`

## 数据来源

实地采集于国内灌区/水利站点水尺图像，由河海大学物联网工程研究团队标注整理。

## 引用 / Citation

如使用本权重，请注明：

```
基于视觉基础模型的云边协同水尺水位智能视觉测量系统
河海大学信息科学与工程学院 李家乐等
2026年全国大学生物联网设计竞赛参赛作品
https://github.com/stars-spark/water-gauge-sam3
```
