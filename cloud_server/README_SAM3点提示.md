# SAM3 点提示 / 框提示分割（给队友的对比实验包）

> 一句话结论：**本机已有的那个 `sam3.pt` 就能跑点提示，不用再下任何权重、不用联网。**
> 点提示和概念分割(文本)打包在同一个 ckpt 里，走不同的子模块而已。

## 为什么之前以为"点提示没开源"

两个常见误会，都不成立：

1. **去 HuggingFace 权重仓库 `facebook/sam3` 找 `Sam3TrackerModel`** —— 找不到是正常的。
   那是 `transformers` 库里的 **Python 类名**，不是权重文件。权重仓库里只有 `sam3.pt`，
   而这一个文件**同时含 detector(概念分割) + tracker(点提示) 两套组件**（848M 参数，官方说明）。
2. **transformers 版本太旧** → `from transformers import Sam3TrackerModel` 直接 ImportError，
   看起来像"没这东西"。（但我们这个包**不依赖 transformers**，走的是 facebookresearch/sam3 原生代码。）

实测：把 `sam3.pt` 权重键拆开，`tracker.sam_prompt_encoder.point_embeddings.*`、
`tracker.sam_mask_decoder.*` 全都在文件里 —— 点提示的编码器和解码器，本地齐活。

## 原理（和 SAM2 的 image predictor 同款 API）

```
build_sam3_image_model(enable_inst_interactivity=True)
        │  ← 激活内置 SAM3InteractiveImagePredictor，并把 ckpt 的 tracker.* 权重灌进去
        │  ← 点提示器【共享 detector 的 vision backbone】(ckpt 里 tracker 没有独立 backbone)
        ▼
Sam3Processor.set_image(img)        # 跑一次 detector backbone，产出 sam2_backbone_out
        ▼
model.predict_inst(state, point_coords=..., point_labels=..., box=...)
        ▼                            # 走 tracker.sam_prompt_encoder + sam_mask_decoder
   masks[C,H,W], ious[C]            # C=3 个候选，按 IoU 选最优
```

## 环境要求

- 一套能 `import sam3` 的环境（官方 facebookresearch/sam3 包；你现有的 SAM3_LoRA 环境就行）
- `sam3.pt`（本机已有，见下方路径）
- BPE 词表 `sam3/assets/bpe_simple_vocab_16e6.txt.gz`（包自带）
- torch + CUDA（CPU 也能跑，加 `--device cpu`，慢一些）

本机默认路径（脚本里已写死，拷到别处时改 `--ckpt` / `--bpe` 即可）：
- ckpt：`03_云端SAM3/cloud_server/checkpoints/sam3.pt`
- bpe ：`03_云端SAM3/cloud_server/sam3/assets/bpe_simple_vocab_16e6.txt.gz`

## 用法

```bash
# 单点（前景）
python sam3_point_prompt.py --image test_image.jpg --points "400,300"

# 多点：前景1 / 背景0
python sam3_point_prompt.py --image test_image.jpg \
    --points "300,400;320,500" --labels "1,0"

# 框提示（XYXY 像素）
python sam3_point_prompt.py --image test_image.jpg --box "330,300,690,560"

# 点+框联合，指定输出与设备
python sam3_point_prompt.py --image a.jpg --box "100,50,400,760" \
    --points "250,400" --labels "1" --out result.png --device cuda
```

输出：原图 + 最优 mask(半透明绿) + 提示点(前景蓝/背景红)/框(红框) 的可视化 PNG；
终端打印 3 个候选 mask 的 IoU 分数和选用项。

### 当库用（写对比脚本时）

```python
from sam3_point_prompt import SAM3PointPrompter
pr = SAM3PointPrompter(device="cuda")          # 加载一次
masks, ious, best = pr.predict(                # 反复调
    "img.jpg", points=[[400, 300]], labels=[1])
best_mask = masks[best]                         # bool [H,W]
```

## 本机已验证（2026-06）

`test_image.jpg`(800×600，自行车靠木墙的通用测试图)：
- 框提示 `--box "330,300,690,560"` → 分出自行车，**IoU 0.836**，0.99s
- 中心单点 → 3 候选 IoU 0.095/0.416/0.469，正常出 mask

## ⚠️ 做 SAM2 vs SAM3 对比时务必注意

tracker 这条是**基础 SAM3 权重，没经过我们的 water_gauge / Waterline LoRA 微调**
（LoRA 是训在 detector 概念分支上的）。所以这里得到的是「通用 SAM3 点提示」效果——
**正好对标「通用 SAM2 点提示」，是公平的 apples-to-apples 几何分割对比。**

建议的两组对比：
1. **几何 vs 几何**：SAM2 点/框 ↔ 本脚本 SAM3 点/框（同图同点，比 IoU）。
2. **概念优势（更亮的卖点）**：SAM2 必须人工点/框 ↔ SAM3 文本一句话自动分出全部水尺实例
   （后者用 detector 概念分支，见 `sam3_infer.py`）。卖点是**零人工交互 + 自适应提示**。

## 跟我们项目主链路的关系

部署管线（`sam3_infer.py` / 云端服务）走的是 **detector 概念分支**（文本/框 + 你们的 LoRA）。
本脚本的点提示是**给对比实验用的支线**，不进生产链路，但跟部署同环境、同一个 `sam3.pt`，零额外依赖。
