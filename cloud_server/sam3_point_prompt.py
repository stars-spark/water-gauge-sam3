# -*- coding: utf-8 -*-
"""
SAM3 点提示 / 框提示分割（PVS，Promptable Visual Segmentation）独立 Demo
=====================================================================

给队友做「SAM2 vs SAM3 点提示对比」用。证明：本机已有的单个 `sam3.pt`
**无需再下任何权重**就能跑点提示——detector(概念分割) 和 tracker(点提示) 两套
组件打包在同一个 ckpt 里，点提示走 tracker 的 sam_prompt_encoder + sam_mask_decoder。

原理（和 SAM2 的 image predictor 同款 API）：
  1) build_sam3_image_model(enable_inst_interactivity=True) —— 激活内置的
     SAM3InteractiveImagePredictor，并把 ckpt 里 `tracker.*` 权重灌进去；
     点提示器**共享 detector 的 vision backbone**（ckpt 里 tracker 无独立 backbone）。
  2) Sam3Processor.set_image(img) —— 跑一次 detector backbone，产出 sam2_backbone_out。
  3) model.predict_inst(state, point_coords=..., point_labels=...) —— 吃点/框出 mask。

⚠️ 注意：tracker 这条是**基础 SAM 权重，没经过你们的 water_gauge / Waterline LoRA 微调**
（LoRA 训在 detector 概念分支上）。所以这里得到的是「通用 SAM3 点提示」效果，
正好用于和「通用 SAM2 点提示」做公平对比。

用法
----
单点（前景）：
    python sam3_point_prompt.py --image test_image.jpg --points "300,400"

多点（前景 1 / 背景 0）：
    python sam3_point_prompt.py --image test_image.jpg \
        --points "300,400;320,500" --labels "1,1"

框提示（XYXY 像素）：
    python sam3_point_prompt.py --image test_image.jpg --box "100,50,400,760"

点+框联合，并指定输出：
    python sam3_point_prompt.py --image a.jpg --box "100,50,400,760" \
        --points "250,400" --labels "1" --out result.png --device cuda

输出：原图 + 最优 mask 叠加 + 提示点/框可视化，存为 PNG；终端打印每个候选 mask 的 IoU 分数。
"""
import os
import sys
import argparse

import numpy as np
from PIL import Image

# 让脚本能找到同目录下的 sam3 包
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

# 某些环境 import sam3 前需要 stub 掉 decord（无视频依赖时）
try:
    import decord  # noqa: F401
except Exception:
    import types
    sys.modules["decord"] = types.ModuleType("decord")

import torch  # noqa: E402
from sam3.model_builder import build_sam3_image_model  # noqa: E402
from sam3.model.sam3_image_processor import Sam3Processor  # noqa: E402


# ----------------------------- 默认路径 -----------------------------
# 队友若把本脚本拷到自己的 sam3 安装目录，改这两个路径即可。
DEFAULT_CKPT = os.path.join(HERE, "checkpoints", "sam3.pt")
DEFAULT_BPE = os.path.join(HERE, "sam3", "assets", "bpe_simple_vocab_16e6.txt.gz")


class SAM3PointPrompter:
    """加载一次模型，可反复对不同图做点/框提示分割。"""

    def __init__(self, checkpoint_path=DEFAULT_CKPT, bpe_path=DEFAULT_BPE,
                 resolution=1008, device="cuda"):
        self.device = "cuda" if (device == "cuda" and torch.cuda.is_available()) else "cpu"
        print(f"[SAM3-PVS] 设备: {self.device}  ckpt: {checkpoint_path}")
        self.model = build_sam3_image_model(
            bpe_path=bpe_path,
            device=self.device,
            eval_mode=True,
            checkpoint_path=checkpoint_path,
            load_from_HF=False,
            enable_segmentation=True,
            enable_inst_interactivity=True,   # ← 关键：激活点提示器并加载 tracker.* 权重
        )
        assert self.model.inst_interactive_predictor is not None, \
            "点提示器未激活：enable_inst_interactivity 必须为 True"
        self.processor = Sam3Processor(self.model, resolution=resolution, device=self.device)
        print("[SAM3-PVS] ✅ 模型就绪（点提示走 tracker.sam_prompt_encoder + sam_mask_decoder）")

    @torch.inference_mode()
    def predict(self, image, points=None, labels=None, box=None, multimask_output=True):
        """对单图做点/框提示分割。

        参数：
          image:  PIL.Image 或路径
          points: Nx2 (x,y) 像素坐标，list 或 ndarray，可为 None
          labels: 长度 N，1=前景点 0=背景点；points 给了就必须给
          box:    [x1,y1,x2,y2] 像素，XYXY，可为 None
          multimask_output: True 时返回 3 个候选 mask（单点歧义时更稳）

        返回：(masks[C,H,W] bool, ious[C], best_idx)
        """
        if isinstance(image, str):
            image = Image.open(image)
        image = image.convert("RGB")

        state = self.processor.set_image(image)

        kwargs = {"multimask_output": multimask_output}
        if points is not None:
            pc = np.asarray(points, dtype=np.float32).reshape(-1, 2)
            if labels is None:
                labels = np.ones(len(pc), dtype=np.int32)
            kwargs["point_coords"] = pc
            kwargs["point_labels"] = np.asarray(labels, dtype=np.int32).reshape(-1)
        if box is not None:
            kwargs["box"] = np.asarray(box, dtype=np.float32).reshape(4)

        masks, ious, _ = self.model.predict_inst(state, **kwargs)
        masks = masks.astype(bool)
        ious = np.asarray(ious).reshape(-1)
        best_idx = int(np.argmax(ious))
        return masks, ious, best_idx


# ----------------------------- 可视化 -----------------------------

def overlay(image_pil, mask, points=None, labels=None, box=None):
    """把 mask（半透明绿）+ 提示点/框叠到原图上，返回 PIL.Image。仅用 numpy/PIL。"""
    from PIL import ImageDraw
    img = np.array(image_pil.convert("RGB")).astype(np.float32)
    color = np.array([0, 255, 0], dtype=np.float32)
    m = mask.astype(bool)
    img[m] = 0.5 * img[m] + 0.5 * color
    out = Image.fromarray(img.astype(np.uint8))

    draw = ImageDraw.Draw(out)
    if box is not None:
        x1, y1, x2, y2 = [float(v) for v in np.asarray(box).reshape(4)]
        draw.rectangle([x1, y1, x2, y2], outline=(255, 0, 0), width=3)
    if points is not None:
        pc = np.asarray(points, dtype=np.float32).reshape(-1, 2)
        lb = (np.asarray(labels).reshape(-1) if labels is not None
              else np.ones(len(pc), dtype=int))
        for (x, y), l in zip(pc, lb):
            c = (0, 128, 255) if l == 1 else (255, 0, 0)  # 前景蓝 / 背景红
            r = 6
            draw.ellipse([x - r, y - r, x + r, y + r], fill=c, outline=(255, 255, 255))
    return out


# ----------------------------- CLI -----------------------------

def _parse_points(s):
    if not s:
        return None
    pts = []
    for pair in s.split(";"):
        pair = pair.strip()
        if not pair:
            continue
        x, y = pair.split(",")
        pts.append([float(x), float(y)])
    return pts


def _parse_labels(s):
    if not s:
        return None
    return [int(v) for v in s.split(",") if v.strip() != ""]


def _parse_box(s):
    if not s:
        return None
    return [float(v) for v in s.split(",")]


def main():
    ap = argparse.ArgumentParser(description="SAM3 点/框提示分割 Demo（本机 sam3.pt，无需联网）")
    ap.add_argument("--image", required=True, help="输入图片路径")
    ap.add_argument("--points", default="", help='点坐标，如 "300,400;320,500"（x,y 用;分隔多点）')
    ap.add_argument("--labels", default="", help='点标签，如 "1,0"（1前景 0背景），不填默认全前景')
    ap.add_argument("--box", default="", help='框 XYXY，如 "100,50,400,760"')
    ap.add_argument("--out", default="sam3_point_result.png", help="输出可视化 PNG")
    ap.add_argument("--ckpt", default=DEFAULT_CKPT)
    ap.add_argument("--bpe", default=DEFAULT_BPE)
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--single-mask", action="store_true", help="只输出单 mask（不输出3候选）")
    args = ap.parse_args()

    points = _parse_points(args.points)
    labels = _parse_labels(args.labels)
    box = _parse_box(args.box)
    if points is None and box is None:
        ap.error("必须至少给 --points 或 --box 之一")

    import time
    pr = SAM3PointPrompter(checkpoint_path=args.ckpt, bpe_path=args.bpe, device=args.device)

    img = Image.open(args.image)
    t0 = time.time()
    masks, ious, best = pr.predict(
        img, points=points, labels=labels, box=box,
        multimask_output=not args.single_mask)
    dt = time.time() - t0

    print(f"[结果] 候选 mask 数: {len(ious)}  IoU 分数: "
          f"{', '.join(f'{v:.3f}' for v in ious)}")
    print(f"[结果] 选用最优 mask #{best}（IoU={ious[best]:.3f}），"
          f"前景像素 {int(masks[best].sum())}，推理 {dt:.2f}s")

    vis = overlay(img, masks[best], points=points, labels=labels, box=box)
    vis.save(args.out)
    print(f"[结果] 可视化已保存: {os.path.abspath(args.out)}")


if __name__ == "__main__":
    main()
