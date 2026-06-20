# -*- coding: utf-8 -*-
"""
统一 SAM3 LoRA 推理核心（原生路径，非 transformers）。

相对旧 batch_infer_sam_json.py 的修正：
  1) 构造 LoRAConfig 时读取**全部** apply_to_* 开关（含 mask_decoder / geometry），
     避免 strict=False 静默丢弃 segmentation_head 等 LoRA 权重；
  2) 加载后输出**权重覆盖率**（matched / unexpected），第一时间暴露配置-权重不匹配；
  3) 内置水位线解算（基于分割掩码），供云端服务直接返回 waterline_y。

被 cloud_service.py（FastAPI）与命令行自测共用。
"""
import sys
import os
import yaml
import numpy as np
from PIL import Image as PILImage
import torch

# 确保能找到同目录的 sam3 包与 lora_layers
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sam3.model_builder import build_sam3_image_model
from sam3.train.data.sam3_image_dataset import (
    Datapoint, Image as SAMImage, FindQueryLoaded, InferenceMetadata,
)
from sam3.train.data.collator import collate_fn_api
from sam3.train.transforms.basic_for_api import (
    ComposeAPI, RandomResizeAPI, ToTensorAPI, NormalizeAPI,
)
from lora_layers import LoRAConfig, apply_lora_to_model


def _move_to_device_rec(obj, dev):
    if isinstance(obj, torch.Tensor):
        return obj.to(dev)
    if isinstance(obj, list):
        return [_move_to_device_rec(x, dev) for x in obj]
    if hasattr(obj, "__dataclass_fields__"):
        for field in obj.__dataclass_fields__:
            setattr(obj, field, _move_to_device_rec(getattr(obj, field), dev))
        return obj
    return obj


class SAM3Inferencer:
    """加载一套 (基础模型 + LoRA) 并提供分割 / 水位线解算。"""

    def __init__(self, config_path, weights_path, checkpoint_path,
                 resolution=1008, threshold=0.4, device="cuda",
                 bpe_path="sam3/assets/bpe_simple_vocab_16e6.txt.gz"):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

        self.device = torch.device(device if torch.cuda.is_available() or device == "cpu" else "cpu")
        self.resolution = resolution
        self.threshold = threshold

        print(f"[SAM3] 基础模型: {checkpoint_path}")
        print(f"[SAM3] LoRA权重: {weights_path}")
        print(f"[SAM3] 设备: {self.device.type}  分辨率: {resolution}  阈值: {threshold}")

        # 1) 基础模型（本地权重，不走 HF）
        self.model = build_sam3_image_model(
            device=self.device.type,
            compile=False,
            load_from_HF=False,
            checkpoint_path=checkpoint_path,
            bpe_path=bpe_path,
            eval_mode=True,
        )

        # 2) 应用 LoRA —— 读取全部开关
        lcfg = self.config["lora"]
        lora_config = LoRAConfig(
            rank=lcfg["rank"],
            alpha=lcfg["alpha"],
            dropout=0.0,
            target_modules=lcfg["target_modules"],
            apply_to_vision_encoder=lcfg.get("apply_to_vision_encoder", False),
            apply_to_text_encoder=lcfg.get("apply_to_text_encoder", False),
            apply_to_geometry_encoder=lcfg.get("apply_to_geometry_encoder", False),
            apply_to_detr_encoder=lcfg.get("apply_to_detr_encoder", False),
            apply_to_detr_decoder=lcfg.get("apply_to_detr_decoder", True),
            apply_to_mask_decoder=lcfg.get("apply_to_mask_decoder", False),
        )
        self.model = apply_lora_to_model(self.model, lora_config)

        # 3) 加载 LoRA 权重 + 覆盖率校验
        self._load_lora_with_report(weights_path)

        self.model.to(self.device)
        self.model.eval()

        # 4) 预处理（与训练一致）
        self.transform = ComposeAPI(transforms=[
            RandomResizeAPI(sizes=resolution, max_size=resolution, square=True,
                            consistent_transform=False),
            ToTensorAPI(),
            NormalizeAPI(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])
        print("[SAM3] ✅ 模型就绪")

    def _load_lora_with_report(self, weights_path):
        sd = torch.load(weights_path, map_location="cpu")
        result = self.model.load_state_dict(sd, strict=False)
        n_file = len(sd)
        unexpected = list(result.unexpected_keys)
        matched = n_file - len(unexpected)
        cov = 100.0 * matched / n_file if n_file else 0.0
        print(f"[SAM3] LoRA权重: {matched}/{n_file} 命中 ({cov:.1f}%)")
        if unexpected:
            print(f"[SAM3] ⚠️ {len(unexpected)} 个权重未匹配(配置可能不符)，示例:")
            for k in unexpected[:5]:
                print(f"         - {k}")
            if cov < 90.0:
                print("[SAM3] ⚠️⚠️ 覆盖率<90%，LoRA 可能基本未生效，请核对 config 的 target_modules/apply_to_*")

    @torch.no_grad()
    def _segment_one(self, pil_image, text, thr):
        """单提示词单图分割 -> {'masks': (N,H,W)bool, 'scores': (N,)}。"""
        w, h = pil_image.size
        query = FindQueryLoaded(
            query_text=text, image_id=0, object_ids_output=[], is_exhaustive=True,
            inference_metadata=InferenceMetadata(
                coco_image_id=0, original_image_id=0, original_category_id=0,
                original_size=[w, h], object_id=0, frame_index=0),
        )
        sam_image = SAMImage(data=pil_image, objects=[], size=[h, w])
        datapoint = Datapoint(find_queries=[query], images=[sam_image])
        datapoint = self.transform(datapoint)
        batch = collate_fn_api([datapoint], dict_key="input")["input"]
        batch = _move_to_device_rec(batch, self.device)

        outputs = self.model(batch)
        res = outputs[0]
        logits = res["pred_logits"]
        masks = res.get("pred_masks")
        if logits.ndim == 3:
            logits = logits.squeeze(0)
        if masks.ndim == 4:
            masks = masks.squeeze(0)
        scores = logits.sigmoid().max(dim=-1)[0]
        keep = scores > thr
        if keep.sum() == 0:
            return {"masks": np.zeros((0, h, w), bool), "scores": np.zeros((0,))}
        fm = masks[keep]
        fs = scores[keep]
        fm = torch.nn.functional.interpolate(
            fm.unsqueeze(1).float(), size=(h, w),
            mode="bilinear", align_corners=False).squeeze(1) > 0.5
        return {"masks": fm.cpu().numpy(), "scores": fs.cpu().numpy()}

    def segment(self, pil_image, text_prompts, threshold=None):
        """对单图按多文本提示分割（逐提示词前向）。
        返回 {prompt: {'masks': (N,H,W)bool, 'scores': (N,)}}。"""
        thr = self.threshold if threshold is None else threshold
        pil_image = pil_image.convert("RGB")
        return {p: self._segment_one(pil_image, p, thr) for p in text_prompts}


# --------------------------- 水位线解算 ---------------------------

def union_mask(seg_entry):
    """把某提示下所有实例掩码并起来。"""
    masks = seg_entry["masks"]
    if masks.shape[0] == 0:
        return None
    return np.any(masks, axis=0)


def robust_extreme_y(mask, side="bottom", quantile=0.02):
    """掩码的稳健极值 y。

    side='bottom': 取最低端但用分位数抗孤立噪点(默认丢弃最底 2% 像素的极值)。
    side='top':    取最高端。
    返回 int 或 None。
    """
    ys = np.where(mask.any(axis=1))[0]  # 有掩码像素的行
    if ys.size == 0:
        return None
    if side == "bottom":
        return int(np.quantile(ys, 1.0 - quantile))
    return int(np.quantile(ys, quantile))


def solve_waterline(seg, mode):
    """根据分割结果解算水位线 y。

    mode='waterline': 用 Gauge_Air 底边为主，Gauge_Water 顶边交叉校验。
    mode='water_gauge': 用水尺整体掩码底边（次优代理）。
    返回 dict: {waterline_y, method, confidence, detail}
    """
    if mode == "waterline":
        air = seg.get("Gauge_Air")
        water = seg.get("Gauge_Water")
        air_m = union_mask(air) if air else None
        water_m = union_mask(water) if water else None
        y_air = robust_extreme_y(air_m, "bottom") if air_m is not None else None
        y_water = robust_extreme_y(water_m, "top") if water_m is not None else None

        if y_air is not None and y_water is not None:
            wl = int(round((y_air + y_water) / 2))
            gap = abs(y_air - y_water)
            return {"waterline_y": wl, "method": "air_bottom∩water_top",
                    "confidence": "high" if gap <= 25 else "medium",
                    "detail": {"gauge_air_bottom_y": y_air, "gauge_water_top_y": y_water,
                               "gap_px": gap}}
        if y_air is not None:
            return {"waterline_y": y_air, "method": "air_bottom",
                    "confidence": "medium", "detail": {"gauge_air_bottom_y": y_air}}
        if y_water is not None:
            return {"waterline_y": y_water, "method": "water_top",
                    "confidence": "low", "detail": {"gauge_water_top_y": y_water}}
        return {"waterline_y": None, "method": "none", "confidence": "none", "detail": {}}

    # mode == water_gauge
    keys = list(seg.keys())
    gauge_m = union_mask(seg[keys[0]]) if keys else None
    y = robust_extreme_y(gauge_m, "bottom") if gauge_m is not None else None
    return {"waterline_y": y, "method": "gauge_mask_bottom",
            "confidence": "low" if y is not None else "none",
            "detail": {"note": "整尺掩码底边，非Air/Water交界，建议用 waterline 模型"}}


# --------------------------- 命令行自测 ---------------------------

if __name__ == "__main__":
    import argparse, json, time
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["water_gauge", "waterline"], default="water_gauge")
    ap.add_argument("--image", default="test_image.jpg")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    PROFILES = {
        "water_gauge": dict(
            config="configs/full_lora_config.yaml",
            weights="SAM3_LoRa_outputs/best_lora_weights.pt",
            prompts=["water_gauge"]),
        "waterline": dict(
            config="configs/waterline_lora_config.yaml",
            weights="../autodl_training_backup/SAM3_LoRA-main/SAM3_LoRa_Waterline_outputs/best_lora_weights.pt",
            prompts=["Gauge_Air", "Gauge_Water"]),
    }
    p = PROFILES[args.mode]
    t0 = time.time()
    inf = SAM3Inferencer(
        config_path=p["config"], weights_path=p["weights"],
        checkpoint_path="checkpoints/sam3.pt", resolution=1008,
        threshold=0.4, device=args.device)
    print(f"[selftest] load {time.time()-t0:.1f}s")

    img = PILImage.open(args.image)
    t1 = time.time()
    seg = inf.segment(img, p["prompts"])
    wl = solve_waterline(seg, args.mode)
    dt = time.time() - t1
    for k, v in seg.items():
        n = v["masks"].shape[0]
        top = float(v["scores"][0]) if n else None
        px = int(v["masks"][0].sum()) if n else 0
        print(f"[selftest] prompt={k!r}: {n} masks, top_score={top}, px(top)={px}")
    print(f"[selftest] 图像尺寸 HxW = {img.size[1]}x{img.size[0]}")
    print(f"[selftest] 水位线解算: {json.dumps(wl, ensure_ascii=False)}")
    print(f"[selftest] 推理 {dt:.2f}s")
