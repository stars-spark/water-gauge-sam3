#!/usr/bin/env python3
"""
SAM3 + LoRA Batch Inference Script (Enhanced with JSON Output)
功能：批量推理，输出可视化图片，并生成对应的 JSON 掩码数据文件。
"""

import argparse
import os
import glob
import sys
import json
from tqdm import tqdm
import torch
import numpy as np
from PIL import Image as PILImage

# ⚠️ 必须在导入 pyplot 前设置 Agg 后端
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import yaml
from pycocotools import mask as mask_utils # 用于高效压缩 Mask

# SAM3 imports
from sam3.model_builder import build_sam3_image_model
from sam3.train.data.sam3_image_dataset import (
    Datapoint, Image as SAMImage, FindQueryLoaded, InferenceMetadata
)
from sam3.train.data.collator import collate_fn_api
from sam3.train.transforms.basic_for_api import (
    ComposeAPI, RandomResizeAPI, ToTensorAPI, NormalizeAPI,
)
from lora_layers import LoRAConfig, apply_lora_to_model, load_lora_weights

# 辅助函数：将 Mask 转换为 RLE 格式 (压缩存储)
def mask_to_rle(binary_mask):
    """
    将二进制 Mask (numpy bool) 转换为 COCO RLE 格式。
    这样可以极大减小 JSON 文件体积。
    """
    rle = mask_utils.encode(np.asfortranarray(binary_mask.astype(np.uint8)))
    # mask_utils.encode 返回的 counts 是 bytes，需要转为字符串才能存 JSON
    rle['counts'] = rle['counts'].decode('utf-8')
    return rle

class SAM3LoRABatchInference:
    def __init__(self, config_path, weights_path=None, resolution=1008, 
                 threshold=0.5, device="cuda"):
        
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        if weights_path is None:
            weights_path = "SAM3_LoRa_outputs/best_lora_weights.pt"
            
        print(f"🔧 初始化模型...")
        print(f"   权重文件: {weights_path}")
        print(f"   分辨率: {resolution}x{resolution}")

        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.resolution = resolution
        self.threshold = threshold

        # 1. 构建基础模型
        self.model = build_sam3_image_model(
            device=self.device.type,
            compile=False,
            load_from_HF=True,
            bpe_path="./sam3/assets/bpe_simple_vocab_16e6.txt.gz", 
            eval_mode=True
        )

        # 2. 应用 LoRA
        lora_cfg = self.config["lora"]
        lora_config = LoRAConfig(
            rank=lora_cfg["rank"],
            alpha=lora_cfg["alpha"],
            dropout=0.0,
            target_modules=lora_cfg["target_modules"],
            apply_to_vision_encoder=lora_cfg["apply_to_vision_encoder"],
            apply_to_text_encoder=lora_cfg["apply_to_text_encoder"],
            apply_to_detr_encoder=lora_cfg["apply_to_detr_encoder"],
            apply_to_detr_decoder=lora_cfg["apply_to_detr_decoder"],
        )
        self.model = apply_lora_to_model(self.model, lora_config)

        # 3. 加载权重
        load_lora_weights(self.model, weights_path)
        self.model.to(self.device)
        self.model.eval()

        # 4. 预处理
        self.transform = ComposeAPI(
            transforms=[
                RandomResizeAPI(sizes=resolution, max_size=resolution, square=True, consistent_transform=False),
                ToTensorAPI(),
                NormalizeAPI(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )
        print("✅ 模型加载完毕，准备推理！\n")

    @torch.no_grad()
    def process_image(self, image_path, text_prompts):
        pil_image = PILImage.open(image_path).convert("RGB")
        w, h = pil_image.size

        # 构造输入
        queries = []
        for idx, text in enumerate(text_prompts):
            q = FindQueryLoaded(
                query_text=text, image_id=0, object_ids_output=[], is_exhaustive=True,
                inference_metadata=InferenceMetadata(
                    coco_image_id=0, original_image_id=0, original_category_id=0,
                    original_size=[w, h], object_id=0, frame_index=0
                )
            )
            queries.append(q)

        sam_image = SAMImage(data=pil_image, objects=[], size=[h, w])
        datapoint = Datapoint(find_queries=queries, images=[sam_image])
        
        datapoint = self.transform(datapoint)
        batch = collate_fn_api([datapoint], dict_key="input")["input"]
        
        # 递归移动到 GPU
        def move_to_device_rec(obj, dev):
            if isinstance(obj, torch.Tensor): return obj.to(dev)
            elif isinstance(obj, list): return [move_to_device_rec(x, dev) for x in obj]
            elif hasattr(obj, "__dataclass_fields__"):
                for field in obj.__dataclass_fields__:
                    setattr(obj, field, move_to_device_rec(getattr(obj, field), dev))
                return obj
            return obj

        batch = move_to_device_rec(batch, self.device)

        # 推理
        outputs = self.model(batch)
        
        res = outputs[0]
        logits = res['pred_logits']
        masks = res.get('pred_masks')
        
        # 处理 batch 维度
        if logits.ndim == 3: logits = logits.squeeze(0)
        if masks.ndim == 4: masks = masks.squeeze(0)

        scores = logits.sigmoid().max(dim=-1)[0] # [num_queries]

        keep = scores > self.threshold
        if keep.sum() == 0:
            return None

        final_masks = masks[keep] # [N, H, W]
        final_scores = scores[keep]

        # 插值回原图尺寸
        final_masks_resized = torch.nn.functional.interpolate(
            final_masks.unsqueeze(1).float(),
            size=(h, w),
            mode='bilinear',
            align_corners=False
        ).squeeze(1) > 0.5

        return {
            "masks": final_masks_resized.cpu().numpy(), # bool array
            "scores": final_scores.cpu().numpy(),
            "image": pil_image,
            "filename": os.path.basename(image_path),
            "orig_size": (w, h)
        }

    def save_result(self, result, output_dir):
        if result is None: return

        filename_base = os.path.splitext(result['filename'])[0]
        
        # 1. 保存可视化图片 (JPG)
        img_save_path = os.path.join(output_dir, result['filename'])
        fig, ax = plt.subplots(1, figsize=(10, 10))
        ax.imshow(result['image'])
        
        masks = result['masks']
        scores = result['scores']
        
        for i in range(len(masks)):
            color = np.array([0, 1, 0, 0.4]) # 绿色
            h, w = masks[i].shape
            colored_mask = np.zeros((h, w, 4))
            colored_mask[masks[i]] = color
            ax.imshow(colored_mask)

        ax.axis('off')
        plt.savefig(img_save_path, bbox_inches='tight', pad_inches=0, dpi=100)
        plt.close()

        # 2. 保存 Mask 数据 (JSON)
        json_save_path = os.path.join(output_dir, f"{filename_base}.json")
        
        json_data = {
            "image_file": result['filename'],
            "width": result['orig_size'][0],
            "height": result['orig_size'][1],
            "annotations": []
        }

        for i in range(len(masks)):
            # 压缩 Mask 数据
            rle = mask_to_rle(masks[i])
            annotation = {
                "id": i,
                "score": float(scores[i]),
                "segmentation": rle,  # COCO RLE 格式
                "bbox": mask_utils.toBbox(rle).tolist() # [x, y, w, h]
            }
            json_data["annotations"].append(annotation)
        
        with open(json_save_path, 'w') as f:
            json.dump(json_data, f)

def main():
    parser = argparse.ArgumentParser(description="批量推理脚本 (含 JSON 输出)")
    parser.add_argument("--config", required=True)
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--prompt", default=["water_gauge"], nargs="+")
    parser.add_argument("--weights", default=None)
    parser.add_argument("--resolution", type=int, default=1008)
    
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    inferencer = SAM3LoRABatchInference(
        config_path=args.config,
        weights_path=args.weights,
        resolution=args.resolution
    )

    image_files = []
    for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
        image_files.extend(glob.glob(os.path.join(args.input_dir, ext)))
    
    print(f"📂 发现 {len(image_files)} 张图片，开始处理...")

    for img_path in tqdm(image_files):
        try:
            result = inferencer.process_image(img_path, args.prompt)
            if result:
                inferencer.save_result(result, args.output_dir)
        except Exception as e:
            print(f"❌ 失败: {os.path.basename(img_path)} - {e}")
            continue

    print(f"✨ 全部完成！结果保存在: {args.output_dir}")

if __name__ == "__main__":
    main()