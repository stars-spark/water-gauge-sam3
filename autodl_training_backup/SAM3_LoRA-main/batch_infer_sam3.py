#!/usr/bin/env python3
"""
SAM3 + LoRA Batch Inference Script
专门用于批量处理文件夹内的所有图片
"""

import argparse
import os
import glob
import sys
from tqdm import tqdm
import torch
import numpy as np
from PIL import Image as PILImage

# ⚠️ 必须在导入 pyplot 前设置 Agg 后端，防止 AutoDL 报错
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import yaml
from torchvision.ops import nms

# SAM3 imports
from sam3.model_builder import build_sam3_image_model
from sam3.train.data.sam3_image_dataset import (
    Datapoint,
    Image as SAMImage,
    FindQueryLoaded,
    InferenceMetadata
)
from sam3.train.data.collator import collate_fn_api
from sam3.model.utils.misc import copy_data_to_device
from sam3.train.transforms.basic_for_api import (
    ComposeAPI, RandomResizeAPI, ToTensorAPI, NormalizeAPI,
)

# LoRA imports
from lora_layers import LoRAConfig, apply_lora_to_model, load_lora_weights

class SAM3LoRABatchInference:
    def __init__(self, config_path, weights_path=None, resolution=1008, 
                 threshold=0.5, nms_iou=0.5, device="cuda"):
        
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        if weights_path is None:
            # 尝试自动寻找 output 目录下的 best_lora_weights.pt
            weights_path = "SAM3_LoRa_outputs/best_lora_weights.pt"
            
        print(f"🔧 初始化模型...")
        print(f"   配置文件: {config_path}")
        print(f"   权重文件: {weights_path}")
        print(f"   分辨率: {resolution}x{resolution}")

        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.resolution = resolution
        self.threshold = threshold
        self.nms_iou = nms_iou

        # 1. 构建基础模型
        self.model = build_sam3_image_model(
            device=self.device.type,
            compile=False,
            load_from_HF=True,
            # ⚠️ 注意：这里指向你项目里的 bpe 文件路径
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

        # 4. 预处理流程
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
        
        # 预处理
        datapoint = self.transform(datapoint)
        batch = collate_fn_api([datapoint], dict_key="input")["input"]
        
        # 递归移动到 GPU (防止 AttributeError)
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
        
        # 解析结果 (只取第一个 batch)
        res = outputs[0]
        logits = res['pred_logits']
        masks = res.get('pred_masks')
        scores = logits.sigmoid().max(dim=-1)[0].squeeze() # [num_queries]

        keep = scores > self.threshold
        if keep.sum() == 0:
            return None

        final_masks = masks[0, keep] # [N, H, W]
        final_scores = scores[keep]

        # 插值回原图尺寸
        final_masks = torch.nn.functional.interpolate(
            final_masks.unsqueeze(1).float(),
            size=(h, w),
            mode='bilinear',
            align_corners=False
        ).squeeze(1) > 0.5

        return {
            "masks": final_masks.cpu().numpy(),
            "scores": final_scores.cpu().numpy(),
            "image": pil_image
        }

    def save_result(self, result, save_path):
        if result is None:
            return
            
        fig, ax = plt.subplots(1, figsize=(10, 10))
        ax.imshow(result['image'])
        
        # 画 Masks
        masks = result['masks']
        for i in range(len(masks)):
            color = np.array([0, 1, 0, 0.4]) # 绿色
            h, w = masks[i].shape
            colored_mask = np.zeros((h, w, 4))
            colored_mask[masks[i]] = color
            ax.imshow(colored_mask)

        ax.axis('off')
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0, dpi=100)
        plt.close()

def main():
    parser = argparse.ArgumentParser(description="批量推理脚本")
    parser.add_argument("--config", required=True, help="Config yaml路径")
    parser.add_argument("--input_dir", required=True, help="输入图片文件夹")
    parser.add_argument("--output_dir", required=True, help="输出结果文件夹")
    parser.add_argument("--prompt", default=["water_gauge"], nargs="+", help="提示词")
    parser.add_argument("--weights", default=None, help="LoRA权重路径")
    parser.add_argument("--resolution", type=int, default=1008, help="分辨率 (必须与训练一致)")
    
    args = parser.parse_args()

    # 1. 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)

    # 2. 初始化模型 (只做一次)
    inferencer = SAM3LoRABatchInference(
        config_path=args.config,
        weights_path=args.weights,
        resolution=args.resolution
    )

    # 3. 遍历图片
    image_exts = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
    image_files = []
    for ext in image_exts:
        image_files.extend(glob.glob(os.path.join(args.input_dir, ext)))
    
    print(f"📂 发现 {len(image_files)} 张图片，开始处理...")

    # 4. 批量循环
    for img_path in tqdm(image_files):
        try:
            filename = os.path.basename(img_path)
            save_path = os.path.join(args.output_dir, filename)
            
            # 推理
            result = inferencer.process_image(img_path, args.prompt)
            
            # 保存
            if result:
                inferencer.save_result(result, save_path)
            else:
                # 没检测到也复制原图或者跳过，这里选择跳过
                pass
                
        except Exception as e:
            print(f"❌ 处理 {filename} 失败: {e}")
            continue

    print(f"✨ 全部完成！结果保存在: {args.output_dir}")

if __name__ == "__main__":
    main()