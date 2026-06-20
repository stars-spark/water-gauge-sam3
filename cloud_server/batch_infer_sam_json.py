# 文件名: batch_infer_sam_json.py
import torch
import os
import yaml
import numpy as np
from PIL import Image as PILImage
import matplotlib
matplotlib.use('Agg') # 防止本地无GUI报错
import matplotlib.pyplot as plt
from pycocotools import mask as mask_utils

# 引入项目依赖 (确保你已经把 sam3 文件夹和 lora_layers.py 下载下来了)
from sam3.model_builder import build_sam3_image_model
from sam3.train.data.sam3_image_dataset import Datapoint, Image as SAMImage, FindQueryLoaded, InferenceMetadata
from sam3.train.data.collator import collate_fn_api
from sam3.train.transforms.basic_for_api import ComposeAPI, RandomResizeAPI, ToTensorAPI, NormalizeAPI
from lora_layers import LoRAConfig, apply_lora_to_model, load_lora_weights

class SAM3LoRABatchInference:
    def __init__(self, config_path, weights_path, checkpoint_path, resolution=1008, threshold=0.5, device="cuda"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        print(f"🔧 初始化模型...")
        print(f"   基础模型: {checkpoint_path}")
        print(f"   LoRA权重: {weights_path}")

        self.device = torch.device(device)
        self.resolution = resolution
        self.threshold = threshold

        # 1. 构建基础模型 (显式指定 checkpoint_path)
        self.model = build_sam3_image_model(
            device=self.device.type,
            compile=False,
            load_from_HF=False, # 本地加载，不走HF
            checkpoint_path=checkpoint_path, # 👈 关键：指定本地大模型路径
            bpe_path="sam3/assets/bpe_simple_vocab_16e6.txt.gz", 
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

        # 3. 加载 LoRA 权重
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
        print("✅ 模型加载完毕！")

    @torch.no_grad()
    def process_image(self, image_path, text_prompts):
        pil_image = PILImage.open(image_path).convert("RGB")
        w, h = pil_image.size

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
        
        def move_to_device_rec(obj, dev):
            if isinstance(obj, torch.Tensor): return obj.to(dev)
            elif isinstance(obj, list): return [move_to_device_rec(x, dev) for x in obj]
            elif hasattr(obj, "__dataclass_fields__"):
                for field in obj.__dataclass_fields__:
                    setattr(obj, field, move_to_device_rec(getattr(obj, field), dev))
                return obj
            return obj

        batch = move_to_device_rec(batch, self.device)
        outputs = self.model(batch)
        
        res = outputs[0]
        logits = res['pred_logits']
        masks = res.get('pred_masks')
        if logits.ndim == 3: logits = logits.squeeze(0)
        if masks.ndim == 4: masks = masks.squeeze(0)

        scores = logits.sigmoid().max(dim=-1)[0]
        keep = scores > self.threshold
        
        if keep.sum() == 0: return None

        final_masks = masks[keep]
        final_scores = scores[keep]

        # 插值回原图
        final_masks_resized = torch.nn.functional.interpolate(
            final_masks.unsqueeze(1).float(),
            size=(h, w),
            mode='bilinear',
            align_corners=False
        ).squeeze(1) > 0.5

        return {
            "masks": final_masks_resized.cpu().numpy(),
            "scores": final_scores.cpu().numpy(),
            "image": pil_image
        }