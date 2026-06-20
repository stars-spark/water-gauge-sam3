import os
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
from tqdm import tqdm

try:
    # 🌟 导入 Meta 官方针对 SAM 3 全新设计的 API
    from sam3 import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor
except ImportError:
    print("❌ 找不到 sam3 库。请确保你的环境里安装了 Meta 官方的 sam3 代码！")
    exit(1)


def setup_baseline_model():
    """加载原版 SAM 3 模型"""
    print("⏳ 正在调用官方 API 加载 SAM 3 模型...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint_path = "../checkpoints/sam3.pt"

    try:
        # SAM 3 的新加载方式
        model = build_sam3_image_model(checkpoint_path=checkpoint_path)
        model.to(device)
        model.eval()

        processor = Sam3Processor(model)
        print(f"✅ SAM 3 本地模型加载成功！使用设备: {device}")
        return processor, device
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        return None, None


def draw_mask(image, mask, alpha=0.5, color=(0, 255, 0)):
    """在原图上绘制半透明的绿色掩码"""
    overlay = image.copy()
    # 确保 mask 是 boolean
    mask_bool = mask > 0
    overlay[mask_bool, 1] = 255
    return cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)


def process_image(processor, device, image_path, output_dir):
    """处理单张图片"""
    img_name = os.path.basename(image_path)
    image = cv2.imread(image_path)
    if image is None: return False

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        # 1. SAM 3 新架构：先将图片设置为状态 (State)
        state = processor.set_image(image_rgb)

        # 2. SAM 3 杀手锏：直接输入文本 Prompt！
        text_prompt = "water gauge"
        results = processor.set_text_prompt(state=state, prompt=text_prompt)

    # 3. 提取 Mask (兼容不同的返回结构)
    if hasattr(results, 'masks'):
        masks = results.masks
    elif isinstance(results, dict) and 'masks' in results:
        masks = results['masks']
    elif isinstance(results, tuple):
        masks = results[0]
    else:
        masks = results

    if isinstance(masks, torch.Tensor):
        masks = masks.cpu().numpy()

    # 4. 可视化
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].imshow(image_rgb)
    axes[0].set_title(f"Original Image\nPrompt: '{text_prompt}'")
    axes[0].axis('off')

    if masks is not None and len(masks) > 0 and masks[0].max() > 0:
        vis_image = image_rgb.copy()
        # SAM 3 可能会找出图里所有匹配的对象，我们全部画出来
        for m in masks:
            vis_image = draw_mask(vis_image, m.squeeze())
        axes[1].imshow(vis_image)
        axes[1].set_title("SAM 3 Zero-shot Predict")
    else:
        axes[1].imshow(image_rgb)
        axes[1].set_title("SAM 3 Found Nothing")
    axes[1].axis('off')

    save_path = os.path.join(output_dir, img_name)
    plt.savefig(save_path, bbox_inches='tight')
    plt.close(fig)
    return True


def process_dataset_split(processor, device, split_name, image_dir, output_root, max_images=10):
    split_out_dir = os.path.join(output_root, split_name)
    os.makedirs(split_out_dir, exist_ok=True)

    image_paths = glob(os.path.join(image_dir, "*.jpg")) + glob(os.path.join(image_dir, "*.png"))
    if not image_paths: return

    print(f"\n📂 开始测试 [{split_name}] 集 (抽取 {min(max_images, len(image_paths))} 张)...")
    np.random.shuffle(image_paths)
    test_paths = image_paths[:max_images]

    success_count = 0
    for img_path in tqdm(test_paths, desc=f"Processing {split_name}"):
        if process_image(processor, device, img_path, split_out_dir):
            success_count += 1
    print(f"✅ [{split_name}] 测试完成，结果保存在: {split_out_dir}")


def main():
    processor, device = setup_baseline_model()
    if processor is None: return

    dataset_root = "./data"
    output_root = "./baseline_results"

    splits = {
        "train": os.path.join(dataset_root, "train"),
        "valid": os.path.join(dataset_root, "valid"),
        "test": os.path.join(dataset_root, "test")
    }

    for split_name, img_dir in splits.items():
        if os.path.exists(img_dir):
            process_dataset_split(processor, device, split_name, img_dir, output_root, max_images=10)

    print("\n🎉 SAM 3 纯文本 Zero-shot 测试已完成！")


if __name__ == "__main__":
    main()