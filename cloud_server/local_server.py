# 文件名: local_server.py
import uvicorn
from fastapi import FastAPI, File, UploadFile
import torch
import shutil
import os
import time
from batch_infer_sam_json import SAM3LoRABatchInference

app = FastAPI()

# ==========================================
# ⚠️ 这里根据你电脑情况修改
# 如果你有 NVIDIA 显卡 -> "cuda"
# 如果你是 Mac M芯片 或 普通笔记本 -> "cpu"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# ==========================================

print(f"🚀 启动服务 (使用设备: {DEVICE})...")

# 实例化推理器
try:
    inferencer = SAM3LoRABatchInference(
        config_path="configs/full_lora_config.yaml",
        weights_path="SAM3_LoRa_outputs/best_lora_weights.pt",
        # ⚠️ 注意这里的文件名，要和你下载下来的大模型文件名一致
        checkpoint_path="checkpoints/sam3_hiera_large.pt", 
        resolution=1008,
        threshold=0.4,
        device=DEVICE
    )
except Exception as e:
    print(f"❌ 致命错误: {e}")
    exit(1)

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    temp_filename = f"temp_{int(time.time())}.jpg"
    try:
        with open(temp_filename, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        print(f"🔍 正在推理: {file.filename}")
        result = inferencer.process_image(temp_filename, ["water_gauge"])
        
        if result is None:
            return {"status": "fail", "msg": "未检测到水尺"}

        mask_count = int(result['masks'][0].sum())
        score = float(result['scores'][0])
        print(f"✅ 检测成功! 置信度: {score:.2f}, 掩码像素: {mask_count}")

        return {
            "status": "success",
            "confidence": score,
            "mask_pixel_count": mask_count
        }

    finally:
        if os.path.exists(temp_filename):
            os.remove(temp_filename)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)