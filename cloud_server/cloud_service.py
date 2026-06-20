# -*- coding: utf-8 -*-
"""
统一云端 SAM3 水位测量服务（FastAPI，原生路径）。

替代旧的 local_server.py：
  - 统一到原生 SAM3 + LoRA 推理（sam3_infer.SAM3Inferencer），抛弃 Gradio 的
    transformers/MockInferenceSession 那条脆弱链路（即 waterline_y=None 的根因）；
  - 返回**完整水位测量契约**(含 waterline_y / 方法 / 置信度 / 检测明细)，供边缘端经
    L610 上行后解析；
  - 可选回传可视化叠加图(base64)。

启动:
  .venv-linux/bin/python cloud_service.py            # 默认 waterline 模型
  MODE=water_gauge .venv-linux/bin/python cloud_service.py
接口:
  GET  /health          健康检查
  POST /predict         multipart 上传图片 -> 水位结果 JSON
"""
import os
import io
import time
import base64

import numpy as np
import torch
from PIL import Image as PILImage
from fastapi import FastAPI, File, UploadFile, Query
from fastapi.responses import JSONResponse

from sam3_infer import SAM3Inferencer, solve_waterline, union_mask

# ----------------------------- 配置 -----------------------------
DEVICE = os.environ.get("DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
MODE = os.environ.get("MODE", "waterline")           # waterline | water_gauge
CHECKPOINT = os.environ.get("SAM3_CKPT", "checkpoints/sam3.pt")
RESOLUTION = int(os.environ.get("RESOLUTION", "1008"))
THRESHOLD = float(os.environ.get("THRESHOLD", "0.4"))

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

app = FastAPI(title="云边协同水尺水位测量 - 云端SAM3服务", version="1.0")
_INF = None
_PROFILE = PROFILES[MODE]


@app.on_event("startup")
def _load():
    global _INF
    print(f"🚀 启动云端服务  mode={MODE}  device={DEVICE}")
    _INF = SAM3Inferencer(
        config_path=_PROFILE["config"],
        weights_path=_PROFILE["weights"],
        checkpoint_path=CHECKPOINT,
        resolution=RESOLUTION,
        threshold=THRESHOLD,
        device=DEVICE,
    )


@app.get("/health")
def health():
    return {"status": "ok" if _INF is not None else "loading",
            "mode": MODE, "device": DEVICE,
            "prompts": _PROFILE["prompts"], "resolution": RESOLUTION}


def _overlay_b64(pil_img, seg, waterline_y):
    """生成可视化叠加图(掩码半透明 + 水位线)，返回 base64 JPEG。"""
    import cv2
    img = cv2.cvtColor(np.array(pil_img.convert("RGB")), cv2.COLOR_RGB2BGR)
    colors = {"Gauge_Air": (0, 200, 0), "Gauge_Water": (255, 120, 0),
              "water_gauge": (0, 200, 0)}
    for prompt, entry in seg.items():
        m = union_mask(entry)
        if m is None:
            continue
        color = colors.get(prompt, (0, 200, 0))
        overlay = img.copy()
        overlay[m] = color
        img = cv2.addWeighted(overlay, 0.4, img, 0.6, 0)
    if waterline_y is not None:
        h, w = img.shape[:2]
        y = min(max(0, int(waterline_y)), h - 1)
        cv2.line(img, (0, y), (w, y), (0, 0, 255), 3)
        cv2.putText(img, f"waterline y={y}", (10, max(20, y - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return base64.b64encode(buf.tobytes()).decode() if ok else None


@app.post("/predict")
async def predict(file: UploadFile = File(...),
                  visualize: bool = Query(False, description="是否回传可视化base64")):
    if _INF is None:
        return JSONResponse({"status": "error", "msg": "模型尚未加载完成"}, status_code=503)
    t0 = time.time()
    try:
        raw = await file.read()
        pil = PILImage.open(io.BytesIO(raw)).convert("RGB")
    except Exception as e:
        return JSONResponse({"status": "error", "msg": f"图片解码失败: {e}"}, status_code=400)

    w, h = pil.size
    seg = _INF.segment(pil, _PROFILE["prompts"])
    wl = solve_waterline(seg, MODE)

    detections = {}
    for prompt, entry in seg.items():
        n = int(entry["masks"].shape[0])
        detections[prompt] = {
            "instances": n,
            "top_score": float(entry["scores"][0]) if n else None,
            "total_pixels": int(entry["masks"].sum()) if n else 0,
        }

    resp = {
        "status": "success" if wl["waterline_y"] is not None else "no_waterline",
        "mode": MODE,
        "image_size": {"height": h, "width": w},
        "waterline_y": wl["waterline_y"],
        "waterline_y_ratio": (round(wl["waterline_y"] / h, 4)
                              if wl["waterline_y"] is not None else None),
        "waterline_method": wl["method"],
        "waterline_confidence": wl["confidence"],
        "waterline_detail": wl["detail"],
        "detections": detections,
        "latency_sec": round(time.time() - t0, 3),
    }
    if visualize:
        resp["overlay_jpeg_b64"] = _overlay_b64(pil, seg, wl["waterline_y"])
    return resp


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", "8000")))
