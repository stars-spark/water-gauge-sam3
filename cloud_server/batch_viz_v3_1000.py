# -*- coding: utf-8 -*-
"""V3 水位线分割批量效果图（无GT，纯推理+叠加渲染）。
流程(同部署/viz_v3): gauge LoRA→水尺(water_gauge,取最大连通域)+倒影(reflection)
                     → V3 air LoRA→干区(Gauge_Air) → decide_waterline 定水位线
                     → 渲染 青=干区掩膜 + 红=水位线 叠加图，逐张存 PNG。
显存只够单模型：两段式(先 gauge 全跑缓存掩码[packbits省内存]，再换 V3)。
用法: cd cloud_server; PYTHONPATH=. N=1000 /home/jiale/sam3_test_venv/bin/python batch_viz_v3_1000.py
"""
import sys, os, gc, glob, time, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from measure_engine import union_mask, largest_cc, decide_waterline  # 导入即装 decord stub
from batch_infer_sam_json import SAM3LoRABatchInference
from lora_layers import load_lora_weights
import torch
from PIL import Image
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = "/media/jiale/AI_Local/水尺水位测量系统"
SRC  = os.environ.get("SRC", f"{BASE}/02_数据集/water_gauge_v3_coco")
OUT  = os.environ.get("OUT", f"{BASE}/05_演示/v3_水位线分割_1000")
N    = int(os.environ.get("N", "1000"))
RES  = int(os.environ.get("RES", "1008")); DEV = "cuda"
GCFG = os.environ.get("GAUGE_CFG", "configs/full_lora_config.yaml")
GW   = os.environ.get("GAUGE_W",   "SAM3_LoRa_outputs/best_lora_weights.pt")
V3CFG= os.environ.get("V3_CFG", "checkpoints/waterline_v3_clean/v3_encoder_clean.yaml")
V3W  = os.environ.get("V3_W",   "checkpoints/waterline_v3_clean/epoch05_lora_weights.pt")
SAM3PT = os.environ.get("SAM3_PT", "checkpoints/sam3.pt")

for f in (GCFG, GW, V3CFG, V3W, SAM3PT):           # fail fast
    if not os.path.exists(f): sys.exit(f"[缺文件] {f}  (cwd={os.getcwd()})")
os.makedirs(OUT, exist_ok=True)

paths = (sorted(glob.glob(f"{SRC}/train/*.jpg")) +
         sorted(glob.glob(f"{SRC}/valid/*.jpg")) +
         sorted(glob.glob(f"{SRC}/test/*.jpg")))[:N]
print(f"[共 {len(paths)} 张] -> {OUT}", flush=True)
sz = {p: Image.open(p).size for p in paths}

t0 = time.time()
print("[1/2] gauge 模型: water_gauge + reflection ...", flush=True)
g = SAM3LoRABatchInference(GCFG, GW, SAM3PT, RES, 0.4, DEV)
Gpk = {}; Rpk = {}
for i, p in enumerate(paths):
    w, h = sz[p]
    G = largest_cc(union_mask(g.process_image(p, ["water_gauge"]), h, w))
    R = union_mask(g.process_image(p, ["reflection"]), h, w)
    Gpk[p] = (np.packbits(G), G.shape); Rpk[p] = (np.packbits(R), R.shape)
    if i % 50 == 0: print(f"  gauge {i}/{len(paths)}  {time.time()-t0:.0f}s", flush=True)
del g; gc.collect(); torch.cuda.empty_cache()

print("[2/2] V3 air 模型: Gauge_Air + 渲染 ...", flush=True)
a = SAM3LoRABatchInference(V3CFG, V3W, SAM3PT, RES, 0.4, DEV)
load_lora_weights(a.model, V3W); a.model.eval()
ok_n = 0
for i, p in enumerate(paths):
    w, h = sz[p]
    A = union_mask(a.process_image(p, ["Gauge_Air"]), h, w)
    G = np.unpackbits(Gpk[p][0])[:h*w].reshape(Gpk[p][1]).astype(bool)
    R = np.unpackbits(Rpk[p][0])[:h*w].reshape(Rpk[p][1]).astype(bool)
    r = decide_waterline(G, A, R, h, w)
    img = np.array(Image.open(p).convert("RGB"))
    fig, ax = plt.subplots(figsize=(w/100.0, h/100.0), dpi=100)
    ax.imshow(img)
    ov = np.zeros((*A.shape, 4)); ov[A.astype(bool)] = [0, 1, 1, 0.35]; ax.imshow(ov)
    wl = r.get("waterline_y") if r.get("ok") else None
    if wl is not None:
        ax.axhline(wl, color="red", lw=2); ok_n += 1
    ax.axis("off")
    base = os.path.splitext(os.path.basename(p))[0]
    fig.savefig(f"{OUT}/{base}_v3seg.png", bbox_inches="tight", pad_inches=0, dpi=100)
    plt.close(fig)
    if i % 50 == 0: print(f"  v3 {i}/{len(paths)} ok={ok_n}  {time.time()-t0:.0f}s", flush=True)
print(f"[完成] {len(paths)} 张, 出水位线 {ok_n} 张, 用时 {time.time()-t0:.0f}s -> {OUT}", flush=True)
