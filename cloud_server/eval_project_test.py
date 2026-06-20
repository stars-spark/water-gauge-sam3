# SAM3 LoRA 项目性能实测脚本
# 用法: PYTHONPATH=. python eval_project_test.py <gauge|waterline>
import sys, types, os, time, json, glob
# --- stub decord (仅图像推理，无需视频) ---
_m = types.ModuleType('decord'); _m.cpu = lambda *a, **k: None; _m.VideoReader = object
_m.bridge = types.SimpleNamespace(set_bridge=lambda *a, **k: None)
sys.modules['decord'] = _m

import numpy as np
import torch
from PIL import Image
from pycocotools.coco import COCO
from pycocotools import mask as mask_utils
from batch_infer_sam_json import SAM3LoRABatchInference

BASE = "/media/jiale/AI_Local/水尺水位测量系统"
CKPT = "checkpoints/sam3.pt"

def gt_union_mask(coco, img_id, h, w, cat_ids):
    ann_ids = coco.getAnnIds(imgIds=img_id, catIds=cat_ids)
    anns = coco.loadAnns(ann_ids)
    m = np.zeros((h, w), bool)
    for a in anns:
        rle = coco.annToMask(a).astype(bool)
        m |= rle
    return m, len(anns)

def iou(a, b):
    inter = np.logical_and(a, b).sum()
    uni = np.logical_or(a, b).sum()
    return float(inter) / float(uni) if uni > 0 else (1.0 if a.sum()==b.sum()==0 else 0.0)

def run(mode):
    if mode == "gauge":
        cfg, weights = "configs/full_lora_config.yaml", "SAM3_LoRa_outputs/best_lora_weights.pt"
        test_dir = f"{BASE}/02_数据集/water_gauge_v3_coco/test"
        prompts = ["water_gauge"]; gt_names = ["water_gauge"]
    else:  # waterline
        cfg = f"{BASE}/03_云端SAM3/autodl_training_backup/SAM3_LoRA-main/configs/full_lora_config.yaml"
        weights = f"{BASE}/03_云端SAM3/autodl_training_backup/SAM3_LoRA-main/SAM3_LoRa_Waterline_outputs/best_lora_weights.pt"
        test_dir = f"{BASE}/02_数据集/waterline_v1_coco/test"
        prompts = ["Gauge_Water"]; gt_names = ["Gauge_Water"]

    coco = COCO(f"{test_dir}/_annotations.coco.json")
    cat_ids = [c['id'] for c in coco.loadCats(coco.getCatIds()) if c['name'] in gt_names]
    print(f"\n=== 模式={mode}  prompt={prompts}  GT类别ids={cat_ids} ===")

    inf = SAM3LoRABatchInference(config_path=cfg, weights_path=weights,
                                 checkpoint_path=CKPT, resolution=1008, threshold=0.4, device="cuda")

    outdir = f"eval_out_{mode}"; os.makedirs(outdir, exist_ok=True)
    ious, scores, lats, wl_err = [], [], [], []
    n_det = 0; imgs = coco.loadImgs(coco.getImgIds())
    for im in imgs:
        path = os.path.join(test_dir, im['file_name'])
        if not os.path.exists(path): continue
        h, w = im['height'], im['width']
        gt, ngt = gt_union_mask(coco, im['id'], h, w, cat_ids)
        t0 = time.time(); res = inf.process_image(path, prompts); lat = time.time()-t0
        lats.append(lat)
        if res is None:
            ious.append(0.0 if gt.sum()>0 else 1.0)
            print(f"  {im['file_name'][:30]:30s} 未检出  GT像素={int(gt.sum())}")
            continue
        n_det += 1
        pred = np.zeros((h, w), bool)
        for mk in res['masks']: pred |= mk.astype(bool)
        i = iou(pred, gt); ious.append(i); sc = float(res['scores'].max()); scores.append(sc)
        # 水位线Y：预测水区掩码的最上沿行
        info = ""
        if mode == "waterline" and pred.sum() > 0:
            pred_wl = int(np.argmax(pred.any(axis=1)))  # 第一行有水的y
            if gt.sum() > 0:
                gt_wl = int(np.argmax(gt.any(axis=1)))
                wl_err.append(abs(pred_wl - gt_wl)); info = f" 水位线y pred={pred_wl} gt={gt_wl} 误差={abs(pred_wl-gt_wl)}px"
        print(f"  {im['file_name'][:30]:30s} IoU={i:.3f} score={sc:.3f} {lat:.2f}s{info}")

    n = len(ious)
    print(f"\n----- 汇总 [{mode}] (n={n}) -----")
    print(f"检出率           : {n_det}/{n} = {n_det/n*100:.1f}%")
    print(f"平均IoU(全部)    : {np.mean(ious):.3f}")
    if scores: print(f"平均IoU(检出)    : {np.mean([x for x in ious if x>0]):.3f}   平均置信度: {np.mean(scores):.3f}")
    print(f"延迟 中位/均值   : {np.median(lats):.2f}s / {np.mean(lats):.2f}s (首张含冷启动)")
    if wl_err: print(f"水位线Y误差 中位/均值: {np.median(wl_err):.1f}px / {np.mean(wl_err):.1f}px (n={len(wl_err)})")
    summary = dict(mode=mode, n=n, det_rate=n_det/n, miou=float(np.mean(ious)),
                   miou_det=float(np.mean([x for x in ious if x>0])) if scores else 0,
                   mscore=float(np.mean(scores)) if scores else 0,
                   lat_med=float(np.median(lats)),
                   wl_err_med=float(np.median(wl_err)) if wl_err else None)
    json.dump(summary, open(f"{outdir}/summary.json","w"), indent=2)
    print(f"已保存 {outdir}/summary.json")

if __name__ == "__main__":
    run(sys.argv[1] if len(sys.argv)>1 else "gauge")
