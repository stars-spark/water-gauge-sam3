# -*- coding: utf-8 -*-
"""
闭环评估：WaterLine test16 → V2引擎测水位线 → 渲染叠加图(供多模态目检) + 误差表
输出: /tmp/wl_eval/overlay_XX.jpg (绿=水尺掩码 蓝=干区掩码 红线=预测水位线 黄线=GT)
     /tmp/wl_eval/report.txt
运行: cd cloud_server && sam3_venv/bin/python eval_visual_loop.py
"""
import sys, types, os, glob
_m=types.ModuleType('decord'); _m.cpu=lambda *a,**k:None; _m.VideoReader=object
_m.bridge=types.SimpleNamespace(set_bridge=lambda *a,**k:None); sys.modules['decord']=_m
import numpy as np
from PIL import Image, ImageDraw
from pycocotools.coco import COCO
from measure_engine import WaterLevelMeasurerV2, union_mask

BASE = "/media/jiale/AI_Local/水尺水位测量系统"
WL   = f"{BASE}/02_数据集/waterline_v1_coco"
OUT  = "/tmp/wl_eval"
os.makedirs(OUT, exist_ok=True)

c = COCO(f"{WL}/test/_annotations.coco.json")
fn2id = {im['file_name']: im['id'] for im in c.loadImgs(c.getImgIds())}

def gt_wl(fn, h, w):
    iid = fn2id.get(fn)
    if iid is None: return None
    m = np.zeros((h, w), bool)
    for a in c.loadAnns(c.getAnnIds(imgIds=iid, catIds=[1])):
        m |= c.annToMask(a).astype(bool)
    return int(np.where(m.any(axis=1))[0].max()) if m.sum() > 0 else None

M = WaterLevelMeasurerV2()
reals = sorted(glob.glob(f"{WL}/test/water_gauge_*.jpg"))
rows, errs = [], []
for k, p in enumerate(reals):
    fn = os.path.basename(p)
    im = Image.open(p).convert("RGB"); w, h = im.size
    g = gt_wl(fn, h, w)
    if g is None: continue
    r = M.measure(p)
    # 重取两掩码用于渲染
    gmask = union_mask(M.gauge.process_image(p, ["water_gauge"]), h, w)
    amask = union_mask(M.air.process_image(p, ["Gauge_Air"]), h, w)
    ov = np.zeros((h, w, 4), np.uint8)
    ov[gmask] = (0, 230, 0, 70)
    ov[amask] = (40, 90, 255, 70)
    img = im.copy(); img.paste(Image.fromarray(ov), (0, 0), Image.fromarray(ov))
    dr = ImageDraw.Draw(img)
    wl_y = r.get("waterline_y")
    if wl_y is not None:
        dr.line([(0, wl_y), (w, wl_y)], fill=(255, 0, 0), width=max(2, h//200))
    dr.line([(0, g), (w, g)], fill=(255, 220, 0), width=max(2, h//300))
    img.thumbnail((560, 560))
    img.save(f"{OUT}/overlay_{k:02d}.jpg", quality=85)
    err = abs((wl_y or h) - g); errs.append(err)
    rows.append(f"{k:02d} {fn[:26]:26s} GT={g:5d} pred={int(wl_y or -1):5d} err={err:5.0f}px conf={r.get('conf',0):.2f}")
    print(rows[-1], flush=True)

errs = np.array(errs)
summary = (f"\n中位={np.median(errs):.0f}px 均值={np.mean(errs):.0f}px 最大={errs.max():.0f}px "
           f"离群(>50px)={np.sum(errs>50)}/{len(errs)}")
print(summary)
with open(f"{OUT}/report.txt", "w") as f:
    f.write("\n".join(rows) + summary + "\n")
print("叠加图与报告已写入", OUT)
