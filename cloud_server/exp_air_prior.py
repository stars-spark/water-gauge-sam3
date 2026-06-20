# -*- coding: utf-8 -*-
"""air自先验 科研消融实验:全集148场景配对A(关)/B(开)+阈值敏感性+回归检查。输出 /tmp/air_exp.json。"""
import os, glob, json, numpy as np
from PIL import Image
from pycocotools.coco import COCO
from measure_engine import union_mask, largest_cc, decide_waterline, WaterLevelMeasurerV2
WL="/media/jiale/AI_Local/水尺水位测量系统/02_数据集/waterline_v1_coco_clean"
# 收集唯一场景 + GT(尺体范围Gauge_Air∪Water, 水位线=Gauge_Air下沿)
items=[]; seen=set()
for split in ["train","valid","test"]:
    d=f"{WL}/{split}"; coco=COCO(f"{d}/_annotations.coco.json")
    for iid in coco.getImgIds():
        im=coco.loadImgs(iid)[0]; fn=im['file_name']; base=fn.split(".rf.")[0]
        if base in seen: continue
        seen.add(base); h,w=im['height'],im['width']
        air=np.zeros((h,w),bool); full=np.zeros((h,w),bool)
        for a in coco.loadAnns(coco.getAnnIds(imgIds=iid,catIds=[1])): air|=coco.annToMask(a).astype(bool)
        for a in coco.loadAnns(coco.getAnnIds(imgIds=iid,catIds=[1,2])): full|=coco.annToMask(a).astype(bool)
        wl=int(np.where(air.any(axis=1))[0].max()) if air.sum()>0 else None
        gy=np.where(full.any(axis=1))[0]; ext=(int(gy.min()),int(gy.max())) if len(gy) else None
        gtread=None
        if wl is not None and ext and ext[1]>ext[0]:
            frac=(wl-ext[0])/(ext[1]-ext[0]); gtread=round((1-min(1,max(0,frac)))*100,1)
        items.append((base,f"{d}/{fn}",w,h,gtread))
print(f"唯一场景 {len(items)}",flush=True)
M=WaterLevelMeasurerV2(); rt,rb=M.rt,M.rb
SWEEP=[0.1,0.2,0.3,0.4,0.5]; SUP=max(SWEEP)
rows=[]
for i,(base,p,w,h,gtread) in enumerate(items):
    gm=largest_cc(union_mask(M.gauge.process_image(p,["water_gauge"]),h,w))
    am=union_mask(M.air.process_image(p,["Gauge_Air"]),h,w)
    rm=union_mask(M.gauge.process_image(p,["reflection"]),h,w)
    ov=(gm & am).sum()/float(am.sum()) if am.sum()>0 else 1.0
    rA=decide_waterline(gm,am,rm,h,w,rt,rb)
    rec=None
    if ov < SUP and am.sum()>0 and gm.sum()>0:
        g2=M._gauge_air_prior(p,am,h,w)
        if g2 is not None and g2.sum()>0 and (g2 & am).sum()/float(am.sum())>ov:
            rB=decide_waterline(g2,am,rm,h,w,rt,rb)
            rec={"level":rB.get("level_cm"),"reliable":rB.get("reliable")}
    rows.append({"base":base,"gt":gtread,"ov":round(float(ov),3),
                 "A":{"level":rA.get("level_cm"),"reliable":rA.get("reliable")},"rec":rec})
    if i%30==0: print(f"  {i}/{len(items)}",flush=True)
json.dump({"sweep":SWEEP,"rows":rows}, open("/tmp/air_exp.json","w"))
print("SAVED /tmp/air_exp.json","@@@EXP_DONE@@@",flush=True)
