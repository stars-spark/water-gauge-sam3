# -*- coding: utf-8 -*-
"""端云协同demo step2(sam3 venv): 云端SAM3(V3)在 全幅 vs YOLO-ROI 上分割,渲3列流程图+A/B。"""
import os, json, numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib import font_manager as fm
from PIL import Image
from scipy import ndimage
FP=fm.FontProperties(fname="/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
from measure_engine import union_mask, largest_cc, WaterLevelMeasurerV2
D=json.load(open("/tmp/demo_yolo.json"))
# 取2张750全景 + 1张strip
order=["000028","000034","water_gauge_000051"]
items=[(k,D[k]) for k in order if k in D]
M=WaterLevelMeasurerV2()
def outline(m,it=3):
    return (m & ~ndimage.binary_erosion(m,iterations=it)) if m.sum() else m
def seg(path):
    w,h=Image.open(path).size
    gm=largest_cc(union_mask(M.gauge.process_image(path,["water_gauge"]),h,w))
    am=union_mask(M.air.process_image(path,["Gauge_Air"]),h,w)
    r=M.measure(path)
    return gm,am,r,(w,h)
n=len(items); fig,axes=plt.subplots(n,3,figsize=(13,5.2*n)); 
if n==1: axes=axes[None,:]
for row,(k,d) in enumerate(items):
    full=d["full"]; roi=d["roi"]; box=d["box"]; rb=d["roi_box"]
    imgF=np.array(Image.open(full).convert("RGB")); imgR=np.array(Image.open(roi).convert("RGB"))
    # col0: 全幅+YOLO框
    a=axes[row,0]; a.imshow(imgF); a.axis("off")
    x0,y0,x1,y1=box; a.add_patch(mpatches.Rectangle((x0,y0),x1-x0,y1-y0,fill=False,ec="lime",lw=3))
    a.set_title(f"① 边缘 YOLO 检测水尺\n{d['full_size'][0]}×{d['full_size'][1]} conf={d['conf']}",fontproperties=FP,fontsize=12)
    # col1: ROI
    a=axes[row,1]; a.imshow(imgR); a.axis("off")
    a.set_title(f"② 裁ROI上传 (云边协同关键)\n{d['roi_kb']}KB vs 全幅{d['full_kb']}KB  省{int(d['save_pct'])}%带宽",fontproperties=FP,fontsize=12)
    # col2: 云端SAM3 on ROI
    gm,am,r,(w,h)=seg(roi)
    a=axes[row,2]; ov=imgR.astype(float).copy()
    Ac=largest_cc(am) if am.any() else am
    for c,v in zip(range(3),[0,210,210]): ov[...,c]=np.where(Ac, ov[...,c]*0.6+v*0.4, ov[...,c])
    gl=outline(gm)
    for c,v in zip(range(3),[255,230,0]): ov[...,c]=np.where(gl,v,ov[...,c])
    a.imshow(ov.astype(np.uint8)); a.axis("off")
    wy=r.get("waterline_y"); 
    if wy is not None: a.axhline(wy,color="red",lw=2.5)
    lv=r.get("level_cm"); rel=r.get("reliable")
    a.set_title(f"③ 云端 SAM3 精分割+水位线\n黄=尺体 青=干区 红=水位线  读数{lv}cm reliable={rel}",fontproperties=FP,fontsize=12)
fig.suptitle("端云协同实证：边缘 YOLO 粗检测裁ROI(省~60%带宽/低功耗) → 云端 SAM3 精分割解算水位(中位3px/0.4cm)",fontproperties=FP,fontsize=14)
plt.tight_layout(); out="/media/jiale/AI_Local/水尺水位测量系统/05_演示/端云协同实证.png"
plt.savefig(out,dpi=85,bbox_inches="tight"); print("SAVED",out)
