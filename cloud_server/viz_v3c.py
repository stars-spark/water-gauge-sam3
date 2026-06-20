# -*- coding: utf-8 -*-
"""V3 ep04 关键图大图审片:指定几张,每张全图大尺寸+水位线区放大,看掩码与红(预测)/绿(GT)线。"""
import os, glob, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from pycocotools.coco import COCO
from measure_engine import union_mask, largest_cc, decide_waterline
from batch_infer_sam_json import SAM3LoRABatchInference
from lora_layers import load_lora_weights

WL=os.environ["WL_DIR"]; SAM3PT=os.environ.get("SAM3_PT","checkpoints/sam3.pt")
GCFG=os.environ.get("GAUGE_CFG"); GW=os.environ.get("GAUGE_W")
V3CFG=os.environ["V3_CFG"]; V3W=os.environ["V3_W"]; RES=int(os.environ.get("RES","1008")); DEV="cuda"
TARGETS=os.environ.get("TARGETS","005,000003,000150").split(",")

allp=sorted(glob.glob(f"{WL}/test/*.jpg"))
paths=[p for p in allp if any(t in os.path.basename(p) for t in TARGETS)]
coco=COCO(f"{WL}/test/_annotations.coco.json"); fn2id={im['file_name']:im['id'] for im in coco.loadImgs(coco.getImgIds())}
def gt_wl(fn,h,w):
    iid=fn2id.get(fn);
    if iid is None: return None
    m=np.zeros((h,w),bool)
    for a in coco.loadAnns(coco.getAnnIds(imgIds=iid,catIds=[1])): m|=coco.annToMask(a).astype(bool)
    return int(np.where(m.any(axis=1))[0].max()) if m.sum()>0 else None

g=SAM3LoRABatchInference(GCFG,GW,SAM3PT,RES,0.4,DEV)
G={};R={};sz={}
for p in paths:
    w,h=Image.open(p).size; sz[p]=(w,h); G[p]=largest_cc(union_mask(g.process_image(p,["water_gauge"]),h,w)); R[p]=union_mask(g.process_image(p,["reflection"]),h,w)
import gc,torch; del g; gc.collect(); torch.cuda.empty_cache()
a=SAM3LoRABatchInference(V3CFG,V3W,SAM3PT,RES,0.4,DEV); load_lora_weights(a.model,V3W); a.model.eval()

ncol=len(paths)*2  # 每张:全图 + 水位线放大
fig,axes=plt.subplots(1,ncol,figsize=(ncol*3.2,13))
for i,p in enumerate(paths):
    w,h=sz[p]; img=np.array(Image.open(p).convert("RGB"))
    A=union_mask(a.process_image(p,["Gauge_Air"]),h,w); r=decide_waterline(G[p],A,R[p],h,w)
    wl=r.get("waterline_y") if r.get("ok") else None; gt=gt_wl(os.path.basename(p),h,w)
    pe=abs(wl-gt) if (wl is not None and gt is not None) else -1
    ov=np.zeros((*A.shape,4)); ov[A.astype(bool)]=[0,1,1,0.35]
    # 全图
    ax=axes[2*i]; ax.imshow(img); ax.imshow(ov)
    if wl is not None: ax.axhline(wl,color="red",lw=2)
    if gt is not None: ax.axhline(gt,color="lime",lw=1.5,ls="--")
    ax.set_title(f"{os.path.basename(p)[:10]}  e={pe:.0f}px",fontsize=11); ax.axis("off")
    # 水位线放大(±70px)
    ax2=axes[2*i+1]; c=int(wl if wl is not None else (gt if gt else h//2)); y0=int(max(0,c-70)); y1=int(min(h,c+70))
    ax2.imshow(img[y0:y1]); ov2=ov[y0:y1]; ax2.imshow(ov2)
    if wl is not None: ax2.axhline(wl-y0,color="red",lw=2)
    if gt is not None: ax2.axhline(gt-y0,color="lime",lw=1.5,ls="--")
    ax2.set_title("zoom@waterline",fontsize=10); ax2.axis("off")
plt.suptitle("V3 ep04 key audit: cyan=dry-zone  red=pred  green--=GT",fontsize=13)
plt.tight_layout()
out=os.environ.get("VIZ_OUT","viz_v3")+"/v3_ep04_key.png"; plt.savefig(out,dpi=110,bbox_inches="tight"); print("saved",out)
print("@@@VIZKEY_DONE@@@")
