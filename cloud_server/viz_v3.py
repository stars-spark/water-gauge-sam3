# -*- coding: utf-8 -*-
"""V3 epoch04 视觉审片:渲染 test 16图的 干区掩码(青) + 预测水位线(红实) vs GT水位线(绿虚) 叠加,
出 4x4 montage 供目检(确认是真分割对、非刷指标)。ASCII标题避免字体问题。"""
import os, glob, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from pycocotools.coco import COCO
from measure_engine import union_mask, largest_cc, decide_waterline
from batch_infer_sam_json import SAM3LoRABatchInference
from lora_layers import load_lora_weights

WL=os.environ["WL_DIR"]; SAM3PT=os.environ.get("SAM3_PT","checkpoints/sam3.pt")
GCFG=os.environ.get("GAUGE_CFG","configs/full_lora_config.yaml"); GW=os.environ.get("GAUGE_W","SAM3_LoRa_outputs/best_lora_weights.pt")
V3CFG=os.environ["V3_CFG"]; V3W=os.environ["V3_W"]
OUTDIR=os.environ.get("VIZ_OUT","viz_v3"); os.makedirs(OUTDIR,exist_ok=True)
RES=int(os.environ.get("RES","1008")); DEV="cuda"

paths=sorted(glob.glob(f"{WL}/test/*.jpg"))
coco=COCO(f"{WL}/test/_annotations.coco.json"); fn2id={im['file_name']:im['id'] for im in coco.loadImgs(coco.getImgIds())}
def gt_wl(fn,h,w):
    iid=fn2id.get(fn)
    if iid is None: return None
    m=np.zeros((h,w),bool)
    for a in coco.loadAnns(coco.getAnnIds(imgIds=iid,catIds=[1])): m|=coco.annToMask(a).astype(bool)
    return int(np.where(m.any(axis=1))[0].max()) if m.sum()>0 else None

print("[gauge] 缓存 G/R ...", flush=True)
g=SAM3LoRABatchInference(GCFG,GW,SAM3PT,RES,0.4,DEV)
G={};R={};sz={}
for p in paths:
    w,h=Image.open(p).size; sz[p]=(w,h); G[p]=largest_cc(union_mask(g.process_image(p,["water_gauge"]),h,w)); R[p]=union_mask(g.process_image(p,["reflection"]),h,w)
import gc,torch; del g; gc.collect(); torch.cuda.empty_cache()
print("[v3 air] 加载 epoch04 ...", flush=True)
a=SAM3LoRABatchInference(V3CFG,V3W,SAM3PT,RES,0.4,DEV); load_lora_weights(a.model,V3W); a.model.eval()

n=len(paths); cols=4; rows=(n+cols-1)//cols
fig,axes=plt.subplots(rows,cols,figsize=(cols*4,rows*4)); axes=axes.ravel()
for i,p in enumerate(paths):
    w,h=sz[p]; img=np.array(Image.open(p).convert("RGB"))
    A=union_mask(a.process_image(p,["Gauge_Air"]),h,w)
    r=decide_waterline(G[p],A,R[p],h,w); gt=gt_wl(os.path.basename(p),h,w)
    ax=axes[i]; ax.imshow(img)
    ov=np.zeros((*A.shape,4)); ov[A.astype(bool)]=[0,1,1,0.35]; ax.imshow(ov)  # 干区 青色半透
    wl=r.get("waterline_y") if r.get("ok") else None
    if wl is not None: ax.axhline(wl,color="red",lw=2)
    if gt is not None: ax.axhline(gt,color="lime",lw=1.5,ls="--")
    pe=abs(wl-gt) if (wl is not None and gt is not None) else -1
    fn=os.path.basename(p)[:14]
    ax.set_title(f"{fn} e={pe:.0f}px mode={r.get('mode','-')}",fontsize=9)
    ax.axis("off")
for j in range(n,len(axes)): axes[j].axis("off")
plt.suptitle("V3 ep04: cyan=dry-zone(Gauge_Air)  red=pred waterline  green--=GT",fontsize=12)
plt.tight_layout()
out=f"{OUTDIR}/v3_ep04_montage.png"; plt.savefig(out,dpi=85,bbox_inches="tight"); print("saved",out,flush=True)
print("@@@VIZ_DONE@@@")
