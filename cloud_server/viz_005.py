# -*- coding: utf-8 -*-
"""005清水图:基线air vs V3air 谁对。渲染 干区掩码+预测水位线(红) + GT标注(绿) 两面板对比。
用户指真实水位在'1'附近(靠底部);看基线/ V3 谁的红线接近真实、GT是否标错(偏顶)。"""
import os, glob, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from pycocotools.coco import COCO
from measure_engine import union_mask, largest_cc, decide_waterline
from batch_infer_sam_json import SAM3LoRABatchInference
from lora_layers import load_lora_weights

WL=os.environ["WL_DIR"]; SAM3PT=os.environ.get("SAM3_PT","checkpoints/sam3.pt")
GCFG=os.environ["GAUGE_CFG"]; GW=os.environ["GAUGE_W"]
BCFG=os.environ["BASE_CFG"]; BW=os.environ["BASE_W"]   # 基线air
VCFG=os.environ["V3_CFG"]; VW=os.environ["V3_W"]       # V3 air
RES=int(os.environ.get("RES","1008")); DEV="cuda"
p=[x for x in sorted(glob.glob(f"{WL}/test/*.jpg")) if "005" in os.path.basename(x) and "Clear" in os.path.basename(x)][0]
w,h=Image.open(p).size; img=np.array(Image.open(p).convert("RGB"))
coco=COCO(f"{WL}/test/_annotations.coco.json"); fn2id={im['file_name']:im['id'] for im in coco.loadImgs(coco.getImgIds())}
iid=fn2id[os.path.basename(p)]; m=np.zeros((h,w),bool)
for a in coco.loadAnns(coco.getAnnIds(imgIds=iid,catIds=[1])): m|=coco.annToMask(a).astype(bool)
gt=int(np.where(m.any(axis=1))[0].max()) if m.sum()>0 else None

g=SAM3LoRABatchInference(GCFG,GW,SAM3PT,RES,0.4,DEV)
G=largest_cc(union_mask(g.process_image(p,["water_gauge"]),h,w)); R=union_mask(g.process_image(p,["reflection"]),h,w)
import gc,torch; del g; gc.collect(); torch.cuda.empty_cache()

def get(cfg,wt):
    a=SAM3LoRABatchInference(cfg,wt,SAM3PT,RES,0.4,DEV); load_lora_weights(a.model,wt); a.model.eval()
    A=union_mask(a.process_image(p,["Gauge_Air"]),h,w); r=decide_waterline(G,A,R,h,w)
    del a; gc.collect(); torch.cuda.empty_cache()
    return A, (r.get("waterline_y") if r.get("ok") else None)
Ab,wlb=get(BCFG,BW); Av,wlv=get(VCFG,VW)

fig,ax=plt.subplots(1,2,figsize=(8,16))
for k,(A,wl,name) in enumerate([(Ab,wlb,"BASELINE air"),(Av,wlv,"V3 air ep04")]):
    ax[k].imshow(img); ov=np.zeros((*A.shape,4)); ov[A.astype(bool)]=[0,1,1,0.35]; ax[k].imshow(ov)
    if wl is not None: ax[k].axhline(wl,color="red",lw=2,label=f"pred y={wl}")
    if gt is not None: ax[k].axhline(gt,color="lime",lw=1.5,ls="--",label=f"GT y={gt}")
    ax[k].set_title(f"{name}\npred_y={wl} GT_y={gt} H={h}",fontsize=11); ax[k].legend(loc="lower right",fontsize=8); ax[k].axis("off")
plt.suptitle("005 Clear: cyan=dry-zone red=pred green--=GT (img H=%d)"%h,fontsize=12)
plt.tight_layout(); out=os.environ.get("VIZ_OUT","viz_v3")+"/v005_baseline_vs_v3.png"
plt.savefig(out,dpi=100,bbox_inches="tight"); print("saved",out,"| H=",h,"GT_y=",gt,"baseline_y=",wlb,"v3_y=",wlv)
print("@@@V005_DONE@@@")
