# -*- coding: utf-8 -*-
"""反光防御可视化(显存友好:一次只载一个模型,适配GPU被占只剩~5G的情况)。
原图叠加 水尺掩码(绿)/干区掩码(青)/倒影掩码(红) + 最终水位线(黄)/倒影顶沿(品红)/尺底(白)，标注引擎mode。
判定逻辑复刻自 measure_engine.measure()(引擎已锁定,这里只读不改)。
用法：PYTHONPATH=. /home/jiale/sam3_test_venv/bin/python viz_reflection.py
产物：viz_reflection.png
"""
import sys, types, os, gc, numpy as np
_m=types.ModuleType('decord'); _m.cpu=lambda *a,**k:None; _m.VideoReader=object
_m.bridge=types.SimpleNamespace(set_bridge=lambda *a,**k:None); sys.modules['decord']=_m
import torch
from PIL import Image
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from measure_engine import union_mask, largest_cc
from batch_infer_sam_json import SAM3LoRABatchInference

FONT="/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
def fp(sz,b=False): return fm.FontProperties(fname=FONT,size=sz,weight="bold" if b else "normal") if os.path.exists(FONT) else fm.FontProperties(size=sz)
BASE="/media/jiale/AI_Local/水尺水位测量系统"; V3=f"{BASE}/02_数据集/water_gauge_v3_coco"; WL=f"{BASE}/02_数据集/waterline_v1_coco"
ADL=f"{BASE}/03_云端SAM3/autodl_training_backup/SAM3_LoRA-main"

PICKS=[
  (f"{V3}/train", "water_gauge_000015_jpg.rf.2a2a", "v3反光·干区严重泄漏"),
  (f"{V3}/train", "water_gauge_000016_jpg.rf.98ac", "v3反光"),
  (f"{WL}/test",  "water_gauge_000058", "干净·干区下沿"),
  (f"{WL}/test",  "water_gauge_000012", "干净·过分割护栏"),
]
def resolve(folder, stem):
    if os.path.isdir(folder):
        for f in sorted(os.listdir(folder)):
            if f.startswith(stem) and f.lower().endswith((".jpg",".png")): return os.path.join(folder,f)
    return None

items=[]
for folder,stem,tag in PICKS:
    p=resolve(folder,stem)
    if p: items.append((p,tag))
    else: print("跳过(未找到):",stem)
sizes={p:Image.open(p).size for p,_ in items}   # (w,h)

# ---- 阶段1：gauge 模型 → water_gauge + reflection ----
print("[1/2] 载入 gauge 模型...")
gauge=SAM3LoRABatchInference("configs/full_lora_config.yaml","SAM3_LoRa_outputs/best_lora_weights.pt","checkpoints/sam3.pt",1008,0.4,"cuda")
G={}; R={}
for p,_ in items:
    w,h=sizes[p]
    G[p]=largest_cc(union_mask(gauge.process_image(p,["water_gauge"]),h,w))
    R[p]=union_mask(gauge.process_image(p,["reflection"]),h,w)
del gauge; gc.collect(); torch.cuda.empty_cache()
print("    gauge 释放，剩余显存:", round(torch.cuda.mem_get_info()[0]/1e9,2),"GB")

# ---- 阶段2：air 模型 → Gauge_Air ----
print("[2/2] 载入 air 模型...")
air=SAM3LoRABatchInference(f"{ADL}/configs/full_lora_config.yaml",f"{ADL}/SAM3_LoRa_Waterline_outputs/best_lora_weights.pt","checkpoints/sam3.pt",1008,0.4,"cuda")
A={}
for p,_ in items:
    w,h=sizes[p]; A[p]=union_mask(air.process_image(p,["Gauge_Air"]),h,w)
del air; gc.collect(); torch.cuda.empty_cache()

# ---- 判定逻辑(复刻 measure_engine.measure，只读) ----
def decide(gmask,amask,rmask,h,w):
    gy=np.where(gmask.any(axis=1))[0]
    if len(gy)==0: return None
    g_top,g_bot=int(gy.min()),int(gy.max()); gauge_h=max(1,g_bot-g_top)
    a_bot=float(np.where(amask.any(axis=1))[0].max()) if amask.sum()>0 else None
    air_cov=amask.sum()/float(h*w); refl_cov=rmask.sum()/float(h*w)
    REFL=False; refl_top=None
    if rmask.sum()>0:
        refl_top=float(np.where(rmask.any(axis=1))[0].min()); mid=int(g_top+0.5*gauge_h)
        frac_upper=float(rmask[:mid,:].sum())/float(rmask.sum())
        REFL=(refl_cov>0.003 and frac_upper<0.15 and refl_top>g_top+0.3*gauge_h)
    if a_bot is None: wl_y=float(g_bot); mode="无干区:退尺底"
    elif air_cov>0.25 and a_bot>g_bot+0.15*gauge_h:
        if REFL and refl_top<g_bot-0.10*gauge_h: wl_y=refl_top; mode="反光半淹:倒影顶沿"
        else: wl_y=float(g_bot); mode="过分割护栏:退尺底"
    else: wl_y=a_bot; mode="干区下沿"
    return dict(g_top=g_top,g_bot=g_bot,a_bot=a_bot,wl_y=wl_y,mode=mode,
                refl_top=(refl_top if REFL else None),refl_cov=round(refl_cov,4))

# ---- 渲染 ----
def blend(img,mask,color,a=0.45):
    for c in range(3): img[...,c]=np.where(mask,img[...,c]*(1-a)+color[c]*a,img[...,c])
fig,axes=plt.subplots(1,len(items),figsize=(5.2*len(items),8))
if len(items)==1: axes=[axes]
for ax,(p,tag) in zip(axes,items):
    w,h=sizes[p]; img=np.array(Image.open(p).convert("RGB")).astype(float)
    gm,am,rm=G[p],A[p],R[p]; d=decide(gm,am,rm,h,w)
    blend(img,gm,(60,220,90),0.22); blend(img,am,(40,160,255),0.42)
    if rm.sum()>0: blend(img,rm,(255,60,60),0.42)
    ax.imshow(img.astype(np.uint8)); ax.set_xticks([]); ax.set_yticks([])
    gx=np.where(gm.any(axis=0))[0]; x0,x1=(int(gx.min()),int(gx.max())) if len(gx) else (0,w)
    if d:
        ax.hlines(d["g_bot"],0,w,colors="white",linestyles=":",lw=1.2)
        if d["refl_top"] is not None: ax.hlines(d["refl_top"],0,w,colors="magenta",linestyles="--",lw=1.6)
        ax.hlines(d["wl_y"],max(0,x0-40),min(w,x1+40),colors="yellow",lw=3.2)
        ax.set_title(f"{tag}\nmode={d['mode']}\n水位线y={d['wl_y']:.0f} 尺底={d['g_bot']} 倒影顶={d['refl_top']}",fontproperties=fp(10,True))
    else:
        ax.set_title(f"{tag}\n未检出水尺",fontproperties=fp(10,True))
from matplotlib.patches import Patch; from matplotlib.lines import Line2D
leg=[Patch(color=(60/255,220/255,90/255),alpha=.5,label="水尺掩码"),
     Patch(color=(40/255,160/255,1),alpha=.6,label="干区(Gauge_Air)掩码"),
     Patch(color=(1,60/255,60/255),alpha=.6,label="倒影(reflection)掩码"),
     Line2D([0],[0],color="yellow",lw=3,label="最终水位线"),
     Line2D([0],[0],color="magenta",ls="--",lw=1.6,label="倒影顶沿(水面候选)"),
     Line2D([0],[0],color="white",ls=":",lw=1.2,label="尺底")]
fig.legend(handles=leg,loc="lower center",ncol=6,prop=fp(9),frameon=False,bbox_to_anchor=(0.5,-0.01))
fig.suptitle("反光防御可视化：干区(青)漏进倒影(红)时，护栏/反光分支如何把水位线(黄)定在正确处",fontproperties=fp(12,True))
plt.tight_layout(rect=[0,0.04,1,0.96])
out=f"{BASE}/03_云端SAM3/cloud_server/viz_reflection.png"
plt.savefig(out,dpi=130,facecolor="white",bbox_inches="tight"); print("saved:",out)
