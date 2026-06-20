# -*- coding: utf-8 -*-
"""全测试集逐图水位分割可视化(供人工审片)。WaterLine test 16张(有GT水位线)。
每张: 缩放到水尺区, 叠加 水尺(绿)/干区(青)/倒影(红) + 预测水位线(黄实)/GT水位线(绿虚), 标注 mode/误差。
显存友好(单模型轮流加载)。用法: PYTHONPATH=. /home/jiale/sam3_test_venv/bin/python viz_all_test.py
产物: viz_test_1..4.png  +  误差表打印
"""
import sys, types, os, gc, glob, numpy as np
_m=types.ModuleType('decord'); _m.cpu=lambda *a,**k:None; _m.VideoReader=object
_m.bridge=types.SimpleNamespace(set_bridge=lambda *a,**k:None); sys.modules['decord']=_m
import torch
from PIL import Image
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from pycocotools.coco import COCO
from measure_engine import union_mask, largest_cc, waterline_frac
from batch_infer_sam_json import SAM3LoRABatchInference

FONT="/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
def fp(sz,b=False): return fm.FontProperties(fname=FONT,size=sz,weight="bold" if b else "normal") if os.path.exists(FONT) else fm.FontProperties(size=sz)
BASE="/media/jiale/AI_Local/水尺水位测量系统"; WL=f"{BASE}/02_数据集/waterline_v1_coco"
ADL=f"{BASE}/03_云端SAM3/autodl_training_backup/SAM3_LoRA-main"; RES=1008; DEV="cuda"

paths=sorted(glob.glob(f"{WL}/test/*.jpg"))
coco=COCO(f"{WL}/test/_annotations.coco.json")
fn2id={im['file_name']:im['id'] for im in coco.loadImgs(coco.getImgIds())}
def gt_wl(fn,h,w):
    iid=fn2id.get(fn);
    if iid is None: return None
    m=np.zeros((h,w),bool)
    for a in coco.loadAnns(coco.getAnnIds(imgIds=iid,catIds=[1])): m|=coco.annToMask(a).astype(bool)
    return int(np.where(m.any(axis=1))[0].max()) if m.sum()>0 else None
sizes={p:Image.open(p).size for p in paths}

print(f"[1/2] gauge 模型 (water_gauge + reflection) ...")
g=SAM3LoRABatchInference("configs/full_lora_config.yaml","SAM3_LoRa_outputs/best_lora_weights.pt","checkpoints/sam3.pt",RES,0.4,DEV)
G={}; R={}
for p in paths:
    w,h=sizes[p]; G[p]=largest_cc(union_mask(g.process_image(p,["water_gauge"]),h,w)); R[p]=union_mask(g.process_image(p,["reflection"]),h,w)
del g; gc.collect(); torch.cuda.empty_cache()
print(f"[2/2] air 模型 (Gauge_Air) ...")
a=SAM3LoRABatchInference(f"{ADL}/configs/full_lora_config.yaml",f"{ADL}/SAM3_LoRa_Waterline_outputs/best_lora_weights.pt","checkpoints/sam3.pt",RES,0.4,DEV)
A={}
for p in paths:
    w,h=sizes[p]; A[p]=union_mask(a.process_image(p,["Gauge_Air"]),h,w)
del a; gc.collect(); torch.cuda.empty_cache()

def decide(gmask,amask,rmask,h,w):
    gy=np.where(gmask.any(axis=1))[0]
    if len(gy)==0: return None
    g_top,g_bot=int(gy.min()),int(gy.max()); gauge_h=max(1,g_bot-g_top)
    a_bot=float(np.where(amask.any(axis=1))[0].max()) if amask.sum()>0 else None
    air_cov=amask.sum()/float(h*w); refl_cov=rmask.sum()/float(h*w)
    REFL=False; refl_top=None
    if rmask.sum()>0:
        refl_top=float(np.where(rmask.any(axis=1))[0].min()); mid=int(g_top+0.5*gauge_h)
        fu=float(rmask[:mid,:].sum())/float(rmask.sum()); REFL=(refl_cov>0.003 and fu<0.15 and refl_top>g_top+0.3*gauge_h)
    if a_bot is None: wl_y=float(g_bot); mode="无干区:退尺底"
    elif air_cov>0.25 and a_bot>g_bot+0.15*gauge_h:
        if REFL and refl_top<g_bot-0.10*gauge_h: wl_y=refl_top; mode="反光半淹"
        else: wl_y=float(g_bot); mode="过分割护栏"
    else: wl_y=a_bot; mode="干区下沿"
    return dict(g_top=g_top,g_bot=g_bot,wl_y=wl_y,mode=mode,refl_top=(refl_top if REFL else None))

def blend(img,mask,color,al=0.4):
    for c in range(3): img[...,c]=np.where(mask,img[...,c]*(1-al)+color[c]*al,img[...,c])

rows=[]
PER=4; sheets=(len(paths)+PER-1)//PER
for s in range(sheets):
    chunk=paths[s*PER:(s+1)*PER]
    fig,axes=plt.subplots(1,len(chunk),figsize=(4.6*len(chunk),9)); axes=np.atleast_1d(axes)
    for ax,p in zip(axes,chunk):
        w,h=sizes[p]; fn=os.path.basename(p); img=np.array(Image.open(p).convert("RGB")).astype(float)
        gm,am,rm=G[p],A[p],R[p]; d=decide(gm,am,rm,h,w); g=gt_wl(fn,h,w)
        blend(img,gm,(60,220,90),0.20); blend(img,am,(40,160,255),0.42)
        if rm.sum()>0: blend(img,rm,(255,60,60),0.40)
        ax.imshow(img.astype(np.uint8)); ax.set_xticks([]); ax.set_yticks([])
        if d:
            gx=np.where(gm.any(axis=0))[0]; x0,x1=int(gx.min()),int(gx.max())
            ax.hlines(d["wl_y"],max(0,x0-30),min(w,x1+30),colors="yellow",lw=3)
            if g is not None: ax.hlines(g,max(0,x0-30),min(w,x1+30),colors="lime",linestyles="--",lw=2)
            # 缩放到水尺区
            gy=np.where(gm.any(axis=1))[0]; m=int(0.06*h)
            ax.set_ylim(min(h,int(gy.max())+m), max(0,int(gy.min())-m)); ax.set_xlim(max(0,x0-int(0.15*w)),min(w,x1+int(0.15*w)))
            err=abs(d["wl_y"]-g) if g is not None else None
            rows.append((fn[:22],d["mode"],int(d["wl_y"]),g,err))
            ax.set_title(f"{fn[:20]}\n{d['mode']}  预测y={int(d['wl_y'])}\nGT={g} 误差={err}px",fontproperties=fp(9,True))
        else:
            ax.set_title(f"{fn[:20]}\n未检出水尺",fontproperties=fp(9,True))
    from matplotlib.patches import Patch; from matplotlib.lines import Line2D
    leg=[Patch(color=(40/255,160/255,1),alpha=.6,label="干区掩码"),Patch(color=(1,60/255,60/255),alpha=.6,label="倒影掩码"),
         Line2D([0],[0],color="yellow",lw=3,label="预测水位线"),Line2D([0],[0],color="lime",ls="--",lw=2,label="GT水位线")]
    fig.legend(handles=leg,loc="lower center",ncol=4,prop=fp(9),frameon=False,bbox_to_anchor=(0.5,-0.02))
    plt.tight_layout(rect=[0,0.03,1,1])
    out=f"{BASE}/03_云端SAM3/cloud_server/viz_test_{s+1}.png"; plt.savefig(out,dpi=120,facecolor="white",bbox_inches="tight"); plt.close(); print("saved:",out)

print(f"\n{'图':24s}{'mode':>12s}{'预测y':>7s}{'GT':>6s}{'误差':>6s}")
for fn,mode,wy,g,err in rows:
    gs="-" if g is None else str(int(g)); es="-" if err is None else f"{err:.0f}"
    print(f"{fn:24s}{mode:>12s}{wy:7d}{gs:>6s}{es:>6s}")
errs=np.array([e for *_,e in rows if e is not None])
print(f"\n误差: 中位={np.median(errs):.0f}px 均值={np.mean(errs):.0f}px 最大={errs.max():.0f}px 离群(>30)={int((errs>30).sum())}/{len(errs)}")
print("@@@VIZALLDONE@@@")
