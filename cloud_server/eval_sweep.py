# -*- coding: utf-8 -*-
"""高效扫描评估:gauge模型建一次缓存G/R;每变体air base建一次,逐epoch用load_lora_weights换权重
(免每次重载sam3.pt)。按【水位线px】(中位/均值/最大/离群>OUT) 给 baseline + V0-V3 每epoch 打分。
教训落地:按px(非val_loss)选epoch;V3看清水(005)改善且其余不退。
用法(AutoDL): cd sweep && PYTHONPATH=. WL_DIR=... SWEEP_DIR=... ./venv/bin/python eval_sweep.py
"""
import os, gc, glob, numpy as np
import torch
from PIL import Image
from pycocotools.coco import COCO
from measure_engine import union_mask, largest_cc, decide_waterline
from batch_infer_sam_json import SAM3LoRABatchInference
from lora_layers import load_lora_weights

WL=os.environ["WL_DIR"]; SAM3PT=os.environ.get("SAM3_PT","checkpoints/sam3.pt")
GCFG=os.environ.get("GAUGE_CFG","configs/full_lora_config.yaml")
GW=os.environ.get("GAUGE_W","SAM3_LoRa_outputs/best_lora_weights.pt")
SWEEP=os.environ.get("SWEEP_DIR","/root/autodl-tmp/sam3_sweep")
ADL=os.environ.get("ADL_DIR","adl")
RES=int(os.environ.get("RES","1008")); DEV="cuda"; OUT=15.0
CLEAR="005"  # 清水图文件名包含此(005_Clear_Water);单独跟踪V3域差改善

paths=sorted(glob.glob(f"{WL}/test/*.jpg"))
coco=COCO(f"{WL}/test/_annotations.coco.json"); fn2id={im['file_name']:im['id'] for im in coco.loadImgs(coco.getImgIds())}
def gt_wl(fn,h,w):
    iid=fn2id.get(fn)
    if iid is None: return None
    m=np.zeros((h,w),bool)
    for a in coco.loadAnns(coco.getAnnIds(imgIds=iid,catIds=[1])): m|=coco.annToMask(a).astype(bool)
    return int(np.where(m.any(axis=1))[0].max()) if m.sum()>0 else None
sizes={p:Image.open(p).size for p in paths}
GT={p:gt_wl(os.path.basename(p),sizes[p][1],sizes[p][0]) for p in paths}
valid=[p for p in paths if GT[p] is not None]
print(f"test图 {len(paths)} 张, 有GT {len(valid)} 张; 离群阈值 {OUT}px", flush=True)

# 1) gauge 一次,缓存 G/R
print("[gauge] 建模+缓存 G(water_gauge)/R(reflection) ...", flush=True)
g=SAM3LoRABatchInference(GCFG,GW,SAM3PT,RES,0.4,DEV)
G={}; R={}
for p in paths:
    w,h=sizes[p]; G[p]=largest_cc(union_mask(g.process_image(p,["water_gauge"]),h,w)); R[p]=union_mask(g.process_image(p,["reflection"]),h,w)
del g; gc.collect(); torch.cuda.empty_cache()
print("[gauge] 完成", flush=True)

def px_stats(ckpt_eval):
    """ckpt_eval: dict p->pxerr; 返回(中位,均值,最大,离群数,清水px)"""
    arr=np.array([v for v in ckpt_eval.values() if v is not None])
    clear=[v for p,v in ckpt_eval.items() if CLEAR in os.path.basename(p) and v is not None]
    if not len(arr): return None
    return (np.median(arr),np.mean(arr),arr.max(),int((arr>OUT).sum()),(clear[0] if clear else None))

def eval_variant(name, cfg, ckpts):
    """建air base一次,逐ckpt换LoRA权重,算每图px"""
    if not ckpts: print(f"[{name}] 无checkpoint,跳过"); return {}
    print(f"\n===== 变体 {name} ({len(ckpts)} ckpt) =====", flush=True)
    air=SAM3LoRABatchInference(cfg, ckpts[0], SAM3PT, RES, 0.4, DEV)
    res={}
    for ck in ckpts:
        load_lora_weights(air.model, ck); air.model.eval()
        ev={}
        for p in paths:
            w,h=sizes[p]
            A=union_mask(air.process_image(p,["Gauge_Air"]),h,w)
            r=decide_waterline(G[p],A,R[p],h,w)
            if r.get("ok") and GT[p] is not None: ev[p]=abs(r["waterline_y"]-GT[p])
            elif GT[p] is not None: ev[p]=None  # 未检出
        s=px_stats(ev)
        tag=os.path.basename(ck).replace("_lora_weights.pt","")
        if s: print(f"  {tag:10s} 中位{s[0]:5.1f} 均值{s[1]:5.1f} 最大{s[2]:5.1f} 离群{s[3]:2d} 清水005={s[4] if s[4] is not None else '—'}",flush=True)
        res[tag]=s
    del air; gc.collect(); torch.cuda.empty_cache()
    return res

def ck_list(d):
    return sorted(glob.glob(f"{d}/epoch*_lora_weights.pt"))

ALL={}
# baseline = 当前部署air权重
ALL["baseline"]=eval_variant("baseline", f"{ADL}/configs/full_lora_config.yaml", [f"{ADL}/SAM3_LoRa_Waterline_outputs/best_lora_weights.pt"])
for v,cfg in [("v0",f"{SWEEP}/configs/sweep_overfit/v0_baseline.yaml"),
              ("v1",f"{SWEEP}/configs/sweep_overfit/v1_lowcap.yaml"),
              ("v2",f"{SWEEP}/configs/sweep_overfit/v2_strongreg.yaml"),
              ("v3",f"{SWEEP}/configs/sweep_overfit/v3_encoder.yaml")]:
    ALL[v]=eval_variant(v, cfg, ck_list(f"{SWEEP}/SAM3_LoRa_Waterline_sweep_{v}"))

# 汇总:每变体px最优epoch(按中位,平最大次之) + 对照baseline
print("\n\n========== 汇总:每变体 px 最优 epoch ==========",flush=True)
base=ALL.get("baseline",{}).get("best") or list(ALL.get("baseline",{}).values() or [None])[0]
if base: print(f"[基线 当前部署权重] 中位{base[0]:.1f} 最大{base[2]:.1f} 离群{base[3]} 清水005={base[4]}")
for v in ["v0","v1","v2","v3"]:
    d={k:s for k,s in ALL.get(v,{}).items() if s}
    if not d: continue
    bestk=min(d, key=lambda k:(d[k][0], d[k][2]))  # 中位优先,最大次之
    s=d[bestk]
    flag=""
    if base:
        flag=" ✅赢基线" if (s[0]<=base[0] and s[2]<=base[2]) else " ✗未赢"
    print(f"[{v}] px最优={bestk} 中位{s[0]:.1f} 最大{s[2]:.1f} 离群{s[3]} 清水005={s[4]}{flag}")
print("\n判读铁律: 只有中位&最大都≤基线才算赢(看离群/最差,非只中位); V3额外看清水005是否大降且其余不退。")
print("@@@SWEEP_EVAL_DONE@@@")
