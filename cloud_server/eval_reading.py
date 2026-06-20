# -*- coding: utf-8 -*-
"""端到端"水位线像素→具体水尺数值(cm)"自闭环测试 + 误差分解(把误差地板与瓶颈量化)。
WaterLine test 16张(GT水位线像素)。读数=从尺底起算的cm(reading↑随水位↑)。误差分三块：
  ① 纯水位线引入(用真尺度100/尺高,隔离分割误差)  ② E字自标定cm/px相对误差  ③ 端到端读数误差(自标定×水位线)
显存友好(单模型轮流加载,~3.4G)。同时跑的是真·decide_waterline → 顺带验证clamp/线1-B在16图上不回归。
用法: PYTHONPATH=. /home/jiale/sam3_test_venv/bin/python eval_reading.py
"""
import sys, types, os, gc, glob, numpy as np
_m=types.ModuleType('decord'); _m.cpu=lambda *a,**k:None; _m.VideoReader=object
_m.bridge=types.SimpleNamespace(set_bridge=lambda *a,**k:None); sys.modules['decord']=_m
import torch
from PIL import Image
from pycocotools.coco import COCO
from measure_engine import union_mask, largest_cc, decide_waterline
from scale_calibration import calibrate
from batch_infer_sam_json import SAM3LoRABatchInference

BASE=os.environ.get("PROJ_BASE","/media/jiale/AI_Local/水尺水位测量系统"); WL=os.environ.get("WL_DIR",f"{BASE}/02_数据集/waterline_v1_coco")
ADL=os.environ.get("ADL_DIR",f"{BASE}/03_云端SAM3/autodl_training_backup/SAM3_LoRA-main"); RES=int(os.environ.get("RES","1008")); DEV="cuda"
# [#2 扫描] gauge模型 config/权重/base 也可经env覆盖(AutoDL复用)
GCFG=os.environ.get("GAUGE_CFG","configs/full_lora_config.yaml"); GW=os.environ.get("GAUGE_W","SAM3_LoRa_outputs/best_lora_weights.pt"); SAM3PT=os.environ.get("SAM3_PT","checkpoints/sam3.pt")
print(f"[RES={RES}]")
paths=sorted(glob.glob(f"{WL}/test/*.jpg"))
coco=COCO(f"{WL}/test/_annotations.coco.json"); fn2id={im['file_name']:im['id'] for im in coco.loadImgs(coco.getImgIds())}
def gt_wl(fn,h,w):
    iid=fn2id.get(fn)
    if iid is None: return None
    m=np.zeros((h,w),bool)
    for a in coco.loadAnns(coco.getAnnIds(imgIds=iid,catIds=[1])): m|=coco.annToMask(a).astype(bool)
    return int(np.where(m.any(axis=1))[0].max()) if m.sum()>0 else None
sizes={p:Image.open(p).size for p in paths}

print("[1/2] gauge 模型 (water_gauge + reflection) ...")
g=SAM3LoRABatchInference(GCFG,GW,SAM3PT,RES,0.4,DEV)
G={}; R={}
for p in paths:
    w,h=sizes[p]; G[p]=largest_cc(union_mask(g.process_image(p,["water_gauge"]),h,w)); R[p]=union_mask(g.process_image(p,["reflection"]),h,w)
del g; gc.collect(); torch.cuda.empty_cache()
# [#2 扫描] air 权重可经 AIR_W 环境变量覆盖 → 给任意 epoch checkpoint 打 px 分；AIR_CFG 可换扫描config
AIR_W=os.environ.get("AIR_W", f"{ADL}/SAM3_LoRa_Waterline_outputs/best_lora_weights.pt")
AIR_CFG=os.environ.get("AIR_CFG", f"{ADL}/configs/full_lora_config.yaml")
print(f"[2/2] air 模型 (Gauge_Air) ... weights={AIR_W}")
a=SAM3LoRABatchInference(AIR_CFG,AIR_W,SAM3PT,RES,0.4,DEV)
A={}
for p in paths:
    w,h=sizes[p]; A[p]=union_mask(a.process_image(p,["Gauge_Air"]),h,w)
del a; gc.collect(); torch.cuda.empty_cache()

print(f"\n{'图':22s}{'mode':>8s}{'px误':>5s}{'E率%':>6s}{'Econf':>6s}{'Emethod':>15s}{'周期数':>6s}{'端到端':>8s}")
PXE=[]; WLCM=[]; EREL=[]; FULL=[]; rows=[]; CAL=[]; SUBPX=[]
for p in paths:
    w,h=sizes[p]; fn=os.path.basename(p); gm,am,rm=G[p],A[p],R[p]
    r=decide_waterline(gm,am,rm,h,w)   # 亚像素默认关(实测对掩码式GT回归,见报告§3.12)
    if r.get("subpixel"): SUBPX.append(abs(r["waterline_y"]-r["waterline_y_mask"]))
    if not r.get("ok"): print(f"{fn[:26]:26s}  未检出"); continue
    gt=gt_wl(fn,h,w); g_top,g_bot,wl=r["gauge_top"],r["gauge_bottom"],r["waterline_y"]
    gauge_h=max(1,g_bot-g_top); true_cmpp=100.0/gauge_h     # 标准1m尺、整尺可见假设
    cal=calibrate(Image.open(p),gm); E_cmpp=cal["cm_per_px"] if cal.get("ok") else None
    econf=cal.get("conf"); emeth=cal.get("method","-"); eper=cal.get("n_periods")
    read_pred_true=(g_bot-wl)*true_cmpp
    read_gt=(g_bot-gt)*true_cmpp if gt is not None else None
    pxe=abs(wl-gt) if gt is not None else None
    wlcm=abs(read_pred_true-read_gt) if gt is not None else None
    erel=abs(E_cmpp-true_cmpp)/true_cmpp*100 if E_cmpp else None
    read_pred_E=(g_bot-wl)*E_cmpp if E_cmpp else None
    full=abs(read_pred_E-read_gt) if (E_cmpp and gt is not None) else None
    if pxe is not None: PXE.append(pxe)
    if wlcm is not None: WLCM.append(wlcm)
    if erel is not None: EREL.append(erel)
    if full is not None: FULL.append(full)
    CAL.append((fn[:20], erel, econf, cal.get("implied_len_cm"), cal.get("reliable")))
    f2=lambda v,n=2: ("%.*f"%(n,v)) if v is not None else "—"
    print(f"{fn[:22]:22s}{r['mode']:>8s}{(pxe if pxe is not None else -1):5.0f}"
          f"{f2(erel,0):>6s}{(econf if econf else 0):6.2f}{emeth[:14]:>15s}{(eper if eper else 0):6.1f}{f2(full):>8s}")
def stat(a,nm,u):
    a=np.array([x for x in a if x is not None])
    if not len((a)): print(f"  {nm}: 无数据"); return
    print(f"  {nm}: 中位={np.median(a):.2f}{u} 均值={np.mean(a):.2f}{u} 最大={a.max():.2f}{u} (n={len(a)})")
print("\n===== 误差分解 ====="); stat(PXE,"水位线像素误差","px"); stat(WLCM,"①纯水位线→cm(真尺度)","cm")
stat(EREL,"②E字自标定cm/px相对误差","%"); stat(FULL,"③端到端读数误差(自标定×水位线)","cm")
if SUBPX:
    import numpy as _np
    print(f"\n[亚像素] {len(SUBPX)}/{len(PXE)}张'干区下沿'启用亚像素细化; 相对掩码边界平移 中位={_np.median(SUBPX):.2f}px 最大={_np.max(SUBPX):.2f}px (bounded)")
    print("  注:px误差vs整数GT看不出亚像素增益(GT是整数掩码行);此处只确认不引起回归+细化已激活+平移有界。真亚像素增益已在合成尺验证(报告§3.12: 0.6px)")
print("\n判读: ①是分割精度地板; ②是标尺度精度; ③是真实产品路径。瓶颈=三者中最大者。")
print("⚠️ 假设: 标准1m尺且整尺可见(true_cmpp=100/尺高); 005_Clear_Water合成清水域差是已知离群。")
# ---- E标定置信门分析:看"E率失败(谐波)"的图 conf 是否可分 ----
print("\n===== E标定置信门验证(物理合理性: 隐含尺长25~320cm) =====")
print(f"{'图':22s}{'E率%':>6s}{'Econf':>6s}{'隐含尺长cm':>10s}{'reliable门':>10s}")
gated=[]
for fn,erel,econf,ilen,rel in CAL:
    if rel is False: gated.append(fn)
    print(f"{fn:22s}{(erel if erel else 0):6.0f}{(econf if econf else 0):6.2f}{(ilen if ilen else 0):10.1f}{str(rel):>10s}")
print(f"→ 门拦截(reliable=False,弃E回退两点标定): {gated if gated else '无'}")
print("  期望:只拦000150(隐含399cm谐波失败),其余reliable")
print("@@@READDONE@@@")
