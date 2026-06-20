# 反光防御端到端验证：用新 measure()(含反光分支)在17张含reflection标注图上跑，
# 对比 旧干沿误差 vs 新水位线误差 (以尺底为真实水面代理——这些图水尺基本全露,水面≈尺底)。
import sys, types, os, numpy as np
_m=types.ModuleType('decord'); _m.cpu=lambda *a,**k:None; _m.VideoReader=object
_m.bridge=types.SimpleNamespace(set_bridge=lambda *a,**k:None); sys.modules['decord']=_m
from PIL import Image
from pycocotools.coco import COCO
from measure_engine import WaterLevelMeasurerV2, union_mask

BASE="/media/jiale/AI_Local/水尺水位测量系统"; V3=f"{BASE}/02_数据集/water_gauge_v3_coco"
samples=[]
for split in ("train","valid"):
    jp=f"{V3}/{split}/_annotations.coco.json"
    if not os.path.exists(jp): continue
    c=COCO(jp); cats={ci['id']:ci['name'] for ci in c.loadCats(c.getCatIds())}
    refl_ids=[i for i,n in cats.items() if 'refl' in n.lower()]
    for iid in c.getImgIds():
        if c.getAnnIds(imgIds=iid,catIds=refl_ids):
            samples.append((split, c.loadImgs(iid)[0]['file_name']))
M=WaterLevelMeasurerV2()
print(f"\n{'图':30s}{'尺底*':>6s}{'旧干沿':>7s}{'新水线':>7s}{'旧误差':>7s}{'新误差':>7s}{'mode':>22s}")
old_e=[]; new_e=[]
for split,fn in samples:
    p=f"{V3}/{split}/{fn}"
    if not os.path.exists(p): continue
    im=Image.open(p); w,h=im.size
    amask=union_mask(M.air.process_image(p,["Gauge_Air"]),h,w)
    old_y=int(np.where(amask.any(axis=1))[0].max()) if amask.sum()>0 else h
    r=M.measure(p)
    if not r.get("ok"): print(f"{fn[:30]:30s}  未检出"); continue
    gb=r["gauge_bottom"]; ny=r["waterline_y"]
    oe=abs(old_y-gb); ne=abs(ny-gb); old_e.append(oe); new_e.append(ne)
    print(f"{fn[:30]:30s}{gb:6d}{old_y:7d}{ny:7.0f}{oe:7.0f}{ne:7.0f}{r.get('mode',''):>22s}")
def stat(e,n):
    e=np.array(e); print(f"  {n}: 中位={np.median(e):.0f}px 均值={np.mean(e):.0f}px 最大={e.max():.0f}px 离群(>50)={int((e>50).sum())}/{len(e)}")
print("\n----- 误差 vs 尺底(真实水面代理) -----"); stat(old_e,"旧法(干区下沿)"); stat(new_e,"新法(反光防御)")
print("* 这些图水尺基本全露,水面≈尺底; 真正半淹场景需演示水槽实拍验证")
