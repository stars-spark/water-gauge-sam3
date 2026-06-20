# 升级版水位测量引擎 (A2+A3)：水尺∩干区约束 + 最大连通域 + 沿尺轴投影 + 亚像素水位线
# 对比旧法(Gauge_Air掩码最底行)在真实测试集上的离群与误差。
import sys, types, os, glob, numpy as np
_m=types.ModuleType('decord'); _m.cpu=lambda *a,**k:None; _m.VideoReader=object
_m.bridge=types.SimpleNamespace(set_bridge=lambda *a,**k:None); sys.modules['decord']=_m
import torch
from PIL import Image
from scipy import ndimage
from pycocotools.coco import COCO
from batch_infer_sam_json import SAM3LoRABatchInference
from subpixel_waterline import subpixel_edge   # §3.12 亚像素细化

BASE="/media/jiale/AI_Local/水尺水位测量系统"
WL=f"{BASE}/02_数据集/waterline_v1_coco"
ADL=f"{BASE}/03_云端SAM3/autodl_training_backup/SAM3_LoRA-main"

def union_mask(res,h,w):
    if res is None: return np.zeros((h,w),bool)
    m=np.zeros((h,w),bool)
    for mk in res['masks']: m|=mk.astype(bool)
    return m

def largest_cc(mask):
    if mask.sum()==0: return mask
    lab,n=ndimage.label(mask)
    if n<=1: return mask
    sizes=ndimage.sum(mask,lab,range(1,n+1))
    return lab==(1+int(np.argmax(sizes)))

def robust_bottom_row(mask, pct=99.5):
    """干区下沿的稳健(亚像素近似)行坐标：最大连通域行坐标的高分位。"""
    ys=np.where(mask.any(axis=1))[0]
    if len(ys)==0: return None
    return float(np.percentile(ys, pct))

import math
def gauge_axis(gmask):
    """水尺掩码PCA主轴。返回 (质心c, 单位主轴axis[指向图像下方], 轴上投影范围t_top/t_bot, 偏离竖直角度deg)。"""
    gy,gx=np.where(gmask)
    pts=np.stack([gx,gy],1).astype(float); c=pts.mean(0)
    _,_,vt=np.linalg.svd(pts-c, full_matrices=False); axis=vt[0]
    if axis[1]<0: axis=-axis                     # 统一指向图像下方(y增大)=水的方向
    gproj=(pts-c)@axis
    ang=abs(math.degrees(math.atan2(axis[0], axis[1])))   # 0=竖直
    return c, axis, float(gproj.min()), float(gproj.max()), ang

def waterline_frac(gmask, g_top, g_bot, wl_y, wl_pixels=None, tilt_thresh=8.0):
    """水位线沿尺归一化位置 (0=尺顶 1=尺底)。
    ★safe-by-construction：近竖直(角度≤阈值)→直接用行坐标比例，与旧行为【逐字节一致】(零回归)；
      倾斜>阈值→沿尺PCA主轴投影【真实水位线像素】(非行坐标,后者在斜尺上是角点已被污染)，消斜拍系统偏差。
      返回 (frac, 是否走了轴投影, 角度deg)。"""
    gauge_h=max(1,g_bot-g_top)
    c,axis,t_top,t_bot,ang=gauge_axis(gmask)
    if ang<=tilt_thresh:                          # 近竖直：旧行为
        return (wl_y-g_top)/gauge_h, False, ang
    if wl_pixels is not None and len(wl_pixels)>=3:
        pr=float(np.median((np.asarray(wl_pixels,float)-c)@axis))   # 投影真实水位线像素带,取中位
    else:
        pr=float((np.array([c[0],wl_y],float)-c)@axis)              # 退化：尺中列+wl_y
    frac=(pr-t_top)/max(1e-6,(t_bot-t_top))
    return float(min(1.0,max(0.0,frac))), True, ang

def decide_waterline(gmask, amask, rmask, h, w, rt=100.0, rb=0.0, gray=None):
    """纯numpy判定(不碰模型)：三掩码→水位线/frac/level/mode。抽出以便单模型/离线测试复用同一逻辑。
    gray: 可选灰度图(HxW)。提供则在"干区下沿"模式对掩码边界做亚像素细化(§3.12)。"""
    if gmask.sum()==0: return {"ok":False,"msg":"未检出水尺"}
    gy=np.where(gmask.any(axis=1))[0]; g_top,g_bot=int(gy.min()),int(gy.max())
    gauge_h=max(1,g_bot-g_top)
    a_bot = float(np.where(amask.any(axis=1))[0].max()) if amask.sum()>0 else None
    air_cov = amask.sum()/float(h*w)
    refl_cov = rmask.sum()/float(h*w)
    # 反光真伪判别：真倒影只在尺下部、不覆盖上部干尺(干净图reflection prompt会把整尺误当倒影)
    REFL=False; refl_top=None
    if rmask.sum()>0:
        r_rows=np.where(rmask.any(axis=1))[0]; refl_top=float(r_rows.min())
        mid=int(g_top+0.5*gauge_h)
        frac_upper=float(rmask[:mid,:].sum())/float(rmask.sum())
        REFL=(refl_cov>0.003 and frac_upper<0.15 and refl_top>g_top+0.3*gauge_h)
    # 决策：默认干区下沿(精确);干沿越尺底>5%尺高=泄漏退尺底(物理:干区⊆水尺);半淹+真倒影用倒影顶沿
    if a_bot is None:
        wl_y=float(g_bot); conf=0.4; mode="无干区:退尺底"
    elif air_cov > 0.25 and a_bot > g_bot + 0.15*gauge_h:
        # 过分割护栏(V2已验证,中位6px/0离群)：仅当干区掩码气球化(占比>25%)且明显越尺底才退尺底。
        # ⚠️2026-06-18审片曾试更激进的"越尺底>5%即退尺底",AutoDL实测回归000003(10→41px,因水尺欠分割),已回退。
        if REFL and refl_top < g_bot - 0.10*gauge_h:
            wl_y=refl_top; conf=0.6; mode="反光半淹:倒影顶沿"
        else:
            wl_y=float(g_bot); conf=0.5; mode="过分割护栏:退尺底"
    else:
        wl_y=a_bot; conf=round(1.0-min(1.0,abs(a_bot-g_bot)/gauge_h),3); mode="干区下沿"
    # ★亚像素细化(§3.12,model-free,合成验证0.6px)：仅"干区下沿"精确模式+提供灰度图时,
    #   在掩码边界±K窗口对【沿尺宽平均】灰度剖面做亚像素边缘,绕开二值掩码整数粒度。bounded防跑偏。
    wl_y_mask=float(wl_y); subpixel_used=False
    if gray is not None and mode=="干区下沿":
        K=max(4,int(0.012*gauge_h)); gx=np.where(gmask.any(axis=0))[0]
        if len(gx)>4:
            x0,x1=int(gx.min()),int(gx.max())
            r0=max(0,int(round(wl_y))-K); r1=min(h,int(round(wl_y))+K+1)
            if r1-r0>=5 and x1>x0:
                prof=np.asarray(gray)[r0:r1,x0:x1+1].astype(float).mean(axis=1)  # 全宽平均压E纹理
                sub=r0+subpixel_edge(prof,smooth=1.0)
                if abs(sub-wl_y)<=K: wl_y=float(sub); subpixel_used=True            # bounded:不越窗口才采纳
    # 抗斜拍：取对应掩码wl_y附近边界带作真实水位线像素(近竖直时waterline_frac内部忽略)
    src = rmask if mode.startswith("反光") else (amask if mode=="干区下沿" else gmask)
    _sy,_sx = np.where(src); _half=max(2,int(0.01*gauge_h))
    _sel = np.abs(_sy-wl_y)<=_half
    wl_pixels = np.stack([_sx[_sel],_sy[_sel]],1) if _sel.any() else None
    frac, tilt_handled, tilt_deg = waterline_frac(gmask, g_top, g_bot, wl_y, wl_pixels)
    level=rb+(1-frac)*(rt-rb)
    # ★输出合法性门(2026-06-20)：水位线必须落在水尺范围内(frac∈[0,1]±5%容差)。越界=gauge检测碎片化
    #   /水位线越尺,读数无物理意义→标unreliable触发重采(不clamp硬报,避免"读错还上报")。
    #   全集批量验证(v3_水位读数_全集.csv):拦住越界灾难7/9且0误杀正常图(含合法"略越尺底-1.8cm")。
    LEGAL_TOL=0.05
    reliable=bool(-LEGAL_TOL <= frac <= 1.0+LEGAL_TOL)
    reason="" if reliable else f"读数越界(frac={frac:.2f},level={level:.0f}cm):水位线落在水尺外,疑gauge碎片化→建议重采"
    return {"ok":True,"reliable":reliable,"reason":reason,
            "frac":round(frac,4),"waterline_y":round(wl_y,1),"level_cm":round(level,1),
            "conf":conf,"gauge_top":g_top,"gauge_bottom":g_bot,"mode":mode,
            "tilt_deg":round(tilt_deg,1),"tilt_handled":tilt_handled,
            "refl_top":(round(refl_top,1) if refl_top is not None else None),
            "refl_cov":round(refl_cov,4),
            "waterline_y_mask":round(wl_y_mask,1),"subpixel":subpixel_used}


class WaterLevelMeasurerV2:
    def __init__(self, gauge_top_cm=100.0, gauge_bottom_cm=0.0, device="cuda"):
        self.rt,self.rb=gauge_top_cm,gauge_bottom_cm
        self.gauge=SAM3LoRABatchInference("configs/full_lora_config.yaml",
            "SAM3_LoRa_outputs/best_lora_weights.pt","checkpoints/sam3.pt",1008,0.4,device)
        # 2026-06-20 部署 V3_clean ep05(encoder LoRA rank4)：clean test 中位3px(原decoder-only部署6px),
        #   epoch4-8稳定+视觉审片过关(05_演示/v3_clean_audit.png)。⚠️候选权重,演示水槽真值确认后定;
        #   回退到原decoder权重 = 设环境变量:
        #   AIR_CFG={ADL}/configs/full_lora_config.yaml  AIR_W={ADL}/SAM3_LoRa_Waterline_outputs/best_lora_weights.pt
        _air_cfg=os.environ.get("AIR_CFG","checkpoints/waterline_v3_clean/v3_encoder_clean.yaml")
        _air_w=os.environ.get("AIR_W","checkpoints/waterline_v3_clean/epoch05_lora_weights.pt")
        self.air=SAM3LoRABatchInference(_air_cfg, _air_w,"checkpoints/sam3.pt",1008,0.4,device)
    def _gauge_air_prior(self, path, amask, h, w):
        """air自先验:用干区(V3)bbox裁掉远处背景→在ROI上重跑gauge→映射回全图。救gauge背景误检。"""
        import tempfile
        ys=np.where(amask.any(axis=1))[0]; xs=np.where(amask.any(axis=0))[0]
        if len(ys)==0 or len(xs)==0: return None
        a0,a1=int(ys.min()),int(ys.max()); ah=max(1,a1-a0)
        x0,x1=int(xs.min()),int(xs.max()); aw=max(1,x1-x0)
        cy0=max(0,int(a0-0.2*ah)); cy1=min(h,int(a1+0.6*ah))   # 下方留尺以纳入水下段,但裁掉远处背景
        cx0=max(0,int(x0-0.4*aw)); cx1=min(w,int(x1+0.4*aw))
        crop=Image.open(path).convert("RGB").crop((cx0,cy0,cx1,cy1)); ch,cw=cy1-cy0,cx1-cx0
        with tempfile.NamedTemporaryFile(suffix=".jpg",delete=False) as f: crop.save(f.name); cp=f.name
        try: g=largest_cc(union_mask(self.gauge.process_image(cp,["water_gauge"]),ch,cw))
        finally: os.unlink(cp)
        if g.sum()==0: return None, False
        # 过冲检测(尺度复核):救出的尺体下沿触碰裁剪下边界=尺体掩膜跑进背景,下沿(=尺度分母)不可信
        gy=np.where(g.any(axis=1))[0]; m=max(3,int(0.03*ch))
        overshoot = bool(gy.max() >= ch-m and cy1<h)   # 撞到裁剪下边界,且该边界不是图像真实下沿
        full=np.zeros((h,w),bool); full[cy0:cy1,cx0:cx1]=g
        return full, overshoot

    def measure(self, path):
        im=Image.open(path); w,h=im.size
        gmask=largest_cc(union_mask(self.gauge.process_image(path,["water_gauge"]),h,w))
        amask=union_mask(self.air.process_image(path,["Gauge_Air"]),h,w)  # 干区下沿用原始union(精确)，不做largest_cc以免拆裂丢失下沿
        rmask=union_mask(self.gauge.process_image(path,["reflection"]),h,w)  # ★倒影掩码(gauge LoRA自带reflection类)
        # ★air自先验救援(2026-06-20)：gauge误检到背景时(与干区严重不重叠),用可靠的V3干区位置裁掉背景重跑gauge。
        #   只在失败时触发(正常图overlap≈1零开销);本地实测8张灾难救回6/8(纯算法,不依赖YOLO)。
        rec_overshoot=False
        if os.environ.get("AIR_PRIOR","1")=="1" and amask.sum()>0 and gmask.sum()>0:
            ov=(gmask & amask).sum()/float(amask.sum())
            if ov < float(os.environ.get("AIR_PRIOR_THRESH","0.3")):   # 干区大部分不落在尺体内 → gauge误检到背景
                res=self._gauge_air_prior(path, amask, h, w)
                if res is not None:
                    g2,overshoot=res
                    if g2 is not None and g2.sum()>0 and (g2 & amask).sum()/float(amask.sum()) > ov:
                        gmask=g2; rec_overshoot=overshoot
        # ⚠️亚像素细化(gray参数)默认【关闭】：2026-06-19 AutoDL实测对"掩码式GT"回归(6→10px)——
        #   SAM掩码已拟合标注GT,亚像素推向图像强度边缘反而偏离。需独立实测亚像素物理真值才有意义(外业)。
        #   能力保留在decide_waterline(gray=...),有亚像素真值时再启用。
        r=decide_waterline(gmask, amask, rmask, h, w, self.rt, self.rb)
        # 救援后尺度复核：救出的尺体下沿撞裁剪边界(过冲)→下沿即尺度分母不可信,降级重采(防"自信但不准")
        if rec_overshoot and r.get("ok"):
            r["reliable"]=False; r["reason"]="air救援后尺体下沿触碰裁剪边界(尺度不可信)→重采"; r["recovery"]="overshoot"
        return r

# ---------------- 对比验证：旧法 vs 新法 ----------------
if __name__=="__main__":
    c=COCO(f"{WL}/test/_annotations.coco.json"); fn2id={im['file_name']:im['id'] for im in c.loadImgs(c.getImgIds())}
    def gtwl(fn,h,w):
        iid=fn2id.get(fn); m=np.zeros((h,w),bool)
        if iid is None: return None
        for a in c.loadAnns(c.getAnnIds(imgIds=iid,catIds=[1])): m|=c.annToMask(a).astype(bool)
        return int(np.where(m.any(axis=1))[0].max()) if m.sum()>0 else None
    M=WaterLevelMeasurerV2()
    reals=sorted(glob.glob(f"{WL}/test/water_gauge_*.jpg"))
    old_err=[]; new_err=[]
    print(f"\n{'图':24s}{'GT_y':>6s}{'旧法y':>7s}{'新法y':>7s}{'旧误差':>7s}{'新误差':>7s}{'conf':>6s}")
    for p in reals:
        im=Image.open(p); w,h=im.size; fn=os.path.basename(p); g=gtwl(fn,h,w)
        if g is None: continue
        # 旧法：Gauge_Air 掩码最底行(不做交集/连通域)
        amask=union_mask(M.air.process_image(p,["Gauge_Air"]),h,w)
        old_y=int(np.where(amask.any(axis=1))[0].max()) if amask.sum()>0 else h
        r=M.measure(p); new_y=r.get("waterline_y") or h
        oe=abs(old_y-g); ne=abs(new_y-g); old_err.append(oe); new_err.append(ne)
        print(f"{fn[:24]:24s}{g:6d}{old_y:7d}{int(new_y):7d}{oe:7d}{int(ne):7d}{r.get('conf',0):6.2f}  {r.get('mode','')}")
    def stat(e,name):
        e=np.array(e); out=np.sum(e>50)
        print(f"  {name}: 中位={np.median(e):.0f}px 均值={np.mean(e):.0f}px 最大={e.max():.0f}px 离群(>50px)={out}/{len(e)}")
    print("\n----- 水位线像素误差(vs GT) -----"); stat(old_err,"旧法(Gauge_Air最底行)"); stat(new_err,"新法(∩水尺+连通域+沿轴)")
    # 折算cm(按标准1米尺，用GT尺高近似)：仅示意改进幅度
    print("\n(注：像素误差×每像素cm即水位误差；新法重点在压离群、提稳健性)")
