# -*- coding: utf-8 -*-
"""多帧时域融合 (A3)：把逐帧水位值融合成稳定读数。
=====================================================================
解决直播演示两大问题：①单帧抖动 ②偶发离群帧(反光泄漏等护栏漏网的)；
并且【真实涨/落水必须跟随、不滞后】(演示倒水=缓慢斜坡)。
方法：置信门限 + 滑窗【鲁棒局部线性(Theil-Sen)】 + 残差MAD跳变剔除 + regime阶跃跟随。
设计要点(灵魂)：
  - 不用"窗内中位"做估计——它对【移动信号滞后半个窗】(实测涨水滞后达9.5cm)。
    改用 Theil-Sen 在窗内拟合直线、外推到"当下"取值：平稳时斜率≈0退化为中位(满抗噪)，
    涨水时斜率≈真实涨速、外推到当下≈零滞后；成对斜率取中位→对离群免疫。
  - 跳变判据中心 = 直线在当下的【预测值】(非滞后中位)，故真实斜坡帧不被误判为跳变；
    阈值 = max(mad_k×残差MAD, min_jump)，绝对下限防MAD塌缩误杀微动。
  - ★regime阶跃：连续 regime_confirm 帧朝同侧大跳【且彼此一致】→判真实阶跃突变，清窗接受。
    (斜坡靠Theil-Sen跟随；阶跃靠regime跟随——两类真实变化都不卡屏)
单位无关(px或cm)；线程不安全，每站点一个实例。
=====================================================================
"""
from collections import deque
import numpy as np

class TemporalFusion:
    def __init__(self, window=10, conf_floor=0.4, mad_k=3.5, min_jump=2.0,
                 min_inliers=3, regime_confirm=3, regime_tol=None):
        self.window=window; self.conf_floor=conf_floor; self.mad_k=mad_k
        self.min_jump=min_jump; self.min_inliers=min_inliers
        self.regime_confirm=regime_confirm
        self.regime_tol=regime_tol if regime_tol is not None else 4*min_jump
        self.buf=deque(maxlen=window)      # 入窗内点 (k, value)
        self.pending=[]                    # 连续被拒离群 (k, value)，用于regime识别
        self.k=0                           # 帧计数(含被拒帧，保证时间轴连续)
        self.n_lowconf=0; self.n_jump=0; self.n_accept=0; self.n_regime=0

    @staticmethod
    def _robust(vals):
        v=np.asarray(vals,float); med=float(np.median(v))
        mad=float(np.median(np.abs(v-med)))*1.4826
        return med, mad

    def _fit(self):
        """Theil-Sen 鲁棒直线：成对斜率中位 + 鲁棒截距。返回(slope, intercept)。"""
        pts=list(self.buf); ks=np.array([p[0] for p in pts],float); vs=np.array([p[1] for p in pts],float)
        slopes=[(vs[j]-vs[i])/(ks[j]-ks[i]) for i in range(len(pts)) for j in range(i+1,len(pts)) if ks[j]!=ks[i]]
        slope=float(np.median(slopes)) if slopes else 0.0
        intercept=float(np.median(vs-slope*ks))
        return slope, intercept

    def _predict(self, k):
        if not self.buf: return None
        if len(self.buf)==1: return self.buf[-1][1]
        s,b=self._fit(); return s*k+b

    def _resid_mad(self):
        if len(self.buf)<2: return self._robust([v for _,v in self.buf])[1] if self.buf else 0.0
        s,b=self._fit(); res=np.array([v-(s*k+b) for k,v in self.buf])
        return float(np.median(np.abs(res-np.median(res))))*1.4826

    def fused(self):
        """当前稳定估计 = 鲁棒直线外推到最新帧(零滞后)。"""
        if not self.buf: return None
        return self._predict(self.buf[-1][0])

    def update(self, value, conf=1.0):
        value=float(value); self.k+=1; k=self.k
        # 1) 置信门限
        if conf is not None and conf < self.conf_floor:
            self.n_lowconf+=1; return self._rep(False,"低置信丢弃",value,conf)
        # 2) 跳变剔除(冷启动 min_inliers 前全收)
        if len(self.buf) >= self.min_inliers:
            center=self._predict(k)                      # 直线外推到当下(非滞后中位)
            thr=max(self.mad_k*self._resid_mad(), self.min_jump)
            if abs(value-center) > thr:
                self.pending.append((k,value))
                if len(self.pending) >= self.regime_confirm:
                    pv=[v for _,v in self.pending[-self.regime_confirm:]]
                    if self._robust(pv)[1] <= self.regime_tol:    # 连续几帧彼此一致→真实阶跃
                        self.buf.clear()
                        for kk,vv in self.pending[-self.regime_confirm:]: self.buf.append((kk,vv))
                        self.pending=[]; self.n_regime+=1; self.n_accept+=1
                        return self._rep(True,"regime阶跃:接受新水位",value,conf,center=center,thr=thr)
                self.n_jump+=1
                return self._rep(False,"跳变离群(观察中)",value,conf,center=center,thr=thr)
        # 3) 正常入窗
        self.buf.append((k,value)); self.pending=[]; self.n_accept+=1
        return self._rep(True,"入窗",value,conf)

    def _rep(self, accepted, reason, raw, conf, center=None, thr=None):
        f=self.fused(); spread=self._resid_mad() if self.buf else None
        return {"accepted":accepted,"reason":reason,"raw":round(raw,2),"conf":conf,
                "fused":(round(f,2) if f is not None else None),
                "n_window":len(self.buf),"spread":(round(spread,2) if spread is not None else None),
                "stable":(spread is not None and spread<=self.min_jump),
                "stats":{"accept":self.n_accept,"lowconf":self.n_lowconf,"jump":self.n_jump,"regime":self.n_regime}}


# ---------------- 合成序列自测：抖动+离群+真实涨水(斜坡)+阶跃 ----------------
if __name__=="__main__":
    rng=np.random.RandomState(42)
    truth=[]
    for i in range(60):
        if i<30: truth.append(20.0)
        elif i<45: truth.append(20.0+(45-20)*(i-30)/15.0)   # 缓慢涨水(斜坡)
        else: truth.append(45.0)
    obs=[]; confs=[]
    for i,t in enumerate(truth):
        v=t+rng.randn()*0.4; c=0.95
        if i in (10,18,38): v=t+30.0; c=0.6     # 偶发离群(护栏漏网反光帧)
        if i==25: c=0.2                           # 低置信帧
        obs.append(v); confs.append(c)
    tf=TemporalFusion(window=10, conf_floor=0.4, mad_k=3.5, min_jump=1.5, regime_confirm=3)
    print(f"{'帧':>3s}{'真值':>7s}{'观测':>7s}{'conf':>5s}{'融合':>7s}{'|融-真|':>8s}  判定")
    raw_err=[]; fus_err=[]
    for i,(t,v,c) in enumerate(zip(truth,obs,confs)):
        r=tf.update(v,c); f=r["fused"]
        if f is not None: fus_err.append(abs(f-t))
        raw_err.append(abs(v-t))
        if i<22 or i in (24,25,26) or 35<=i<=46:
            fe=f"{abs(f-t):8.2f}" if f is not None else f"{'—':>8s}"
            print(f"{i:3d}{t:7.1f}{v:7.1f}{c:5.2f}{(f if f is not None else 0):7.1f}{fe}  {r['reason']}")
    raw_err=np.array(raw_err); fus_err=np.array(fus_err)
    print(f"\n原始观测 vs 真值: 中位={np.median(raw_err):.2f}cm 最大={raw_err.max():.2f}cm")
    print(f"时域融合 vs 真值: 中位={np.median(fus_err):.2f}cm 最大={fus_err.max():.2f}cm")
    print(f"统计: {tf.n_accept}入窗 / {tf.n_jump}跳变剔除 / {tf.n_lowconf}低置信丢 / {tf.n_regime}次regime阶跃")
    print("期望: 离群帧(10/18/38)剔除; 第30~45帧涨水被Theil-Sen跟随【低滞后】; 低置信帧(25)丢弃")
