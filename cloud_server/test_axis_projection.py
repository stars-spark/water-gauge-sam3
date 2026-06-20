# -*- coding: utf-8 -*-
"""沿尺轴投影抗倾斜——合成几何验证(纯numpy,无需SAM3/GPU)。
构造角度已知的水尺掩码+已知水位比例f，对比：
  旧法(行坐标比例) vs 新法(waterline_frac轴投影) vs 真值f。
证明：竖直θ=0时新法==旧法(零回归)；倾斜时旧法系统偏差、新法仍准。
用法：PYTHONPATH=. /home/jiale/sam3_test_venv/bin/python test_axis_projection.py
"""
import sys, types, numpy as np, math
_m=types.ModuleType('decord'); _m.cpu=lambda *a,**k:None; _m.VideoReader=object
_m.bridge=types.SimpleNamespace(set_bridge=lambda *a,**k:None); sys.modules['decord']=_m
from measure_engine import waterline_frac

def make(theta_deg, f, H=900, W=900, L=640, Wd=44):
    th=math.radians(theta_deg)
    d=np.array([math.sin(th), math.cos(th)])          # 主轴(指向下方),(x,y)
    perp=np.array([math.cos(th), -math.sin(th)])
    c=np.array([W/2.0, H/2.0])
    ys,xs=np.mgrid[0:H,0:W]; pts=np.stack([xs.ravel(),ys.ravel()],1).astype(float)
    rel=pts-c; along=rel@d; perpd=rel@perp
    gauge=(np.abs(along)<=L/2)&(np.abs(perpd)<=Wd/2)
    dry=gauge&(along<=(f-0.5)*L)                       # 从尺顶(along=-L/2)到f处=露出水面(干区)
    return gauge.reshape(H,W), dry.reshape(H,W)

print(f"{'倾角°':>6s}{'真值f':>7s}{'旧法(行)':>9s}{'新法(轴)':>9s}{'旧误差':>8s}{'新误差':>8s}{'轴投影?':>8s}")
for theta in (0,5,10,20,35):
    f=0.62
    gmask,dmask=make(theta,f)
    gy=np.where(gmask.any(axis=1))[0]; g_top,g_bot=int(gy.min()),int(gy.max())
    dy=np.where(dmask.any(axis=1))[0]; wl_y=int(dy.max())        # 干区最低行(measure里的a_bot)
    # 真实水位线像素带(measure里同款取法)
    sy,sx=np.where(dmask); half=max(2,int(0.01*(g_bot-g_top)))
    sel=np.abs(sy-wl_y)<=half; wl_pixels=np.stack([sx[sel],sy[sel]],1)
    row_frac=(wl_y-g_top)/(g_bot-g_top)                          # 旧法
    ax_frac,tilted,ang=waterline_frac(gmask,g_top,g_bot,wl_y,wl_pixels)  # 新法
    print(f"{theta:6d}{f:7.2f}{row_frac:9.3f}{ax_frac:9.3f}{abs(row_frac-f):8.3f}{abs(ax_frac-f):8.3f}{str(tilted):>8s}")
print("\n判读: θ=0 旧法==新法(零回归); θ增大旧法误差线性增大(斜拍系统偏差), 新法误差恒小≈0")
print("折算: 1米水尺上 frac误差0.05 = 5cm水位误差 —— 斜拍不校正的代价")
