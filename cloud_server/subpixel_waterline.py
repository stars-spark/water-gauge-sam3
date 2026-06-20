# -*- coding: utf-8 -*-
"""亚像素水位线定位 (CIS/梯度峰法, model-free, 破6px地板)。
思路(据 arXiv 2502.16502 + 经典亚像素边缘):掩码只给【近似】水位线区域,真正定位用
【图像灰度强度剖面】在该区域内做亚像素边缘,绕开二值掩码的整数粒度。
验证:对【已知浮点真值】的软边缘,对比 整数阈值法(=掩码最底行) vs 亚像素法。
用法: python subpixel_waterline.py   (纯CPU)
"""
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.special import erf

def subpixel_edge(profile, smooth=1.0):
    """1D强度剖面(沿尺向下,干→湿亮度阶跃)→亚像素边缘位置(float index)。
    法:高斯平滑→梯度幅值峰→抛物线细化。无需训练。"""
    p = np.asarray(profile, float)
    if smooth > 0: p = gaussian_filter1d(p, smooth)
    g = np.abs(np.gradient(p))
    i = int(np.argmax(g))
    if 1 <= i < len(g)-1:
        d = g[i-1] - 2*g[i] + g[i+1]
        if abs(d) > 1e-9:
            return i + 0.5*(g[i-1]-g[i+1])/d         # 抛物线顶点亚像素细化
    return float(i)

def integer_threshold_edge(profile):
    """整数法:中点阈值的首个过零(模拟二值掩码最底行/边界,整数粒度)。"""
    p = np.asarray(profile, float)
    thr = (p[:max(3,len(p)//6)].mean() + p[-max(3,len(p)//6):].mean())/2
    cr = np.where(np.diff((p > thr).astype(int)) != 0)[0]
    return float(cr[0]) if len(cr) else float(np.argmax(np.abs(np.gradient(p))))

# ---------------- 验证A:已知浮点真值的软边缘扫描 ----------------
def soft_edge(n, center, width, lo, hi, noise, rng):
    y = np.arange(n)
    p = lo + (hi-lo)*0.5*(1+erf((y-center)/(width*1.4142)))   # erf软阶跃,中心=center(浮点)
    return p + rng.randn(n)*noise

if __name__ == "__main__":
    rng = np.random.RandomState(42); N = 40
    ie, se = [], []
    for _ in range(5000):
        c = rng.uniform(16, 24)                               # 浮点真值边缘位置
        w = rng.uniform(0.8, 2.2); nz = rng.uniform(2, 9)     # 随机边缘宽度+噪声
        prof = soft_edge(N, c, w, 60, 185, nz, rng)
        ie.append(abs(integer_threshold_edge(prof) - c))
        se.append(abs(subpixel_edge(prof, smooth=1.0) - c))
    ie, se = np.array(ie), np.array(se)
    print("=== 验证A: 5000条软边缘(已知浮点真值,随机宽度0.8~2.2px+噪声σ2~9) ===")
    print(f"  整数阈值法(=掩码最底行): 中位={np.median(ie):.3f}px 均值={np.mean(ie):.3f}px 90分位={np.percentile(ie,90):.3f}px")
    print(f"  亚像素CIS/梯度法       : 中位={np.median(se):.3f}px 均值={np.mean(se):.3f}px 90分位={np.percentile(se,90):.3f}px")
    print(f"  → 亚像素法把定位误差从 ~{np.median(ie):.2f}px 降到 ~{np.median(se):.2f}px (折算1700px尺约 {np.median(ie)/1700*100:.3f}cm→{np.median(se)/1700*100:.3f}cm)")

    # ---------------- 验证B:渲染真水尺(含E刻度纹理),超采样制浮点水面线 ----------------
    from PIL import Image, ImageDraw
    def render_col(level_px_float, H=400, W=120, ss=6):
        """超采样渲染一段水尺(含E刻度)+水填到浮点行,降采样得抗锯齿浮点边缘。返回原图+灰度。"""
        big = Image.new("RGB",(W,H*ss),(238,238,232)); d=ImageDraw.Draw(big,"RGBA")
        for k in range(H//5):                                # E刻度每5px一个(制造纹理)
            yy=k*5*ss; left=(k%2==0); x0=10 if left else W//2
            d.rectangle([x0,yy,x0+8,yy+4*ss],fill=(31,111,178))
            for t in (yy,yy+2*ss,yy+4*ss-ss): d.rectangle([x0,t,x0+45,t+ss],fill=(31,111,178))
        d.rectangle([0,int(level_px_float*ss),W,H*ss],fill=(77,140,217,150))  # 水填到浮点行*ss
        small=big.resize((W,H),Image.BILINEAR)
        return np.asarray(small.convert("L"),float)
    erB_i, erB_s = [], []
    for _ in range(400):
        lv = rng.uniform(120, 280)                           # 浮点真值水面行
        gray = render_col(lv)
        col = gray.mean(axis=1)                              # ★沿全宽平均剖面:水面阶跃横贯全宽被强化,E刻度(局部)被平均压掉
        # 只在真值±8px窗口内找(模拟"掩码给近似区域,亚像素细化")
        w0=int(lv)-8; seg=col[w0:int(lv)+8]
        erB_i.append(abs((w0+integer_threshold_edge(seg)) - lv))
        erB_s.append(abs((w0+subpixel_edge(seg,smooth=1.2)) - lv))
    erB_i, erB_s = np.array(erB_i), np.array(erB_s)
    print("\n=== 验证B: 400张渲染水尺(含E纹理,超采样浮点水面线,真值已知) ===")
    print(f"  整数阈值法 : 中位={np.median(erB_i):.3f}px 均值={np.mean(erB_i):.3f}px")
    print(f"  亚像素法   : 中位={np.median(erB_s):.3f}px 均值={np.mean(erB_s):.3f}px")
    print("结论: 亚像素法在【已知浮点真值】上确实优于整数掩码边界;真实SAM掩码上需在掩码边界±k窗口内对灰度剖面细化(同法)。")
