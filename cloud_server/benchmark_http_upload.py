"""
HTTP 图像上传速率基准测试
================================================
测量不同 JPEG 质量档（=不同文件大小）下的 HTTP POST 传输性能。

用法（两个终端）：
  终端1：python pc_http_image_server.py          # 先启动服务（模拟模式即可）
  终端2：python benchmark_http_upload.py          # 运行本脚本

可调参数（见下方 CONFIG）：
  QUALITY_LEVELS  — 要测试的 JPEG 质量档
  REPEAT          — 每档重复发送次数（取平均）
  SERVER_URL      — 服务器地址
  IMAGE_DIR       — 测试图源目录
"""

import os, io, time, statistics, glob
import requests
from PIL import Image

# ─── 配置 ───────────────────────────────────────────────────────────────────
QUALITY_LEVELS = [20, 40, 60, 75, 85, 92, 95]   # JPEG 质量档（越大越清晰/越大）
REPEAT         = 5                                # 每档每张图重复次数
SERVER_URL     = "http://127.0.0.1:8000/measure" # 目标服务器（先跑 pc_http_image_server.py）
IMAGE_DIR      = os.path.join(os.path.dirname(__file__), "received_images")
# ────────────────────────────────────────────────────────────────────────────


def compress_jpeg(pil_img: Image.Image, quality: int) -> bytes:
    buf = io.BytesIO()
    pil_img.save(buf, format="JPEG", quality=quality, optimize=True)
    return buf.getvalue()


def post_image(jpeg_bytes: bytes) -> tuple[float, int]:
    """返回 (耗时秒, HTTP状态码)"""
    t0 = time.perf_counter()
    resp = requests.post(
        SERVER_URL,
        data=jpeg_bytes,
        headers={"Content-Type": "image/jpeg", "X-Station": "BENCH", "X-Conf": "0.99"},
        timeout=15,
    )
    dt = time.perf_counter() - t0
    return dt, resp.status_code


def check_server():
    try:
        r = requests.get(SERVER_URL.replace("/measure", "/health"), timeout=3)
        return r.status_code == 200
    except Exception:
        return False


def main():
    # 1. 检查服务器
    if not check_server():
        print("[错误] 服务器未响应，请先在另一个终端运行：")
        print("       python pc_http_image_server.py")
        return

    # 2. 加载测试图
    jpg_paths = sorted(glob.glob(os.path.join(IMAGE_DIR, "*.jpg")))
    if not jpg_paths:
        print(f"[错误] 未在 {IMAGE_DIR} 找到 .jpg 图片")
        return
    pil_imgs = [Image.open(p).convert("RGB") for p in jpg_paths]
    print(f"[信息] 加载 {len(pil_imgs)} 张图  REPEAT={REPEAT}  共 {len(QUALITY_LEVELS)} 个质量档\n")

    # 3. 逐档测试
    results = []
    for q in QUALITY_LEVELS:
        sizes_kb, times_ms = [], []
        for img in pil_imgs:
            jpeg_bytes = compress_jpeg(img, q)
            for _ in range(REPEAT):
                dt, status = post_image(jpeg_bytes)
                if status == 200:
                    sizes_kb.append(len(jpeg_bytes) / 1024)
                    times_ms.append(dt * 1000)
                else:
                    print(f"  [警告] HTTP {status}，跳过本次")

        if not times_ms:
            continue

        avg_kb    = statistics.mean(sizes_kb)
        avg_ms    = statistics.mean(times_ms)
        med_ms    = statistics.median(times_ms)
        std_ms    = statistics.stdev(times_ms) if len(times_ms) > 1 else 0
        throughput = avg_kb / (avg_ms / 1000)  # KB/s

        results.append((q, avg_kb, avg_ms, med_ms, std_ms, throughput))
        print(f"  质量={q:3d}  大小={avg_kb:6.1f}KB  "
              f"均值={avg_ms:6.1f}ms  中位={med_ms:6.1f}ms  "
              f"σ={std_ms:5.1f}ms  吞吐={throughput:.0f}KB/s")

    # 4. 汇总表格
    print("\n" + "=" * 75)
    print(f"{'质量':>4}  {'大小(KB)':>8}  {'均值RTT(ms)':>11}  {'中位RTT(ms)':>11}  "
          f"{'σ(ms)':>6}  {'吞吐(KB/s)':>10}")
    print("-" * 75)
    for q, kb, avg, med, std, tp in results:
        print(f"{q:4d}  {kb:8.1f}  {avg:11.1f}  {med:11.1f}  {std:6.1f}  {tp:10.0f}")
    print("=" * 75)

    # 5. 给出建议
    if results:
        # 找 RTT < 300ms 且吞吐最高的档
        candidates = [(q, kb, avg, tp) for q, kb, avg, med, std, tp in results if avg < 300]
        if candidates:
            best = max(candidates, key=lambda x: x[3])
            print(f"\n[建议] 质量档 {best[0]} 在 RTT<300ms 约束下吞吐最高：")
            print(f"       大小≈{best[1]:.0f}KB  RTT≈{best[2]:.0f}ms  吞吐≈{best[3]:.0f}KB/s")
        else:
            print("\n[注意] 所有档均超过300ms，网络延迟偏高，建议排查连接")


if __name__ == "__main__":
    main()
