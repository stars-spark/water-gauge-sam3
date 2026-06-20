# 文件名: local_test.py
import requests
import os

URL = "http://127.0.0.1:8000/predict"
# 确保你在 data 文件夹里放了一张图片
TEST_IMAGE = "data/test_01.jpg" 

def run():
    if not os.path.exists(TEST_IMAGE):
        print(f"❌ 找不到测试图片，请在 data 文件夹里放一张 {TEST_IMAGE}")
        return

    print(f"📤 发送图片: {TEST_IMAGE} ...")
    try:
        with open(TEST_IMAGE, "rb") as f:
            resp = requests.post(URL, files={"file": f})
        print("📥 服务器返回:", resp.json())
    except Exception as e:
        print("❌ 连接失败:", e)

if __name__ == "__main__":
    run()