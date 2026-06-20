# -*- coding: utf-8 -*-
"""排涝泵自动控制（执行闭环的云端决策端）。

防泵抖动三层防线：
  1) 滞回带：on_cm(=danger) 开泵，回落到 off_cm(=danger-3) 才关；
  2) 连续确认：连续 confirm_n 个样本越线/回落才动作（抗单帧测量噪声）；
  3) 最短驻留：开泵至少 min_on_s、关泵至少 min_off_s（抗水面波动反复横跳）。

publish_fn(topic:str, payload:dict) 由调用方注入（paho-mqtt 包装），
本模块不直接依赖 MQTT 库，便于单测。
"""
import time


class PumpController:
    def __init__(self, publish_fn, on_cm, off_cm,
                 confirm_n=2, min_on_s=15.0, min_off_s=10.0,
                 cmd_topic="waterguage/flume/cmd", target="out",
                 ack_timeout_s=5.0):
        assert off_cm < on_cm, "滞回带要求 off_cm < on_cm"
        self.publish = publish_fn
        self.on_cm, self.off_cm = on_cm, off_cm
        self.confirm_n = confirm_n
        self.min_on_s, self.min_off_s = min_on_s, min_off_s
        self.cmd_topic = cmd_topic
        self.target = target
        self.ack_timeout_s = ack_timeout_s

        self.state = 0                 # 0=泵关 1=泵开（按已发指令的期望状态）
        self._above = 0                # 连续越线计数
        self._below = 0                # 连续回落计数
        self._last_switch = 0.0
        self._pending_cmd = None       # 等ACK的 (cmd_id, payload, sent_ts)，超时重发一次
        self._resent = False

    # ---- 内部 ----
    def _send(self, state, reason, ts):
        cmd_id = int(ts * 1000)
        payload = {"action": "pump", "target": self.target,
                   "state": state, "reason": reason, "cmd_id": cmd_id}
        self.publish(self.cmd_topic, payload)
        self.state = state
        self._last_switch = ts
        self._pending_cmd = (cmd_id, payload, ts)
        self._resent = False
        return "pump_on" if state else "pump_off"

    # ---- 对外 ----
    def update(self, station, level_cm, ts=None):
        """每个测量样本喂一次（只喂控制对象站，如 S02）。
        返回 None | "pump_on" | "pump_off"（本次是否触发了动作）。"""
        ts = ts if ts is not None else time.time()
        # ACK 超时重发（最多一次，避免风暴）
        if self._pending_cmd and not self._resent:
            cmd_id, payload, sent = self._pending_cmd
            if ts - sent > self.ack_timeout_s:
                self.publish(self.cmd_topic, payload)
                self._resent = True

        if level_cm is None:
            return None
        if level_cm >= self.on_cm:
            self._above += 1
            self._below = 0
        elif level_cm <= self.off_cm:
            self._below += 1
            self._above = 0
        else:                          # 滞回带内：清计数，保持现状
            self._above = self._below = 0
            return None

        if (self.state == 0 and self._above >= self.confirm_n
                and ts - self._last_switch >= self.min_off_s):
            self._above = 0
            return self._send(1, "%s danger %.1fcm" % (station, level_cm), ts)
        if (self.state == 1 and self._below >= self.confirm_n
                and ts - self._last_switch >= self.min_on_s):
            self._below = 0
            return self._send(0, "%s 回落 %.1fcm" % (station, level_cm), ts)
        return None

    def on_ack(self, payload):
        """收到边缘 waterguage/flume/ack 时调用。"""
        if self._pending_cmd and payload.get("cmd_id") == self._pending_cmd[0]:
            self._pending_cmd = None


if __name__ == "__main__":
    # 自测：滞回+确认+驻留全路径
    # 时序：5s/样本。i=2(ts=+10)连续2次越线→开泵；i=5(ts=+25)连续2次回落
    # 且开泵驻留15s已满足→关泵。
    sent = []
    ctl = PumpController(lambda t_, p: sent.append(p), on_cm=16.0, off_cm=13.0,
                         confirm_n=2, min_on_s=15, min_off_s=10)
    t = 1000.0
    expects = [None, None, "pump_on", None, None, "pump_off"]
    levels  = [12.0, 16.5, 16.6, 17.0, 13.0, 12.5]
    for i, (lv, expect) in enumerate(zip(levels, expects)):
        act = ctl.update("S02", lv, ts=t + i * 5)
        assert act == expect, "step%d: got %r want %r" % (i, act, expect)
        if act:
            ctl.on_ack({"cmd_id": sent[-1]["cmd_id"]})   # 模拟边缘及时回ACK
    assert sent[0]["state"] == 1 and sent[1]["state"] == 0
    assert ctl._pending_cmd is None

    # ACK 超时重发路径：不回ACK，>5s 后重发同一指令（且只重发一次）
    sent2 = []
    ctl2 = PumpController(lambda t_, p: sent2.append(p), on_cm=16.0, off_cm=13.0,
                          confirm_n=1, min_on_s=15, min_off_s=10)
    ctl2.update("S02", 17.0, ts=2000.0)                  # 开泵
    ctl2.update("S02", 17.1, ts=2006.0)                  # 超时→重发
    ctl2.update("S02", 17.2, ts=2012.0)                  # 不再重发
    assert len(sent2) == 2 and sent2[0]["cmd_id"] == sent2[1]["cmd_id"]
    # 滞回带内不动作
    assert ctl.update("S02", 14.5, ts=t + 100) is None
    print("PumpController 自测通过 ✓  指令流:", [(p["state"], p["reason"]) for p in sent])
