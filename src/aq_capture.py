# -*- coding: utf-8 -*-
"""系统输出音频采集（WASAPI Loopback）。

合规要点：本模块只读取操作系统混音器输出，不接触游戏进程、不注入、不截图。
"""

import queue
import threading

import numpy as np
import soundcard as sc

DEFAULT_SR = 48000
DEFAULT_BLOCK = 2400   # 50ms @ 48kHz


def loopback_microphones():
    """返回所有 loopback 端点（即各播放设备的镜像采集口）。"""
    return [m for m in sc.all_microphones(include_loopback=True)
            if getattr(m, "isloopback", False)]


def list_devices():
    """列出可选 loopback 设备，返回 [(名称, 是否默认扬声器, 是否默认输入设备)]。"""
    mics = loopback_microphones()
    default_spk = ""
    default_mic = ""
    try:
        spk = sc.default_speaker()
        default_spk = str(spk.name) if spk else ""
    except Exception:
        pass
    try:
        mic = sc.default_microphone()
        default_mic = str(mic.name) if mic else ""
    except Exception:
        pass
    out = []
    for m in mics:
        name = str(m.name)
        out.append((name, name == default_spk, name == default_mic))
    return out


def resolve(device_name=None):
    """解析 loopback 设备对象。

    device_name 为空时优先取默认扬声器对应的 loopback；找不到则取第一个。
    """
    mics = loopback_microphones()
    if not mics:
        raise RuntimeError("未找到任何 loopback 设备：请确认声卡驱动正常、"
                           "Windows 音频服务正在运行，且默认扬声器已启用")
    if device_name:
        for m in mics:
            if str(m.name) == device_name:
                return m
        for m in mics:
            if device_name in str(m.name):
                return m
        raise RuntimeError("未找到匹配的 loopback 设备：%s" % device_name)
    try:
        spk = sc.default_speaker()
        if spk:
            for m in mics:
                if str(m.name) == str(spk.name):
                    return m
    except Exception:
        pass
    return mics[0]


class LoopbackCapture(object):
    """后台线程持续采集系统输出音频，按块投入队列。"""

    def __init__(self, device_name=None, samplerate=DEFAULT_SR,
                 blocksize=DEFAULT_BLOCK):
        self.device_name = device_name
        self.samplerate = int(samplerate)
        self.blocksize = int(blocksize)
        self.device = None
        self.queue = queue.Queue(maxsize=400)
        self.error = None
        self.total_frames = 0
        self._stop = threading.Event()
        self._thread = None

    @property
    def running(self):
        return self._thread is not None and self._thread.is_alive()

    def start(self):
        self.device = resolve(self.device_name)
        self.error = None
        self.total_frames = 0
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="aq-capture")
        self._thread.daemon = True
        self._thread.start()
        return self

    def _run(self):
        try:
            with self.device.recorder(samplerate=self.samplerate,
                                      blocksize=self.blocksize) as rec:
                while not self._stop.is_set():
                    data = rec.record(numframes=self.blocksize)
                    data = np.asarray(data, dtype=np.float32)
                    if data.size == 0:
                        continue
                    self.total_frames += data.shape[0]
                    try:
                        self.queue.put_nowait(data)
                    except queue.Full:
                        try:
                            self.queue.get_nowait()
                            self.queue.put_nowait(data)
                        except Exception:
                            pass
        except Exception as e:
            self.error = "%s: %s" % (type(e).__name__, e)
            self._stop.set()

    def read(self, timeout=0.5):
        """取一块音频；超时返回 None。"""
        try:
            return self.queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def stop(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        self._thread = None

    def __enter__(self):
        return self.start()

    def __exit__(self, *exc):
        self.stop()
        return False


def record_seconds(seconds, device_name=None, samplerate=DEFAULT_SR):
    """一次性录制指定秒数的系统输出音频（用于自测/校准）。"""
    dev = resolve(device_name)
    frames = int(samplerate * float(seconds))
    with dev.recorder(samplerate=samplerate, blocksize=frames) as rec:
        data = rec.record(numframes=frames)
    return np.asarray(data, dtype=np.float32), samplerate
