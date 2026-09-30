# -*- coding: utf-8 -*-
"""音效事件检测：从连续音频流中切出「拾取音」片段。

思路：不比连续分类，而是先切分再识别。自适应噪声底 + 能量门限的块级状态机。
"""

import numpy as np
import collections

DEFAULT_SR = 48000
DEFAULT_BLOCK = 2400      # 50ms @ 48kHz


class SoundEventDetector(object):
    """块级事件检测器。

    用法：
        det = SoundEventDetector()
        for block in stream:
            evt = det.push(block)
            if evt is not None:
                handle(evt)

    push() 返回本次触发的完整音频（np.float32，形状 (n,) 或 (n, ch)），
    未完成则返回 None。
    """

    def __init__(self, samplerate=DEFAULT_SR, block_size=DEFAULT_BLOCK,
                 abs_threshold=0.012,     # 绝对 RMS 门限，防止静音时误触发
                 noise_mult=4.0,          # 相对噪声底的倍数
                 pre_ms=80,               # 保留触发前的音频
                 min_ms=120,              # 低于此时长丢弃
                 max_ms=1600,             # 超过此时长强制收尾
                 tail_ms=200):            # 连续静音多久判定结束
        self.samplerate = int(samplerate)
        self.block_size = int(block_size)
        self.abs_threshold = float(abs_threshold)
        self.noise_mult = float(noise_mult)

        self.pre_blocks = max(1, int(self.samplerate * pre_ms / 1000.0 / self.block_size))
        self.min_samples = int(self.samplerate * min_ms / 1000.0)
        self.max_samples = int(self.samplerate * max_ms / 1000.0)
        self.tail_blocks_needed = max(1, int(round(tail_ms / float(
            self.block_size * 1000.0 / self.samplerate))))

        self.noise_floor = None
        self.active = False
        self.collected = []
        self.pre_buffer = collections.deque(maxlen=self.pre_blocks)
        self.tail_blocks = 0
        self.n_samples = 0
        self.peak_since_active = 0.0

        self.stats = {"blocks": 0, "events": 0, "dropped_short": 0,
                      "truncated": 0}

    # ------------------------------------------------------------------

    def _rms(self, mono):
        if mono.size == 0:
            return 0.0
        return float(np.sqrt((mono.astype(np.float64) ** 2).mean()))

    def _to_mono(self, block):
        b = np.asarray(block, dtype=np.float32)
        if b.ndim == 2:
            return b.mean(axis=1)
        return b

    def _threshold(self):
        if self.noise_floor is None:
            return self.abs_threshold
        return max(self.abs_threshold, self.noise_floor * self.noise_mult)

    # ------------------------------------------------------------------

    def push(self, block):
        block = np.asarray(block, dtype=np.float32)
        if block.size == 0:
            return None
        mono = self._to_mono(block)
        rms = self._rms(mono)
        self.stats["blocks"] += 1
        thr = self._threshold()

        if not self.active:
            # 空闲时缓慢跟踪噪声底（只向下收敛，避免把持续声当成底噪）
            if self.noise_floor is None:
                self.noise_floor = rms
            elif rms < self.noise_floor:
                self.noise_floor = 0.85 * self.noise_floor + 0.15 * rms
            else:
                self.noise_floor = 0.999 * self.noise_floor + 0.001 * rms

            if rms > thr:
                self.active = True
                self.collected = list(self.pre_buffer)
                self.collected.append(block)
                self.pre_buffer.clear()
                self.n_samples = mono.size
                self.tail_blocks = 0
                self.peak_since_active = float(np.abs(mono).max())
            else:
                self.pre_buffer.append(block)
            return None

        # 事件进行中
        self.collected.append(block)
        self.n_samples += mono.size
        peak = float(np.abs(mono).max())
        if peak > self.peak_since_active:
            self.peak_since_active = peak

        if rms < thr:
            self.tail_blocks += 1
        else:
            self.tail_blocks = 0

        finished = (self.tail_blocks >= self.tail_blocks_needed or
                    self.n_samples >= self.max_samples)
        if not finished:
            return None

        truncated = self.n_samples >= self.max_samples
        event = np.concatenate(self.collected, axis=0)
        self.active = False
        self.collected = []
        self.tail_blocks = 0
        self.n_samples = 0

        if truncated:
            event = event[:self.max_samples]
            self.stats["truncated"] += 1

        # 掐掉尾部静音，保留一点自然收尾
        if not truncated:
            keep = self.samplerate * 60 // 1000
            cut = max(0, event.shape[0] - keep - self.block_size * (self.tail_blocks - 1))
            if cut > 0:
                event = event[:max(cut, self.min_samples)]

        if event.shape[0] < self.min_samples:
            self.stats["dropped_short"] += 1
            return None

        self.stats["events"] += 1
        return event

    def reset(self):
        self.noise_floor = None
        self.active = False
        self.collected = []
        self.pre_buffer.clear()
        self.tail_blocks = 0
        self.n_samples = 0
