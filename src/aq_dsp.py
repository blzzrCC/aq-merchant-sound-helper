# -*- coding: utf-8 -*-
"""音频特征提取与相似度计算。

设计约束：
  - 只依赖 numpy，不引入 scipy / librosa，便于后续整体移植到 C#。
  - 所有参数集中在文件头，C# 端移植时必须逐项对齐（见 docs/暗区鉴宝-技术方案-v0.2.md §3.3）。
"""

import numpy as np

# ---- 特征参数（移植 C# 时需严格保持一致）----
TARGET_SR = 16000      # 统一重采样目标采样率
N_FFT = 1024           # FFT 窗长
HOP = 256              # 帧移
N_MELS = 64            # Mel 滤波器数量
FMIN = 50.0            # Mel 下限频率
FMAX = 7800.0          # Mel 上限频率（< TARGET_SR/2）
EPS = 1e-9


def to_mono(x):
    """立体声降混为单声道。左右声道差异主要来自声像位置，属于噪声。"""
    x = np.asarray(x, dtype=np.float32)
    if x.ndim == 2:
        return x.mean(axis=1).astype(np.float32)
    return x


def trim_silence(x, sr, thresh_db=-45.0, pad_ms=20):
    """裁掉首尾静音，只保留有效音频段。

    必要性（2026-09-30 实测）：参考素材 15 条中，首尾静音占总时长 2%~76%。
    DTW 会把静音帧对齐到有效音帧上，使"静音多的音频 vs 静音少的音频"
    相似度被系统性压低 —— 实测相似度矩阵因此出现整块 0.000 的分块结构，
    与声学内容无关。裁剪后有效时长离散度由 3.6 倍收敛到 2.6 倍。

    阈值取相对峰值 -45dB；首尾各留 pad_ms 毫秒，避免削掉音头音尾的瞬态。
    """
    x = np.asarray(x, dtype=np.float32)
    if x.ndim == 2:
        x = x.mean(axis=1)
    if x.size == 0:
        return x

    fl = max(1, int(round(sr * 0.010)))          # 10ms 一帧
    n = x.size // fl
    if n < 3:                                     # 太短，不做裁决
        return x

    seg = x[:n * fl].reshape(n, fl).astype(np.float64)
    rms = np.sqrt((seg ** 2).mean(axis=1))
    peak = float(rms.max())
    if peak <= 1e-9:
        return x

    thr = peak * (10.0 ** (thresh_db / 20.0))
    act = np.nonzero(rms > thr)[0]
    if act.size == 0:
        return x

    pad = int(round(sr * pad_ms / 1000.0))
    lo = max(0, int(act[0]) * fl - pad)
    hi = min(x.size, (int(act[-1]) + 1) * fl + pad)
    return x[lo:hi].astype(np.float32)


def cmn(feat):
    """倒谱均值归一化：逐 Mel 维减去时间轴均值。

    作用：消除通道频响造成的加性偏移。EQ 倾斜、设备频响差异在 log 域
    都表现为"每个 Mel 维整体加减一个常数"，减掉均值即可抵消。
    这是削弱跨设备失配最廉价有效的一步。

    实测（15 条素材，EQ 倾斜 4dB/oct 扰动）：
    未归一化时自相似 0.669、top-1 11/15；归一化后 0.948、top-1 15/15。
    同时类间相似度由 0.795 降到 0.603，可分性一并改善。
    """
    feat = np.asarray(feat, dtype=np.float32)
    if feat.shape[0] < 2:          # 单帧无均值可减，减了会归零
        return feat
    return (feat - feat.mean(axis=0, keepdims=True)).astype(np.float32)


def l2norm(feat):
    """逐帧 L2 归一化。使余弦距离只反映谱形状，不受帧能量影响。"""
    feat = np.asarray(feat, dtype=np.float32)
    if feat.shape[0] == 0:
        return feat
    n = np.linalg.norm(feat, axis=1, keepdims=True) + EPS
    return (feat / n).astype(np.float32)


def peak_normalize(x, target=0.9):
    """峰值归一化，消除音量差异对相似度的影响。"""
    x = np.asarray(x, dtype=np.float32)
    if x.size == 0:
        return x
    peak = float(np.abs(x).max())
    if peak < 1e-6:
        return x
    return (x * (target / peak)).astype(np.float32)


def downsample(x, sr, target=TARGET_SR):
    """降采样到目标采样率。整数倍时用盒式平均（等效抗混叠），否则线性插值。"""
    x = np.asarray(x, dtype=np.float32)
    if sr == target:
        return x, sr
    ratio = float(sr) / float(target)
    r = int(round(ratio))
    if abs(ratio - r) < 1e-6 and r > 1:
        n = (len(x) // r) * r
        if n == 0:
            return x, sr
        return x[:n].reshape(-1, r).mean(axis=1).astype(np.float32), target
    n_out = int(len(x) * target / float(sr))
    if n_out <= 1:
        return x, sr
    t_src = np.arange(len(x), dtype=np.float64) / float(sr)
    t_dst = np.arange(n_out, dtype=np.float64) / float(target)
    return np.interp(t_dst, t_src, x).astype(np.float32), target


def frame(x, n_fft=N_FFT, hop=HOP):
    """分帧。返回 (帧数, n_fft)。"""
    x = np.asarray(x, dtype=np.float32)
    if len(x) < n_fft:
        x = np.pad(x, (0, n_fft - len(x)))
    n = 1 + (len(x) - n_fft) // hop
    if n <= 0:
        n = 1
    idx = np.arange(n_fft)[None, :] + hop * np.arange(n)[:, None]
    return x[idx]


def _hz2mel(f):
    return 2595.0 * np.log10(1.0 + np.asarray(f, dtype=np.float64) / 700.0)


def _mel2hz(m):
    return 700.0 * (10.0 ** (np.asarray(m, dtype=np.float64) / 2595.0) - 1.0)


def mel_filterbank(sr=TARGET_SR, n_fft=N_FFT, n_mels=N_MELS, fmin=FMIN, fmax=FMAX,
                   cache={}):
    """三角 Mel 滤波器组，形状 (n_mels, n_fft//2+1)。结果带缓存。"""
    key = (sr, n_fft, n_mels, round(fmin, 3), round(fmax, 3))
    if key in cache:
        return cache[key]

    n_bins = n_fft // 2 + 1
    fmax = min(fmax, sr / 2.0)
    mels = np.linspace(_hz2mel(fmin), _hz2mel(fmax), n_mels + 2)
    hzs = _mel2hz(mels)
    bins = np.floor((n_fft + 1) * hzs / sr).astype(int)
    bins = np.clip(bins, 0, n_bins - 1)

    fb = np.zeros((n_mels, n_bins), dtype=np.float64)
    for i in range(n_mels):
        left, center, right = bins[i], bins[i + 1], bins[i + 2]
        if center <= left:
            center = left + 1
        if right <= center:
            right = center + 1
        for k in range(left, min(center, n_bins)):
            fb[i, k] = (k - left) / float(center - left)
        for k in range(center, min(right, n_bins)):
            fb[i, k] = (right - k) / float(right - center)

    cache[key] = fb
    return fb


def log_mel(x, sr=TARGET_SR, n_fft=N_FFT, hop=HOP, n_mels=N_MELS,
            fmin=FMIN, fmax=FMAX):
    """对数 Mel 频谱，形状 (帧数, n_mels)。"""
    x = np.asarray(x, dtype=np.float32)
    window = np.hanning(n_fft).astype(np.float32)
    frames = frame(x, n_fft, hop) * window[None, :]
    spec = np.abs(np.fft.rfft(frames, axis=1))
    fb = mel_filterbank(sr, n_fft, n_mels, fmin, fmax)
    mel = spec @ fb.T
    return np.log(mel + EPS).astype(np.float32)


def extract(x, sr, trim=True, trim_opts=None, use_cmn=True):
    """一站式特征提取：降混 -> 裁静音 -> 降采样 -> 峰值归一 -> log-Mel -> CMN。

    trim=True 时先裁掉首尾静音段。这一步不是可选优化：
    静音帧的 log-Mel 是常数向量，参与 DTW 会稀释有效内容并放大时长差异。
    trim_opts 透传给 trim_silence（thresh_db / pad_ms），便于参数标定。

    use_cmn=True 时做倒谱均值归一化，吸收设备频响造成的加性偏移。

    返回 (features, meta)。
    meta.duration 为裁剪后的有效时长，meta.rawDuration 为裁剪前时长。
    """
    mono = to_mono(x)
    raw_dur = (len(mono) / float(sr)) if mono.size else 0.0

    if trim:
        mono = trim_silence(mono, sr, **(trim_opts or {}))

    res, out_sr = downsample(mono, sr)
    res = peak_normalize(res)
    feat = log_mel(res, out_sr)
    if use_cmn:
        feat = cmn(feat)

    act_dur = (len(mono) / float(sr)) if mono.size else 0.0
    meta = {
        "orig_sr": int(sr),
        "work_sr": int(out_sr),
        "rawDuration": round(raw_dur, 4),
        "duration": round(act_dur, 4),
        "silenceRatio": round(1.0 - act_dur / raw_dur, 4) if raw_dur > 1e-9 else 0.0,
        "frames": int(feat.shape[0]),
        "n_mels": int(feat.shape[1]),
        "cmn": bool(use_cmn),
        "peak": round(float(np.abs(mono).max()) if mono.size else 0.0, 6),
        "rms": round(float(np.sqrt((mono.astype(np.float64) ** 2).mean())) if mono.size else 0.0, 6),
    }
    return feat, meta


def dtw_distance(a, b, band=0.3):
    """归一化 DTW 距离（0~1 量级）。a、b 均为 (帧数, 特征维)。

    代价用余弦距离；band 为 Sakoe-Chiba 带宽比例，限制弯曲幅度。
    返回每步平均代价，越小越相似。

    坑（2026-09-30 实测发现）：带宽下限必须覆盖两段帧数之差。
    否则终点 (ta, tb) 不可达，路径代价保持 inf，折算后得到伪 0 相似度。
    实测素材时长跨 0.235s~0.853s（帧数差达 4.5 倍），band=0.3 时矩阵中
    出现成块 0.000 —— 全部是伪值。
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.shape[0] == 0 or b.shape[0] == 0:
        return 1.0

    an = a / (np.linalg.norm(a, axis=1, keepdims=True) + EPS)
    bn = b / (np.linalg.norm(b, axis=1, keepdims=True) + EPS)
    cost = 1.0 - (an @ bn.T)          # (Ta, Tb)，0 表示完全同向
    np.clip(cost, 0.0, 2.0, out=cost)

    ta, tb = cost.shape
    inf = 1e18
    prev = np.full(tb + 1, inf, dtype=np.float64)
    prev[0] = 0.0
    band_w = max(1, int(round(band * max(ta, tb))), abs(ta - tb))

    for i in range(1, ta + 1):
        cur = np.full(tb + 1, inf, dtype=np.float64)
        lo = max(1, i - band_w)
        hi = min(tb, i + band_w)
        for j in range(lo, hi + 1):
            d = min(prev[j], cur[j - 1], prev[j - 1])
            cur[j] = cost[i - 1, j - 1] + d
        prev = cur

    if prev[tb] >= inf:               # 理论不可达，兜底为最大距离
        return 1.0
    # 用较长一侧的帧数归一化，而不是 (ta+tb)。
    # 原因：DTW 路径步数受带宽约束，始终接近 max(ta, tb)；用 (ta+tb) 作分母时，
    # 两段时长差越大分母越小、距离被系统性抬高，表现为"短音频 vs 长音频"
    # 恒得低分（实测矩阵中出现整行整列 0.000）。
    return float(prev[tb] / float(max(ta, tb)))


def similarity(a, b, band=0.3):
    """把 DTW 距离折算为 0~1 相似度。"""
    d = dtw_distance(a, b, band=band)
    return max(0.0, min(1.0, 1.0 - d * 2.0))
