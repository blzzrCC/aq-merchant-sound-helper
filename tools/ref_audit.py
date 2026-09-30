# -*- coding: utf-8 -*-
"""素材可用性审计。

一套逻辑，两类数据都能跑：

  A. 采集到的一批 wav（第三方参考站素材 / 本机实录样本）
     -> 回答"这批音频本身有多少真实区分度、能不能直接用作模板"

  B. 两组 wav（源域模板 vs 目标域查询）
     -> 回答"跨链路直接复用会损失多少精度"

审计三项：

  1. 去重审计    文件级 md5 分组，得出「条目数 -> 唯一音效数」的压缩比。
                 这是共音分组率的第一手证据。
  2. 可分性审计  两两相似度矩阵。若类间相似度普遍逼近类内水平，
                 说明音效本身不足以区分，项目上限受限。
  3. 链鲁棒性审计 对查询集施加典型录制链路扰动，测 top-1 准确率退化。
                 用于判定"外部素材能否跨设备直接进库"。

用法：
    # 单目录审计
    python tools/ref_audit.py --dir <wav目录> [--json out.json]

    # 跨链路审计
    python tools/ref_audit.py --dir <参考目录> --perturb
"""

import argparse
import hashlib
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))

import aq_dsp as dsp          # noqa: E402
import aq_library as lib      # noqa: E402


# ---------------------------------------------------------------- 载入

def scan_wavs(folder):
    names = sorted(
        (n for n in os.listdir(folder) if n.lower().endswith(".wav")),
        key=lambda n: (len(n), n),
    )
    out = []
    for n in names:
        p = os.path.join(folder, n)
        try:
            audio, sr = lib.read_wav(p)
        except Exception as e:                       # noqa: BLE001
            print("  [skip] %s: %s" % (n, e))
            continue
        out.append({"name": n, "path": p, "audio": audio, "sr": sr})
    return out


def md5_of(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------- 扰动链

def sim_gain(x, db):
    """播放增益差异。峰值归一化应当能完全吸收它。"""
    return np.clip(x * (10.0 ** (db / 20.0)), -1.0, 1.0)


def _fft_gain(x, sr, gain_fn):
    """频域施加增益曲线。

    必须补零到 2 倍长度再变换，否则频域相乘等价于时域循环卷积，
    音频首尾会互相回绕污染 —— 实测这会让静音裁剪边界漂移 40%，
    伪造出并不存在的"链路失配"。
    """
    n = len(x)
    n2 = 1
    while n2 < 2 * n:
        n2 *= 2
    spec = np.fft.rfft(x, n=n2)
    f = np.fft.rfftfreq(n2, 1.0 / sr)
    g = gain_fn(np.maximum(f, 20.0))
    return np.fft.irfft(spec * g, n=n2)[:n].astype(np.float32)


def sim_eq_tilt(x, sr, db_per_oct):
    """频响倾斜：模拟不同声卡 / 功放 / 录制设备的频率响应差异。"""
    if abs(db_per_oct) < 1e-6:
        return x
    return _fft_gain(x, sr, lambda f: 10.0 ** (db_per_oct * np.log2(f / 1000.0) / 20.0))


def sim_compress(x, ratio, thresh):
    """动态范围压缩：模拟不同音量下的非线性。"""
    a = np.abs(x)
    over = a > thresh
    a2 = np.where(over, thresh + (a - thresh) / ratio, a)
    return (np.sign(x) * a2).astype(np.float32)


def sim_noise(x, snr_db, rng):
    """底噪：模拟录制端噪声地板。"""
    p = float(np.mean(x.astype(np.float64) ** 2))
    if p <= 0:
        return x
    sigma = np.sqrt(p / (10.0 ** (snr_db / 10.0)))
    return (x + rng.normal(0.0, sigma, size=x.shape)).astype(np.float32)


def sim_lowpass(x, sr, cutoff):
    """高频滚降：模拟采样率转换 / 编码损耗砍掉的高频。"""
    return _fft_gain(x, sr, lambda f: 1.0 / (1.0 + (f / float(cutoff)) ** 4))


PERTURB_SETS = {
    "A_仅增益":      [("gain", 6.0)],
    "B_增益+底噪":   [("gain", -5.0), ("noise", 35.0)],
    "C_EQ倾斜":      [("eq", 4.0)],
    "D_EQ+压缩":     [("eq", 4.0), ("compress", (3.0, 0.3))],
    "E_全链路":      [("eq", 3.0), ("compress", (4.0, 0.25)), ("noise", 30.0), ("lowpass", 7000.0)],
}


def apply_chain(audio, sr, steps, rng):
    # 扰动在降混后的单声道上做：特征链路本身也是单声道，
    # 立体声差异只会引入无关维度。
    x = dsp.to_mono(audio)
    for kind, arg in steps:
        if kind == "gain":
            x = sim_gain(x, arg)
        elif kind == "eq":
            x = sim_eq_tilt(x, sr, arg)
        elif kind == "compress":
            x = sim_compress(x, arg[0], arg[1])
        elif kind == "noise":
            x = sim_noise(x, arg, rng)
        elif kind == "lowpass":
            x = sim_lowpass(x, sr, arg)
    return x


# ---------------------------------------------------------------- 审计

def audit_dedup(items):
    groups = {}
    for it in items:
        groups.setdefault(md5_of(it["path"]), []).append(it["name"])
    uniq = len(groups)
    total = len(items)

    # 按 PCM 帧数分组。帧数相同 = 数据段等长，是"同源"的强信号。
    by_len = {}
    for it in items:
        by_len.setdefault(int(it["audio"].shape[0]), []).append(it["name"])
    len_groups = [
        {"frames": k, "names": sorted(v)}
        for k, v in sorted(by_len.items()) if len(v) > 1
    ]

    return {
        "totalEntries": total,
        "uniqueAudio": uniq,
        "compression": round(uniq / float(total), 4) if total else None,
        "groups": [
            {"md5": k[:12], "names": v}
            for k, v in sorted(groups.items(), key=lambda kv: kv[1][0])
        ],
        "sameLengthGroups": len_groups,
    }


def pairwise(feats, band=0.3):
    n = len(feats)
    m = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(i + 1, n):
            s = dsp.similarity(feats[i], feats[j], band=band)
            m[i, j] = m[j, i] = s
    np.fill_diagonal(m, 1.0)
    return m


def audit_separability(items, band=0.3, trim=True):
    feats = [dsp.extract(it["audio"], it["sr"], trim=trim)[0] for it in items]
    m = pairwise(feats, band=band)
    n = len(items)
    off = m[~np.eye(n, dtype=bool)]
    upper = m[np.triu_indices(n, 1)]

    # 对每个条目，找出它最像的那个"别的条目"
    nearest = []
    for i in range(n):
        row = m[i].copy()
        row[i] = -1.0
        j = int(np.argmax(row))
        nearest.append({
            "name": items[i]["name"],
            "nearest": items[j]["name"],
            "similarity": round(float(row[j]), 4),
        })
    nearest.sort(key=lambda r: -r["similarity"])
    return {
        "band": band,
        "interMax": round(float(upper.max()), 4),
        "interMean": round(float(off.mean()), 4),
        "interP95": round(float(np.percentile(upper, 95)), 4),
        "interMin": round(float(upper.min()), 4),
        "matrix": [[round(float(v), 4) for v in row] for row in m],
        "names": [it["name"] for it in items],
        "topConfusions": nearest[:8],
    }


def audit_chain(items, band=0.3, seed=20260930, trim=True):
    """库=原始，查询=扰动后。看 top-1 是否还认得自己。"""
    rng = np.random.default_rng(seed)
    base = [dsp.extract(it["audio"], it["sr"], trim=trim)[0] for it in items]
    names = [it["name"] for it in items]

    results = []
    for tag, steps in PERTURB_SETS.items():
        hit = 0
        sims = []
        fails = []
        for i, it in enumerate(items):
            pert = apply_chain(it["audio"], it["sr"], steps, rng)
            q = dsp.extract(pert, it["sr"], trim=trim)[0]
            scores = [dsp.similarity(q, b, band=band) for b in base]
            k = int(np.argmax(scores))
            sims.append(scores[i])
            if names[k] == names[i]:
                hit += 1
            else:
                fails.append({
                    "query": names[i], "pred": names[k],
                    "selfSim": round(float(scores[i]), 4),
                    "bestSim": round(float(scores[k]), 4),
                })
        results.append({
            "set": tag,
            "top1": hit,
            "total": len(items),
            "top1Rate": round(hit / float(len(items)), 4),
            "selfSimMean": round(float(np.mean(sims)), 4),
            "fails": fails,
        })
    return results


# ---------------------------------------------------------------- 主流程

def main():
    ap = argparse.ArgumentParser(description="素材可用性审计")
    ap.add_argument("--dir", required=True, help="wav 目录")
    ap.add_argument("--band", type=float, default=0.3)
    ap.add_argument("--perturb", action="store_true", help="附加跨链路鲁棒性审计")
    ap.add_argument("--matrix", action="store_true", help="打印完整相似度矩阵")
    ap.add_argument("--no-trim", action="store_true", help="关闭首尾静音裁剪（对照用）")
    ap.add_argument("--json", default="", help="报告落盘路径")
    args = ap.parse_args()

    items = scan_wavs(args.dir)
    if not items:
        print("目录内没有可用 wav：%s" % args.dir)
        return 1

    print("=" * 66)
    print("素材可用性审计  |  %s" % args.dir)
    print("=" * 66)

    # ---- 0 文件参数
    trim = not args.no_trim
    print("\n[0] 文件参数（trim=%s）" % ("on" if trim else "off"))
    print("    %-14s %9s %9s %9s %7s" % ("file", "rawDur", "actDur", "静音占比", "帧数"))
    for it in items:
        _, m = dsp.extract(it["audio"], it["sr"], trim=trim)
        print("    %-14s %9.3f %9.3f %8.1f%% %7d"
              % (it["name"], m["rawDuration"], m["duration"],
                 m["silenceRatio"] * 100, m["frames"]))

    # ---- 1 去重
    ded = audit_dedup(items)
    print("\n[1] 去重审计")
    print("    条目数        : %d" % ded["totalEntries"])
    print("    唯一音频数    : %d" % ded["uniqueAudio"])
    print("    压缩比        : %.4f   (= 唯一音频 / 条目数)" % ded["compression"])
    dups = [g for g in ded["groups"] if len(g["names"]) > 1]
    if dups:
        print("    字节重复组    :")
        for g in dups:
            print("      %s  <-  %s" % (g["md5"], ", ".join(g["names"])))
    else:
        print("    字节重复组    : 无（每条音频字节都不同）")

    if ded["sameLengthGroups"]:
        print("    等长组（帧数相同，疑似同源）:")
        for g in ded["sameLengthGroups"]:
            print("      %6d frames  <-  %s" % (g["frames"], ", ".join(g["names"])))
    else:
        print("    等长组        : 无")

    # ---- 2 可分性
    sep = audit_separability(items, band=args.band, trim=trim)
    print("\n[2] 可分性审计（DTW 相似度，band=%.2f）" % sep["band"])
    print("    类间相似度    : max=%.4f  P95=%.4f  mean=%.4f  min=%.4f"
          % (sep["interMax"], sep["interP95"], sep["interMean"], sep["interMin"]))
    print("    最易混淆对    :")
    for r in sep["topConfusions"][:5]:
        print("      %-12s ~ %-12s  sim=%.4f" % (r["name"], r["nearest"], r["similarity"]))

    if args.matrix:
        idx = {n: i for i, n in enumerate(sep["names"])}
        head = "".join("%9s" % n.replace(".wav", "") for n in sep["names"])
        print("\n    相似度矩阵：\n    %9s%s" % ("", head))
        for i, n in enumerate(sep["names"]):
            row = "".join("%9.3f" % v for v in sep["matrix"][i])
            print("    %9s%s" % (n.replace(".wav", ""), row))

    if ded["sameLengthGroups"]:
        idx = {n: i for i, n in enumerate(sep["names"])}
        print("\n    等长组内相似度（字节不同但长度相同，检验是否同源）:")
        for g in ded["sameLengthGroups"]:
            ns = [n for n in g["names"] if n in idx]
            for a in range(len(ns)):
                for b in range(a + 1, len(ns)):
                    v = sep["matrix"][idx[ns[a]]][idx[ns[b]]]
                    print("      %-12s ~ %-12s  sim=%.4f" % (ns[a], ns[b], v))

    # ---- 3 链路鲁棒性
    chain = None
    if args.perturb:
        chain = audit_chain(items, band=args.band, trim=trim)
        print("\n[3] 跨链路鲁棒性审计（库=原始，查询=扰动后）")
        for r in chain:
            print("    %-12s  top1=%2d/%-2d (%.1f%%)   自相似=%.4f"
                  % (r["set"], r["top1"], r["total"], r["top1Rate"] * 100,
                     r["selfSimMean"]))
            for f in r["fails"]:
                print("        x %-12s -> 误判为 %-12s   self=%.4f < best=%.4f"
                      % (f["query"], f["pred"], f["selfSim"], f["bestSim"]))

    # ---- 结论
    print("\n[结论]")
    if ded["compression"] is not None and ded["compression"] < 1.0:
        print("    · 该批素材存在共音：%d 条目仅对应 %d 种音频，压缩比 %.3f"
              % (ded["totalEntries"], ded["uniqueAudio"], ded["compression"]))
        print("      -> 输出必须是候选集，不能给单一答案")
    if sep["interMax"] >= 0.95:
        print("    · 存在近乎相同的音效对（max=%.4f），这些条目在识别上不可分" % sep["interMax"])
    if chain:
        worst = max(chain, key=lambda r: r["top1Rate"])
        best = min(chain, key=lambda r: r["top1Rate"])
        print("    · 链路扰动下 top-1 从 %.1f%% 退化到 %.1f%%（%s）"
              % (worst["top1Rate"] * 100, best["top1Rate"] * 100, best["set"]))
        if best["top1Rate"] < 0.9:
            print("      -> 跨链路直接复用不可行，目标域必须自录")

    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump({"dedup": ded, "separability": sep, "chain": chain},
                      f, ensure_ascii=False, indent=2)
        print("\n报告已写入：%s" % args.json)
    return 0


if __name__ == "__main__":
    sys.exit(main())
