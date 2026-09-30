# -*- coding: utf-8 -*-
"""内核自测：合成音效 -> 建库 -> 匹配 -> 混淆矩阵。

不依赖游戏、不依赖真实音效，用带随机扰动的合成音验证整条链路是否收敛。
"""

import os
import shutil
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))

import aq_dsp as dsp            # noqa: E402
import aq_library as lib        # noqa: E402
import aq_detect as detect      # noqa: E402

SR = 48000

# 三组"音效原型"：(低频, 高频, 衰减速率, 基准时长)
PROTOS = [
    # 高频金属感、快衰减
    {"id": "s01", "label": "金属·小件", "band": (1800.0, 5200.0), "decay": 12.0, "dur": 0.50},
    # 中频塑料感、中衰减
    {"id": "s02", "label": "塑料·中件", "band": (700.0, 2000.0), "decay": 7.0, "dur": 0.55},
    # 低频布料/纸感、慢衰减
    {"id": "s03", "label": "织物·大件", "band": (150.0, 900.0), "decay": 4.0, "dur": 0.65},
]


def synth(proto, rng, sr=SR):
    """按原型合成一段带随机扰动的音效：时间伸缩 + 音量抖动 + 底噪。"""
    lo, hi = proto["band"]
    dur = proto["dur"] * rng.uniform(0.85, 1.15)
    n = int(sr * dur)
    noise = rng.standard_normal(n).astype(np.float64)

    spec = np.fft.rfft(noise)
    freqs = np.fft.rfftfreq(n, 1.0 / sr)
    lo_j = lo * rng.uniform(0.9, 1.1)
    hi_j = hi * rng.uniform(0.9, 1.1)
    mask = ((freqs > lo_j) & (freqs < hi_j)).astype(np.float64)
    # 软边缘，避免陡峭掩码带来的伪影
    edge = max(1, int(len(mask) * 0.02))
    k = np.ones(edge) / edge
    mask = np.convolve(mask, k, mode="same")
    x = np.fft.irfft(spec * mask, n=n)

    env = np.exp(-proto["decay"] * rng.uniform(0.9, 1.1) * np.arange(n) / sr)
    attack = max(1, min(n, int(sr * 0.005)))
    env[:attack] *= np.linspace(0.0, 1.0, attack)
    x = x * env

    x = x / (np.abs(x).max() + 1e-9) * rng.uniform(0.45, 0.95)
    x = x + rng.standard_normal(n) * rng.uniform(0.001, 0.004)
    return x.astype(np.float32)


def main():
    rng = np.random.default_rng(20260930)
    tmp = tempfile.mkdtemp(prefix="aq_lib_test_")
    print("临时库目录：", tmp)

    try:
        library = lib.open_library(tmp)

        # ---- 建库：每个原型 5 条样本 ----
        for p in PROTOS:
            for i in range(5):
                audio = synth(p, rng)
                library.add_sample(p["id"], audio, SR,
                                   label=p["label"],
                                   item_names=["%s-物品%d" % (p["label"], k) for k in range(1, 4)])
        library.save()

        rows, total = library.stats()
        print("\n=== 库内容 ===")
        for sid, label, items, n in rows:
            print("  %-5s %-12s 候选物品[%s]  样本=%d" % (sid, label, items, n))
        print("  合计样本：%d" % total)

        # ---- 匹配测试：每个原型 10 次全新变体 ----
        print("\n=== 匹配结果（每类 10 次全新变体）===")
        matrix = {p["id"]: {q["id"]: 0 for q in PROTOS} for p in PROTOS}
        sims_correct = []
        sims_wrong = []

        for p in PROTOS:
            for _ in range(10):
                q = synth(p, rng)
                results, meta = library.match(q, SR, top_k=3)
                if not results:
                    print("  无匹配结果")
                    continue
                top = results[0]
                matrix[p["id"]][top["soundId"]] += 1
                if top["soundId"] == p["id"]:
                    sims_correct.append(top["similarity"])
                else:
                    sims_wrong.append((p["id"], top["soundId"], top["similarity"],
                                       [(r["soundId"], r["distance"]) for r in results]))

        print("\n  混淆矩阵（行=真实，列=判定）")
        header = "        " + "".join("%8s" % q["id"] for q in PROTOS) + "   %8s" % "正确率"
        print(header)
        for p in PROTOS:
            line = "  %-6s" % p["id"]
            ok = 0
            tot = 0
            for q in PROTOS:
                v = matrix[p["id"]][q["id"]]
                line += "%8d" % v
                tot += v
                if q["id"] == p["id"]:
                    ok = v
            line += "   %7.1f%%" % (100.0 * ok / tot if tot else 0.0)
            print(line)

        allq = sum(sum(matrix[p["id"]][q["id"]] for q in PROTOS) for p in PROTOS)
        allok = sum(matrix[p["id"]][p["id"]] for p in PROTOS)
        print("\n  总体准确率：%d/%d = %.1f%%" % (allok, allq, 100.0 * allok / allq))

        if sims_correct:
            print("  命中时相似度  min=%.3f  mean=%.3f  max=%.3f"
                  % (min(sims_correct), float(np.mean(sims_correct)), max(sims_correct)))
        if sims_wrong:
            print("  误判样本 %d 例，例如：" % len(sims_wrong))
            for w in sims_wrong[:3]:
                print("    真实 %s -> 判定 %s (sim=%.3f)  前3名=%s"
                      % (w[0], w[1], w[2], w[3]))
        else:
            print("  误判样本：0 例")

        # ---- 事件检测测试 ----
        print("\n=== 事件检测测试 ===")
        det = detect.SoundEventDetector(samplerate=SR, block_size=2400)
        stream = np.concatenate([
            np.zeros(SR // 2, dtype=np.float32),
            synth(PROTOS[0], rng),
            np.zeros(SR // 2, dtype=np.float32),
            synth(PROTOS[2], rng),
            np.zeros(SR // 2, dtype=np.float32),
        ])
        events = []
        bs = 2400
        for i in range(0, len(stream) - bs + 1, bs):
            evt = det.push(stream[i:i + bs])
            if evt is not None:
                events.append(evt)
        print("  注入 2 个音效，检测到 %d 个事件" % len(events))
        for i, e in enumerate(events):
            print("    #%d  长度=%.3fs  peak=%.4f" % (i + 1, len(e) / float(SR),
                                                      float(np.abs(e).max())))
        print("  检测器统计：", det.stats)

        print("\n自测完成。")
        return 0
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
