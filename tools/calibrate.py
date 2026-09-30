# -*- coding: utf-8 -*-
"""判定阈值标定 + 共音分组率报告。

两件事必须由真实数据决定，不能拍脑袋：
  1. 「相似度 >= 多少算命中」——该阈值随特征参数变化。启用 CMN 后
     相似度的绝对尺度整体下移，旧的 0.75 不再适用。
  2. 「共音分组率」—— 决定项目上限的核心业务指标（见技术方案 §5）：
     音效类数 / 物品总数。越接近 1 说明音效越接近"一物一效"。

做法：留一交叉验证。每次取一条样本当查询、其余样本建库，
记录"最像的同类"与"最像的异类"分数，用两个分布的交界定阈值。

用法：
    python tools/calibrate.py                    # 标定 data/library
    python tools/calibrate.py --root <库路径>
    python tools/calibrate.py --json out.json
"""

import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))

import aq_dsp as dsp          # noqa: E402
import aq_library as lib      # noqa: E402


def collect(library):
    """把库里全部样本读成 (sid, label, 特征) 列表。"""
    entries = []
    for s in library.sounds():
        for smp in s["samples"]:
            feat = library._sample_features(s, smp)      # noqa: SLF001
            if feat is not None and feat.shape[0] > 0:
                entries.append({"sid": s["id"], "label": s.get("label", s["id"]),
                                "file": smp["file"], "feat": feat})
    return entries


def main():
    ap = argparse.ArgumentParser(description="判定阈值标定与共音分组率")
    ap.add_argument("--root", default=lib.default_root())
    ap.add_argument("--json", default="", help="结果落盘路径")
    ap.add_argument("--band", type=float, default=0.3)
    args = ap.parse_args()

    library = lib.open_library(args.root)
    rows, total = library.stats()
    entries = collect(library)

    print("=" * 66)
    print("阈值标定 · %s" % library.root)
    print("=" * 66)

    # ---- 共音分组率
    print("\n[1] 共音分组率")
    for sid, label, items, n in rows:
        print("    %-6s %-18s 样本%3d   物品[%s]" % (sid, label, n, items))

    item_total = sum(len(s["itemNames"]) for s in library.sounds())
    sound_total = len([s for s in library.sounds() if s["samples"]])
    group_rate = (sound_total / float(item_total)) if item_total else None

    print("\n    音效类数      : %d" % sound_total)
    print("    登记物品数    : %d" % item_total)
    if group_rate is not None:
        print("    共音分组率    : %.4f   (= 音效类数 / 物品数)" % group_rate)
        if group_rate >= 0.8:
            print("      -> 接近一物一效，识别粒度可到单件物品")
        elif group_rate >= 0.5:
            print("      -> 部分共音，输出需为候选集（2~5 件）")
        else:
            print("      -> 共音严重，输出只能到「音效类 + 价格区间」")
    else:
        print("    共音分组率    : 无法计算（未登记物品名）")

    if len(entries) < 2:
        print("\n样本不足（%d 条），无法做留一交叉验证。" % len(entries))
        print("先跑 1-采样.cmd 与 2-打标.cmd 建立样本库。")
        return 3

    # 只有"同类样本 >= 2 条"的条目才可评价：同类仅一条时不存在同源配对，
    # 会被误记为误判，把准确率虚假压低。
    from collections import Counter
    cnt = Counter(e["sid"] for e in entries)
    evaluable = [e for e in entries if cnt[e["sid"]] >= 2]
    skipped = len(entries) - len(evaluable)

    if len(evaluable) < 2:
        print("\n可评估样本不足：%d 条样本中仅 %d 条所属音效类有 >=2 条样本。"
              % (len(entries), len(evaluable)))
        print("每类至少采 2 条（建议 5~10 条）才能标定阈值。")
        return 3

    # ---- 留一交叉验证
    print("\n[2] 留一交叉验证（评价 %d 条；跳过 %d 条同类样本不足的）"
          % (len(evaluable), skipped))
    same, diff, fails = [], [], []
    for e in evaluable:
        best_same, best_diff, pred = -1.0, -1.0, None
        for o in entries:
            if o is e:
                continue
            sim = dsp.similarity(e["feat"], o["feat"], band=args.band)
            if o["sid"] == e["sid"]:
                best_same = max(best_same, sim)
            elif sim > best_diff:
                best_diff, pred = sim, o["sid"]
        if best_same >= 0:
            same.append(best_same)
        if best_diff >= 0:
            diff.append(best_diff)
        if best_diff > best_same:
            fails.append({"sid": e["sid"], "pred": pred,
                          "same": round(best_same, 4), "diff": round(best_diff, 4)})

    same_a = np.array(same) if same else np.array([0.0])
    diff_a = np.array(diff) if diff else np.array([0.0])
    acc = 1.0 - len(fails) / float(len(evaluable))

    print("    同源相似度    : min=%.4f  P5=%.4f  mean=%.4f"
          % (same_a.min(), np.percentile(same_a, 5), same_a.mean()))
    print("    异源相似度    : P95=%.4f  max=%.4f  mean=%.4f"
          % (np.percentile(diff_a, 95), diff_a.max(), diff_a.mean()))
    print("    留一准确率    : %.1f%%  (%d/%d)"
          % (acc * 100, len(evaluable) - len(fails), len(evaluable)))
    if fails:
        print("    误判明细    :")
        for f in fails[:10]:
            print("      %-8s -> 误判为 %-8s  同源=%.4f < 异源=%.4f"
                  % (f["sid"], f["pred"], f["same"], f["diff"]))

    # ---- 建议阈值
    p5_same = float(np.percentile(same_a, 5))
    p95_diff = float(np.percentile(diff_a, 95))
    separated = p5_same > p95_diff
    rec = round((p5_same + p95_diff) / 2.0, 4) if separated else round(p5_same, 4)

    print("\n[3] 建议判定阈值")
    if separated:
        print("    两分布可分离：同源 P5=%.4f > 异源 P95=%.4f" % (p5_same, p95_diff))
    else:
        print("    ⚠ 两分布存在重叠（同源 P5=%.4f <= 异源 P95=%.4f）"
              % (p5_same, p95_diff))
        print("      阈值取同源 P5，宁可漏检也不误判；样本量上来后重叠通常会收窄。")
    print("    建议 min-sim = %.4f" % rec)
    print("    用法：match_live.py 会自动读取本文件；也可手动 --min-sim %.2f" % rec)

    out = {
        "root": library.root,
        "soundClasses": sound_total,
        "itemTotal": item_total,
        "groupingRate": round(group_rate, 4) if group_rate is not None else None,
        "sampleCount": len(entries),
        "evaluated": len(evaluable),
        "skipped": skipped,
        "leaveOneOutAccuracy": round(acc, 4),
        "sameSim": {"min": round(float(same_a.min()), 4),
                    "p5": round(p5_same, 4),
                    "mean": round(float(same_a.mean()), 4)},
        "diffSim": {"p95": round(p95_diff, 4),
                    "max": round(float(diff_a.max()), 4),
                    "mean": round(float(diff_a.mean()), 4)},
        "separated": bool(separated),
        "recommendedMinSim": rec,
        "fails": fails,
    }
    path = args.json or os.path.join(library.root, "calibration.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print("\n标定结果已写入：%s" % path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
