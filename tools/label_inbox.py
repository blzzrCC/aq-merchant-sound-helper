# -*- coding: utf-8 -*-
"""打标工具：逐个试听 _inbox 样本，归档到音效类。

打标模型（与方案 §4 一致）：
    音效类（soundId） -- 1 : N --> 物品名
「多件物品共用同一音效」正是通过把多个物品名登记到同一个音效类来表达的。

交互键位：
    回车    播放 / 重播当前样本
    数字    选择已有音效类
    n       新建音效类（会询问类名与对应物品名）
    s       跳过（保留在 _inbox）
    d       删除该样本
    q       保存并退出
"""

import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import aq_library as lib     # noqa: E402


def play_file(path):
    try:
        import soundcard as sc
        audio, sr = lib.read_wav(path)
        a = np.asarray(audio, dtype=np.float32)
        if a.ndim == 1:
            a = np.stack([a, a], axis=1)
        sc.default_speaker().play(a, samplerate=sr)
        return True
    except Exception as e:
        print("    播放失败：%s" % e)
        return False


def inbox_files(root):
    d = os.path.join(root, "_inbox")
    if not os.path.isdir(d):
        return []
    return [os.path.join(d, f) for f in sorted(os.listdir(d))
            if f.lower().endswith(".wav")]


def next_sound_id(library):
    n = 1
    existing = set(s["id"] for s in library.sounds())
    while ("s%02d" % n) in existing:
        n += 1
    return "s%02d" % n


def show_classes(library):
    rows = library.sounds()
    if not rows:
        print("    （暂无音效类）")
        return []
    out = []
    for i, s in enumerate(rows, 1):
        items = "、".join(s["itemNames"]) if s["itemNames"] else "（未登记物品）"
        print("    %2d) %-5s %-14s 样本%2d 物品[%s]"
              % (i, s["id"], s["label"], len(s["samples"]), items))
        out.append(s)
    return out


def split_names(text):
    for sep in ("，", ",", "、", "/", "|"):
        text = text.replace(sep, "\n")
    return [t.strip() for t in text.split("\n") if t.strip()]


def main():
    ap = argparse.ArgumentParser(description="暗区鉴宝 · 样本打标")
    ap.add_argument("--root", default=lib.default_root())
    args = ap.parse_args()

    library = lib.open_library(args.root)
    library.ensure_dirs()

    files = inbox_files(library.root)
    if not files:
        print("_inbox 中没有待打标样本。先运行 tools/record_inbox.py 采样。")
        return 0

    print("=" * 66)
    print("待打标样本：%d 条    库：%s" % (len(files), library.root)
          .replace("%s", str(library.root)))
    print("=" * 66)

    changed = 0
    for idx, path in enumerate(files, 1):
        fname = os.path.basename(path)
        try:
            audio, sr = lib.read_wav(path)
        except Exception as e:
            print("[%d/%d] %s  读取失败：%s" % (idx, len(files), fname, e))
            continue

        dur = len(audio) / float(sr) if audio.ndim == 1 else audio.shape[0] / float(sr)
        peak = float(np.abs(audio).max())
        print("\n[%d/%d] %s   %.3fs  peak=%.3f" % (idx, len(files), fname, dur, peak))

        while True:
            print("  已知音效类：")
            classes = show_classes(library)
            print("  [回车]播放  [数字]选择音效类  [n]新建  [s]跳过  [d]删除  [q]退出")
            try:
                ans = input("  选择> ").strip()
            except EOFError:
                ans = "q"

            if ans == "":
                play_file(path)
                continue

            low = ans.lower()
            if low == "q":
                library.save()
                print("\n已保存。本次归档 %d 条样本。" % changed)
                return 0
            if low == "s":
                break
            if low == "d":
                os.remove(path)
                print("  已删除 %s" % fname)
                changed += 1
                break

            if low == "n":
                label = input("  新音效类的名称（如：金属·小件·A）> ").strip()
                if not label:
                    print("  已取消。")
                    continue
                names = split_names(input("  该音效对应的物品名（多个用逗号分隔，可留空）> ").strip())
                sid = next_sound_id(library)
                library.get_or_create_sound(sid, label=label, item_names=names)
                library.add_sample(sid, audio, sr, label=label, item_names=names)
                changed += 1
                print("  已新建 %s「%s」并归档。物品：%s"
                      % (sid, label, "、".join(names) if names else "（未登记）"))
                break

            if ans.isdigit():
                i = int(ans)
                if 1 <= i <= len(classes):
                    target = classes[i - 1]
                    print("  当前音效类：%s「%s」" % (target["id"], target["label"]))
                    names = split_names(input("  这件物品的名字（多个用逗号分隔，留空=不登记）> ").strip())
                    library.add_sample(target["id"], audio, sr,
                                       label=target["label"], item_names=names)
                    changed += 1
                    print("  已归档到 %s。物品：%s"
                          % (target["id"], "、".join(names) if names else "（未登记）"))
                    break
                print("  序号超出范围。")
                continue

            print("  无法识别的输入。")

    library.save()
    print("\n全部处理完毕。本次归档 %d 条样本。" % changed)
    print("库统计：")
    rows, total = library.stats()
    for sid, label, items, n in rows:
        print("  %-5s %-16s 样本%3d  物品[%s]" % (sid, label, n, items))
    print("  合计样本：%d" % total)
    return 0


if __name__ == "__main__":
    sys.exit(main())
