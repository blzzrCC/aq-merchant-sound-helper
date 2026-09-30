# -*- coding: utf-8 -*-
"""采样工具：监听系统输出音频，自动切出拾取音，存进 _inbox 等待打标。

用法：
    python tools/record_inbox.py                     # 默认扬声器 loopback
    python tools/record_inbox.py --device "耳机"      # 指定 loopback 端点
    python tools/record_inbox.py --list              # 只列出可用设备
"""

import argparse
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import aq_capture as cap_mod     # noqa: E402
import aq_detect as det_mod      # noqa: E402
import aq_library as lib         # noqa: E402


def main():
    ap = argparse.ArgumentParser(description="暗区鉴宝 · 音效采样工具")
    ap.add_argument("--device", default=None, help="loopback 设备名（支持部分匹配）")
    ap.add_argument("--root", default=lib.default_root(), help="声音库根目录")
    ap.add_argument("--list", action="store_true", help="列出可用设备后退出")
    ap.add_argument("--duration", type=float, default=0.0, help="运行秒数，0 表示一直运行")
    ap.add_argument("--threshold", type=float, default=0.012, help="触发门限（RMS）")
    ap.add_argument("--noise-mult", type=float, default=4.0, help="相对噪声底的倍数")
    ap.add_argument("--quiet", action="store_true", help="不打印每次捕获详情")
    args = ap.parse_args()

    if args.list:
        print("可用的 loopback 设备：")
        for name, is_spk, is_mic in cap_mod.list_devices():
            tag = []
            if is_spk:
                tag.append("默认扬声器")
            if is_mic:
                tag.append("默认输入")
            print("  %-46s %s" % (name, " ".join(tag)))
        return 0

    library = lib.open_library(args.root)
    library.ensure_dirs()

    try:
        cap = cap_mod.LoopbackCapture(device_name=args.device)
        cap.start()
    except Exception as e:
        print("采集启动失败：%s" % e)
        return 2

    print("=" * 66)
    print("暗区鉴宝 · 采样工具")
    print("  采集设备 : %s" % cap.device.name)
    print("  采样率   : %d Hz   块大小: %d 帧" % (cap.samplerate, cap.blocksize))
    print("  样本落盘 : %s" % library.inbox_dir)
    print("  提示     : 进入游戏拖动物品。捕获到音效会自动存盘。Ctrl+C 结束。")
    print("=" * 66)

    det = det_mod.SoundEventDetector(samplerate=cap.samplerate,
                                     block_size=cap.blocksize,
                                     abs_threshold=args.threshold,
                                     noise_mult=args.noise_mult)

    saved = 0
    t0 = time.time()
    last_report = t0
    try:
        while True:
            block = cap.read(timeout=0.5)
            if block is None:
                if cap.error:
                    print("采集错误：%s" % cap.error)
                    break
                if args.duration and time.time() - t0 > args.duration:
                    break
                now = time.time()
                if now - last_report > 10.0:
                    print("  ...监听中  已捕获 %d 条" % saved)
                    last_report = now
                continue

            evt = det.push(block)
            if evt is None:
                continue

            stamp = time.strftime("%Y%m%d_%H%M%S")
            fname = "inbox_%s_%02d.wav" % (stamp, saved + 1)
            path = os.path.join(library.inbox_dir, fname)
            lib.write_wav(path, evt, cap.samplerate)
            saved += 1

            peak = float(np.abs(evt).max())
            if not args.quiet:
                print("  [%s] #%-3d  %.3fs  peak=%.3f  ->  %s"
                      % (time.strftime("%H:%M:%S"), saved,
                         len(evt) / float(cap.samplerate), peak, fname))

            if args.duration and time.time() - t0 > args.duration:
                break
    except KeyboardInterrupt:
        print("\n收到中断，正在停止 ...")
    finally:
        cap.stop()

    print("-" * 66)
    print("本次共保存 %d 条样本 -> %s" % (saved, library.inbox_dir))
    print("检测器统计：%s" % det.stats)
    if saved:
        print("下一步：运行 tools/label_inbox.py 给样本打标归档")
    return 0


if __name__ == "__main__":
    sys.exit(main())
