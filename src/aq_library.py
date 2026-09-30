# -*- coding: utf-8 -*-
"""声音库：样本落地、索引维护、模板聚合、匹配查询。

目录结构：
    data/library/
        library.json          # 库索引
        items.json            # 物品表（名称、格数、品质、参考价）
        samples/<soundId>/    # 已打标的样本 wav
        _inbox/               # 待打标样本

设计要点：
  - 每类音效可有多条样本；匹配时对每条样本取最优分数，再按类聚合。
  - library.json 只存元数据与相对路径，样本本体是独立 wav 文件，可随时增删。
"""

import json
import os
import time
import wave

import numpy as np

import aq_dsp as dsp

SCHEMA_VERSION = 1


# ---------------------------------------------------------------- wav I/O

def write_wav(path, audio, sr):
    """写 16bit PCM wav。audio 支持 (n,) 或 (n, 2)。"""
    a = np.asarray(audio)
    if a.ndim == 1:
        a = a[:, None]
    a = np.clip(a, -1.0, 1.0)
    pcm = (a * 32767.0).astype("<i2")
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with wave.open(path, "wb") as w:
        w.setnchannels(a.shape[1])
        w.setsampwidth(2)
        w.setframerate(int(sr))
        w.writeframes(pcm.tobytes())
    return path


def read_wav(path):
    """读 wav，返回 (float32 音频, 采样率)。多声道保持二维。"""
    with wave.open(path, "rb") as w:
        n_ch = w.getnchannels()
        sr = w.getframerate()
        width = w.getsampwidth()
        raw = w.readframes(w.getnframes())
    if width != 2:
        raise ValueError("只支持 16bit PCM wav：%s" % path)
    data = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    if n_ch > 1:
        data = data.reshape(-1, n_ch)
    return data, sr


# ---------------------------------------------------------------- 库本体

class SoundLibrary(object):

    def __init__(self, root):
        self.root = os.path.abspath(root)
        self.index_path = os.path.join(self.root, "library.json")
        self.items_path = os.path.join(self.root, "items.json")
        self.samples_dir = os.path.join(self.root, "samples")
        self.inbox_dir = os.path.join(self.root, "_inbox")

        self.db = {"schemaVersion": SCHEMA_VERSION, "sounds": []}
        self.items = {"schemaVersion": SCHEMA_VERSION, "items": []}
        self._feat_cache = {}
        self._templates = None

    # ------------------------------------------------------------ 载入/保存

    def ensure_dirs(self):
        for d in (self.root, self.samples_dir, self.inbox_dir):
            os.makedirs(d, exist_ok=True)

    def load(self):
        self.ensure_dirs()
        if os.path.isfile(self.index_path):
            with open(self.index_path, "r", encoding="utf-8") as f:
                self.db = json.load(f)
        if os.path.isfile(self.items_path):
            with open(self.items_path, "r", encoding="utf-8") as f:
                self.items = json.load(f)
        self._templates = None
        self._feat_cache = {}
        return self

    def save(self):
        self.ensure_dirs()
        self.db["savedAt"] = time.strftime("%Y-%m-%d %H:%M:%S")
        self.db["schemaVersion"] = SCHEMA_VERSION
        with open(self.index_path, "w", encoding="utf-8") as f:
            json.dump(self.db, f, ensure_ascii=False, indent=2)
        with open(self.items_path, "w", encoding="utf-8") as f:
            json.dump(self.items, f, ensure_ascii=False, indent=2)

    # ------------------------------------------------------------ 音效类

    def find_sound(self, sound_id):
        for s in self.db["sounds"]:
            if s["id"] == sound_id:
                return s
        return None

    def get_or_create_sound(self, sound_id, label=None, item_names=None):
        s = self.find_sound(sound_id)
        if s is None:
            s = {
                "id": sound_id,
                "label": label or sound_id,
                "itemNames": list(item_names or []),
                "sampleCount": 0,
                "samples": [],
                "createdAt": time.strftime("%Y-%m-%d %H:%M:%S"),
            }
            self.db["sounds"].append(s)
        else:
            if label:
                s["label"] = label
            if item_names:
                for n in item_names:
                    if n not in s["itemNames"]:
                        s["itemNames"].append(n)
        return s

    def sounds(self):
        return self.db["sounds"]

    def add_sample(self, sound_id, audio, sr, label=None, item_names=None,
                   note="", trim=True):
        """把一段音频作为样本写入库。返回写入的 wav 路径。

        trim=True 时落盘的是裁剪静音后的有效音频段。库里不该存静音：
        静音帧会参与 DTW 对齐、拉低跨时长配对分数，并让样本时长统计失真。
        """
        s = self.get_or_create_sound(sound_id, label, item_names)
        out_dir = os.path.join(self.samples_dir, sound_id)
        os.makedirs(out_dir, exist_ok=True)

        data = np.asarray(audio)
        if trim:
            data = dsp.trim_silence(data, sr)

        idx = len(s["samples"]) + 1
        fname = "%s_%03d.wav" % (time.strftime("%Y%m%d_%H%M%S"), idx)
        path = os.path.join(out_dir, fname)
        write_wav(path, data, sr)

        rel = os.path.relpath(path, self.root).replace("\\", "/")
        _, meta = dsp.extract(data, sr, trim=False)
        s["samples"].append({
            "file": rel,
            "addedAt": time.strftime("%Y-%m-%d %H:%M:%S"),
            "duration": meta["duration"],
            "peak": meta["peak"],
            "rms": meta["rms"],
            "note": note,
        })
        s["sampleCount"] = len(s["samples"])
        self._templates = None
        return path

    def remove_sample(self, sound_id, rel_file):
        s = self.find_sound(sound_id)
        if not s:
            return False
        keep = []
        removed = False
        for smp in s["samples"]:
            if smp["file"] == rel_file:
                full = os.path.join(self.root, rel_file)
                if os.path.isfile(full):
                    os.remove(full)
                removed = True
                continue
            keep.append(smp)
        s["samples"] = keep
        s["sampleCount"] = len(keep)
        self._templates = None
        return removed

    # ------------------------------------------------------------ 特征与模板

    def _sample_features(self, sound, smp):
        key = smp["file"]
        if key in self._feat_cache:
            return self._feat_cache[key]
        full = os.path.join(self.root, smp["file"])
        if not os.path.isfile(full):
            self._feat_cache[key] = None
            return None
        audio, sr = read_wav(full)
        feat, _ = dsp.extract(audio, sr)
        self._feat_cache[key] = feat
        return feat

    def build_templates(self):
        """为每个音效类构建模板集合（该类全部样本的特征）。

        P0 阶段不做聚类平均 —— 样本少时逐条比对更稳。样本量上来后可改为质心模板。
        """
        templates = []
        for s in self.db["sounds"]:
            feats = []
            for smp in s["samples"]:
                f = self._sample_features(s, smp)
                if f is not None and f.shape[0] > 0:
                    feats.append(f)
            if feats:
                templates.append({"soundId": s["id"], "label": s["label"],
                                  "itemNames": s["itemNames"], "feats": feats})
        self._templates = templates
        return templates

    def stats(self):
        total = 0
        rows = []
        for s in self.db["sounds"]:
            n = len(s["samples"])
            total += n
            rows.append((s["id"], s["label"], " / ".join(s["itemNames"]), n))
        return rows, total

    # ------------------------------------------------------------ 匹配

    def match(self, audio, sr, top_k=5, band=0.3):
        """把一段音频与库中所有样本比对，返回按相似度降序的结果。

        返回列表，每项：
            {soundId, label, itemNames, similarity, distance, sampleFile}
        只在库非空时有效。
        """
        if self._templates is None:
            self.build_templates()
        if not self._templates:
            return []

        query, meta = dsp.extract(audio, sr)
        if query.shape[0] == 0:
            return []

        best = {}
        for tpl in self._templates:
            for feat in tpl["feats"]:
                d = dsp.dtw_distance(query, feat, band=band)
                cur = best.get(tpl["soundId"])
                if cur is None or d < cur:
                    best[tpl["soundId"]] = d

        out = []
        for tpl in self._templates:
            d = best.get(tpl["soundId"])
            if d is None:
                continue
            out.append({
                "soundId": tpl["soundId"],
                "label": tpl["label"],
                "itemNames": list(tpl["itemNames"]),
                "distance": round(float(d), 4),
                "similarity": round(max(0.0, 1.0 - d * 2.0), 4),
                "candidateCount": len(tpl["itemNames"]) or None,
            })
        out.sort(key=lambda r: (r["distance"], r["soundId"]))
        return out[:top_k], meta


# ---------------------------------------------------------------- 便捷构造

def open_library(root):
    return SoundLibrary(root).load()


def default_root():
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(here, "data", "library")
