# -*- coding: utf-8 -*-
"""实时识别：监听系统输出 -> 切分音效 -> 与声音库比对 -> 控制台 + 手机网页显示。

用法：
    python tools/match_live.py                      # 仅控制台输出
    python tools/match_live.py --http-port 8765     # 同时开启手机网页
    手机与电脑连同一 WiFi，浏览器打开 http://<本机IP>:8765

合规：只读取操作系统混音输出，不接触游戏进程、不注入、不截图、不自动化操作。
"""

import argparse
import json
import os
import socket
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

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

_LOCK = threading.Lock()
STATE = {
    "stage": "启动中",
    "matched": False,
    "ts": "",
    "soundId": "",
    "label": "",
    "items": [],
    "similarity": 0.0,
    "others": [],
    "events": 0,
    "uptime": 0,
}


def set_state(**kw):
    with _LOCK:
        STATE.update(kw)


def snapshot():
    with _LOCK:
        return dict(STATE)


PAGE = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1,user-scalable=no">
<title>暗区鉴宝</title>
<style>
  body{margin:0;padding:14px;font-family:-apple-system,"Segoe UI","Microsoft YaHei",sans-serif;
       background:#f2f3f5;color:#1c1c1e;-webkit-text-size-adjust:100%}
  .card{background:#fff;border-radius:14px;padding:16px 18px;margin-bottom:12px;
        box-shadow:0 1px 3px rgba(0,0,0,.07)}
  .row{display:flex;align-items:center;gap:8px}
  .dot{width:9px;height:9px;border-radius:50%;background:#30c46a;flex:0 0 auto}
  .dot.off{background:#c9ccd1}
  .big{font-size:29px;font-weight:600;line-height:1.3;margin-top:10px;word-break:break-all}
  .big.empty{color:#9aa0a6;font-size:22px;font-weight:500}
  .sub{font-size:13px;color:#8a8f98}
  .bar{height:8px;border-radius:4px;background:#e9ebee;overflow:hidden;margin-top:12px}
  .bar>i{display:block;height:100%;width:0;background:#1a7f4b;transition:width .25s}
  ul{padding-left:18px;margin:8px 0 0}
  li{margin:5px 0;font-size:15px;line-height:1.45}
  .tag{display:inline-block;font-size:11px;color:#7a8088;border:1px solid #dfe2e6;
       border-radius:5px;padding:1px 6px;margin-left:6px;vertical-align:1px}
</style>
</head>
<body>
  <div class="card">
    <div class="row"><span class="dot off" id="dot"></span><span class="sub" id="status">连接中…</span></div>
    <div class="big empty" id="title">等待音效</div>
    <div class="sub" id="meta">拖动物品试听</div>
    <div class="bar"><i id="sim"></i></div>
  </div>
  <div class="card">
    <div class="sub">这件可能是</div>
    <ul id="items"><li class="sub">—</li></ul>
  </div>
  <div class="card">
    <div class="sub">其他相近音效</div>
    <ul id="others"><li class="sub">—</li></ul>
  </div>
<script>
var lastTs = "", fails = 0;
function esc(s){var d=document.createElement('div');d.textContent=s==null?'':String(s);return d.innerHTML;}
function tick(){
  fetch('/api/live',{cache:'no-store'}).then(function(r){return r.json();}).then(function(d){
    fails = 0;
    document.getElementById('dot').className = 'dot';
    document.getElementById('status').textContent = '已连接 · ' + (d.stage||'') + ' · 已捕获 ' + (d.events||0) + ' 次';
    if(d.ts === lastTs) return;
    lastTs = d.ts;
    var t = document.getElementById('title');
    if(d.matched){
      t.className = 'big';
      t.textContent = d.label || d.soundId;
      document.getElementById('meta').textContent =
        '相似度 ' + ((d.similarity||0)*100).toFixed(1) + '%  ·  ' + (d.ts||'');
      document.getElementById('sim').style.width = Math.round((d.similarity||0)*100) + '%';
      var ul = document.getElementById('items');
      ul.innerHTML = '';
      var items = d.items && d.items.length ? d.items : ['（该音效尚未登记物品名）'];
      items.forEach(function(n){
        var li = document.createElement('li'); li.textContent = n; ul.appendChild(li);
      });
      if(items.length > 1){
        var li = document.createElement('li');
        li.innerHTML = '<span class="tag">共用音效 · 需结合格数/标价判断</span>';
        ul.appendChild(li);
      }
      var ol = document.getElementById('others');
      ol.innerHTML = '';
      (d.others||[]).forEach(function(o){
        var li = document.createElement('li');
        li.textContent = o.label + '  (' + (o.similarity*100).toFixed(0) + '%)';
        ol.appendChild(li);
      });
      if(!(d.others||[]).length){ ol.innerHTML = '<li class="sub">—</li>'; }
    }else{
      t.className = 'big empty';
      t.textContent = '未命中';
      document.getElementById('meta').textContent = d.note || '音效不在库中，或相似度不足';
      document.getElementById('sim').style.width = '0%';
      document.getElementById('items').innerHTML = '<li class="sub">—</li>';
      document.getElementById('others').innerHTML = '<li class="sub">—</li>';
    }
  }).catch(function(){
    fails++;
    document.getElementById('dot').className = 'dot off';
    document.getElementById('status').textContent = '连接中断（重试 ' + fails + '）';
  });
}
setInterval(tick, 500);
tick();
</script>
</body>
</html>
"""


def make_handler():
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def _send(self, body, ctype):
            if isinstance(body, str):
                body = body.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            try:
                self.wfile.write(body)
            except Exception:
                pass

        def do_GET(self):
            p = self.path.split("?")[0]
            if p in ("/", "/index.html"):
                self._send(PAGE, "text/html; charset=utf-8")
            elif p == "/api/live":
                self._send(json.dumps(snapshot(), ensure_ascii=False),
                           "application/json; charset=utf-8")
            elif p == "/health":
                self._send("ok", "text/plain; charset=utf-8")
            else:
                self.send_response(404)
                self.send_header("Content-Length", "0")
                self.end_headers()

        def do_POST(self):
            self.send_response(405)
            self.send_header("Content-Length", "0")
            self.end_headers()

        def log_message(self, *a):
            pass

    return Handler


def local_ip():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        return s.getsockname()[0]
    except Exception:
        return "127.0.0.1"
    finally:
        s.close()


def main():
    ap = argparse.ArgumentParser(description="暗区鉴宝 · 实时识别")
    ap.add_argument("--device", default=None)
    ap.add_argument("--root", default=lib.default_root())
    ap.add_argument("--min-sim", type=float, default=None,
                    help="命中所需最低相似度；留空则读 calibration.json")
    ap.add_argument("--top-k", type=int, default=3)
    ap.add_argument("--http-port", type=int, default=0, help="0 表示不开启网页")
    ap.add_argument("--bind", default="0.0.0.0")
    args = ap.parse_args()

    library = lib.open_library(args.root)
    rows, total = library.stats()
    if total == 0:
        print("声音库为空，无法匹配。先运行 record_inbox.py + label_inbox.py 建库。")
        return 3

    # 命中阈值随特征参数变化（启用 CMN 后相似度绝对尺度整体下移），
    # 因此优先读标定文件，而不是写死一个数。
    min_sim = args.min_sim
    sim_src = "命令行指定"
    if min_sim is None:
        calib = os.path.join(args.root, "calibration.json")
        if os.path.isfile(calib):
            try:
                with open(calib, "r", encoding="utf-8") as f:
                    min_sim = float(json.load(f)["recommendedMinSim"])
                sim_src = "标定文件"
            except Exception:
                min_sim = None
        if min_sim is None:
            min_sim = 0.55
            sim_src = "未标定，使用保守默认"

    print("=" * 66)
    print("暗区鉴宝 · 实时识别")
    print("  声音库   : %s" % library.root)
    print("  音效类   : %d 类 / %d 条样本" % (len(rows), total))
    for sid, label, items, n in rows:
        print("     %-5s %-16s 样本%3d  物品[%s]" % (sid, label, n, items))
    print("  命中门限 : 相似度 >= %.4f   （%s）" % (min_sim, sim_src))
    if sim_src.startswith("未标定"):
        print("             建议先跑 tools/calibrate.py 标定该阈值")
    print("=" * 66)

    httpd = None
    if args.http_port:
        httpd = ThreadingHTTPServer((args.bind, args.http_port), make_handler())
        th = threading.Thread(target=httpd.serve_forever, daemon=True)
        th.start()
        print("手机端已开启： http://%s:%d" % (local_ip(), args.http_port))
        print("  （手机需与本机处于同一 WiFi；浏览器直接打开该地址）")
        print("-" * 66)

    try:
        cap = cap_mod.LoopbackCapture(device_name=args.device)
        cap.start()
    except Exception as e:
        print("采集启动失败：%s" % e)
        if httpd:
            httpd.shutdown()
        return 2

    print("采集设备：%s" % cap.device.name)
    det = det_mod.SoundEventDetector(samplerate=cap.samplerate,
                                     block_size=cap.blocksize)
    set_state(stage="监听中")

    events = 0
    t0 = time.time()
    try:
        while True:
            block = cap.read(timeout=0.5)
            if block is None:
                if cap.error:
                    print("采集错误：%s" % cap.error)
                    break
                set_state(stage="监听中", uptime=int(time.time() - t0))
                continue

            evt = det.push(block)
            if evt is None:
                continue

            events += 1
            results, meta = library.match(evt, cap.samplerate, top_k=args.top_k)
            stamp = time.strftime("%H:%M:%S")

            if not results:
                set_state(stage="监听中", ts=stamp, matched=False, events=events,
                          note="库中无可用样本")
                continue

            top = results[0]
            matched = top["similarity"] >= min_sim
            others = [{"label": r["label"], "similarity": r["similarity"]}
                      for r in results[1:]]
            set_state(stage="监听中", ts=stamp, matched=matched, events=events,
                      soundId=top["soundId"], label=top["label"],
                      items=top["itemNames"], similarity=top["similarity"],
                      others=others,
                      note="音效不在库中，或相似度不足（最高 %.2f）" % top["similarity"])

            flag = "命中" if matched else "忽略"
            print("  [%s] #%-3d %.3fs  %s  %s  相似度=%.3f  %s"
                  % (stamp, events, len(evt) / float(cap.samplerate),
                     flag, top["soundId"], top["similarity"],
                     "、".join(top["itemNames"]) if top["itemNames"] else ""))
            if matched and len(top["itemNames"]) > 1:
                print("         注意：该音效与 %d 件物品共用，需结合占用格数与标价判断"
                      % len(top["itemNames"]))
            for r in results[1:]:
                print("         · 次选 %s「%s」 相似度=%.3f"
                      % (r["soundId"], r["label"], r["similarity"]))
    except KeyboardInterrupt:
        print("\n收到中断，正在停止 ...")
    finally:
        cap.stop()
        if httpd:
            httpd.shutdown()

    print("共处理 %d 次音效事件。检测器统计：%s" % (events, det.stats))
    return 0


if __name__ == "__main__":
    sys.exit(main())
