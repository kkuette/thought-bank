#!/usr/bin/env python3
"""Dashboard central de la ferme — stdlib uniquement, à lancer sur une machine
qui monte le share NFS (VM data par défaut). Agrège :
  - $TB_MNT/status/*.json   (écrits par node_agent.sh sur chaque rig)
  - $TB_MNT/queue/          (file, running, done, failed)
  - $TB_MNT/runs/*.workerlog (dernier step + dernière éval par job actif)
Usage : farm_dashboard.py [port] [tb_mnt]   (défauts : 8787, /mnt/tb)
"""
import json, os, re, sys, time
from http.server import HTTPServer, BaseHTTPRequestHandler

TB = sys.argv[2] if len(sys.argv) > 2 else os.environ.get("TB_MNT", "/mnt/tb")
PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8787
STALE_S = 45  # agent muet depuis > STALE_S => nœud offline

STEP_RE = re.compile(r"^step\s+(\d+)\s+ic ([\d.]+) \(ppl ([\d.]+)\).*?([\d.]+)s/step")
EVAL_RE = re.compile(r"^\[eval @(\d+)\] \[(\w+)\].*GAP ([+\-][\d.]+)")


def tail(path, nbytes=6000):
    try:
        with open(path, "rb") as f:
            f.seek(0, 2)
            f.seek(max(0, f.tell() - nbytes))
            return f.read().decode(errors="replace").splitlines()
    except OSError:
        return []


def job_info(job_path):
    name = os.path.basename(job_path).replace(".job", "")
    log = os.path.join(TB, "runs", name + ".workerlog")
    info = {"job": name, "step": None, "evals": []}
    last_evals = {}
    for line in tail(log):
        m = STEP_RE.match(line)
        if m:
            info["step"] = {"n": int(m.group(1)), "ic": float(m.group(2)),
                            "ppl": float(m.group(3)), "sps": float(m.group(4))}
        m = EVAL_RE.match(line)
        if m:
            last_evals[m.group(2)] = {"at": int(m.group(1)), "src": m.group(2),
                                      "gap": float(m.group(3))}
    info["evals"] = list(last_evals.values())
    try:
        info["log_age_s"] = int(time.time() - os.path.getmtime(log))
    except OSError:
        info["log_age_s"] = None
    return info


def snapshot():
    q = os.path.join(TB, "queue")
    ls = lambda d: sorted(
        f for f in (os.listdir(os.path.join(q, d)) if os.path.isdir(os.path.join(q, d)) else [])
        if f.endswith(".job"))
    nodes = []
    sdir = os.path.join(TB, "status")
    if os.path.isdir(sdir):
        for f in sorted(os.listdir(sdir)):
            if not f.endswith(".json"):
                continue
            try:
                n = json.load(open(os.path.join(sdir, f)))
                n["offline"] = (time.time() - n.get("ts", 0)) > STALE_S
                nodes.append(n)
            except (json.JSONDecodeError, OSError):
                pass
    return {
        "ts": int(time.time()),
        "nodes": nodes,
        "queued": [f for f in sorted(os.listdir(q)) if f.endswith(".job")] if os.path.isdir(q) else [],
        "running": [job_info(j) for j in ls("running")],
        "done": ls("done"),
        "failed": ls("failed"),
    }


PAGE = """<!doctype html><html><head><meta charset="utf-8"><title>ferme tb</title>
<style>
 body{background:#101418;color:#cdd6e0;font:14px/1.5 monospace;margin:20px;max-width:1100px}
 h1{font-size:18px;color:#7ec8a8} h2{font-size:15px;color:#8fb4d8;margin:18px 0 6px}
 table{border-collapse:collapse;width:100%} td,th{padding:3px 10px;text-align:left;border-bottom:1px solid #232a31}
 th{color:#67707a;font-weight:normal} .ok{color:#7ec8a8} .warn{color:#e0b060} .bad{color:#e07070}
 .dim{color:#5a636d} .bar{display:inline-block;height:9px;background:#2e6e54;vertical-align:middle;border-radius:2px}
 .barbox{display:inline-block;width:90px;height:9px;background:#20262c;border-radius:2px;margin-right:6px}
</style></head><body>
<h1>ferme thought-bank</h1><div id="c">chargement…</div>
<script>
const pct=(a,b)=>b?Math.round(100*a/b):0;
const bar=(a,b)=>`<span class="barbox"><span class="bar" style="width:${pct(a,b)*0.9}px"></span></span>${pct(a,b)}%`;
async function r(){
 let d;try{d=await(await fetch('data.json')).json()}catch(e){document.getElementById('c').innerHTML='<span class=bad>dashboard injoignable</span>';return}
 let h='';
 for(const n of d.nodes){
  h+=`<h2>${n.host} ${n.offline?'<span class=bad>OFFLINE</span>':'<span class=ok>en ligne</span>'}
      <span class=dim>load ${n.load} · RAM ${n.mem_mb[0]}/${n.mem_mb[1]} Mo · swap ${n.swap_mb[0]}/${n.swap_mb[1]} Mo</span></h2>`;
  h+='<table><tr><th>gpu</th><th>util</th><th>vram</th><th>W</th><th>°C</th></tr>';
  for(const g of n.gpus||[]) h+=`<tr><td>${g.i}</td><td>${bar(g.util,100)}</td><td>${bar(g.vram,g.vram_tot)} <span class=dim>${g.vram} Mo</span></td><td>${g.w}</td><td class="${g.temp>80?'bad':g.temp>70?'warn':''}">${g.temp}</td></tr>`;
  h+='</table>';}
 h+=`<h2>jobs actifs (${d.running.length})</h2><table><tr><th>job</th><th>step</th><th>ic</th><th>s/step</th><th>dernier GAP</th><th>log il y a</th></tr>`;
 for(const j of d.running){
  const s=j.step, e=(j.evals||[]).map(x=>`${x.src}@${x.at}: <b>${x.gap>0?'+':''}${x.gap}</b>`).join(' · ');
  const age=j.log_age_s==null?'?':(j.log_age_s>300?`<span class=bad>${Math.round(j.log_age_s/60)} min</span>`:`${j.log_age_s} s`);
  h+=`<tr><td>${j.job}</td><td>${s?s.n:'<span class=dim>init…</span>'}</td><td>${s?s.ic:''}</td><td>${s?s.sps:''}</td><td>${e||'<span class=dim>—</span>'}</td><td>${age}</td></tr>`;}
 h+='</table>';
 h+=`<h2>file (${d.queued.length})</h2><div class=dim>${d.queued.join('<br>')||'vide'}</div>`;
 h+=`<h2>terminés (${d.done.length}) — <span class=bad>échecs (${d.failed.length})</span></h2>`;
 h+=`<div class=dim>${d.done.join('<br>')||'—'}</div>`;
 if(d.failed.length) h+=`<div class=bad>${d.failed.join('<br>')}</div>`;
 h+=`<p class=dim>maj ${new Date(d.ts*1000).toLocaleTimeString()}</p>`;
 document.getElementById('c').innerHTML=h;}
r();setInterval(r,10000);
</script></body></html>"""


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):  # silence
        pass

    def do_GET(self):
        if self.path.startswith("/data.json"):
            body, ctype = json.dumps(snapshot()).encode(), "application/json"
        else:
            body, ctype = PAGE.encode(), "text/html; charset=utf-8"
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


if __name__ == "__main__":
    print(f"dashboard sur 0.0.0.0:{PORT}, TB_MNT={TB}")
    HTTPServer(("0.0.0.0", PORT), H).serve_forever()
