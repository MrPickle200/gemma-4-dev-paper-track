# Colab GPU backend for the Bước 4 grid

The harness only ever speaks **OpenAI-compatible HTTP**, so the model server can live on a Colab
GPU while the agent loop keeps running here (worktrees, traces, patches stay local). This directory
holds the notebook that provides that endpoint.

## Why

The 6 GB laptop GPU runs one `llama-server` slot at a time, so the 400-cell grid is serialised
(~2–5 min/cell). A Colab GPU runs the **same** llama.cpp build and the same Q4_0 GGUF (so results
stay comparable) with several slots, and `run_batch.py --workers N` then fills all of them.

## Use

1. Open `colab/gemma4_backend.ipynb` in Colab. **Runtime → Change runtime type → T4 GPU**
   (an L4 or A100 fits more slots and is the better use of Pro compute units).
2. `Runtime → Run all`. The notebook is **safe to re-run**: it kills stale servers/tunnels, picks a
   free port, restores the build + weights from a Google Drive cache, and auto-sizes the slots.
   It prints three values:

   ```
   BASE_URL = https://<random>.trycloudflare.com
   API_KEY  = <random>
   WORKERS  = <n>          # = the slot count that fit your GPU
   ```

3. **Stop the local batch first** (otherwise two batches contend for the same grid cells), then:

   ```powershell
   .\.venv\Scripts\python.exe scripts\run_batch.py --model gemma-4-E2B `
       --variants B0 M1 M2 M3 --thinking on --max-turns 15 --workers <n> `
       --base-url https://<random>.trycloudflare.com --api-key <random>
   ```

4. The grid is resumable, so if Colab drops the session, re-run the notebook and repeat the same
   command; finished cells are skipped.

## Slots vs VRAM (the one tuning knob)

Each slot needs `32768` tokens of context, and llama.cpp allocates KV for `slots * 32768` tokens.
Gemma 4 E2B Q4_0 costs roughly **100 KB of KV per token**, so:

| GPU | VRAM | slots that fit | `--workers` |
|---|---|---|---|
| T4 | 15 GB | ~3 | 3 |
| L4 | 24 GB | ~4–5 | 4–5 |
| A100 | 40 GB | 8 | 8 |

**Never lower `CTX_PER_SLOT`.** It must equal the laptop server's `-c`; a smaller window truncates
long prompts and changes agent behaviour, which breaks comparability with the laptop traces. Scale
*slots* to fit VRAM instead. The notebook's serve cell tries `8,6,4,3,2,1` and reports which worked.

## Parity with the local server

| | local (`scripts/serve.ps1`) | Colab notebook |
|---|---|---|
| llama.cpp | release `b11696` (win-cuda) | git tag `b11696`, built with `-DGGML_CUDA=ON` |
| weights | `google/gemma-4-E2B-it-qat-q4_0-gguf` | same repo / same file |
| flags | `-ngl 99 --jinja --flash-attn on -c 32768` | same, plus `-np <slots>` |
| sampling | `temperature 0.0`, `max_tokens 16384` (harness-side) | identical |

`run_start` records `base_url` and `api_key_set` (never the key), so local and remote traces stay
distinguishable even though they share the same `model` label — and they are the same experiment, so
they can be mixed in the ablation table.

## Troubleshooting seen so far

- **404 from Cloudflare's edge on every path** → the tunnel hostname has no live `cloudflared`
  (session restarted, or an older URL). Re-run the tunnel cell and use the new URL.
- **502 from Cloudflare** → the tunnel is alive but the **origin is gone**: `llama-server` crashed or
  the Colab runtime recycled. Requests in flight therefore fail and `run_batch.py` aborts on its
  health check. Those runs are recorded with `stop_reason=llama_error` (`llm_error`) and are now
  treated as *incomplete*: `run_batch.py` re-runs them and `summarize_batch.py` excludes them, so a
  backend crash never pollutes the ablation table.
- **`llama_server: exiting due to HTTP server error` / `couldn't bind HTTP server socket`** → the
  port was already taken by a stale server. The notebook now kills stale processes and picks a free
  port, so this should not recur.
- **Server dies at model load on a T4** → `--flash-attn on` is unsupported on Turing; the notebook
  automatically retries with `auto`.
- **Server starts but dies later under load (the edge then returns 502)** → too many slots for the
  instance, and the limiter is usually **host RAM, not VRAM**. The Colab T4 VM has only ~12.7 GB of
  system RAM, and 8 slots x 32768 context was OOM-killed by the *host* while `nvidia-smi` still looked
  comfortable. The runtime survives, so the tunnel stays connected and the edge answers **502** rather
  than 530/404 - only `llama-server` dies. The notebook caps slots by a conservative KV budget
  (`MAX_SLOTS`: T4 -> 3, L4 -> 6, A100 -> 8); lower it further if it recurs.
- **530 "Cloudflare Tunnel error"** → `cloudflared` is not connected (never started, or killed by a
  re-run of the settings cell). The hostname is dead and will never come back: re-run the tunnel cell.
  A quick-tunnel URL is new on **every** start.

## Recovering a dead origin without changing the URL

If `llama-server` dies but `cloudflared` survives, the hostname keeps working and **502** is the only
symptom - so restarting the server on the **same port** restores the *same* URL and key, with no
change needed on the laptop. One paste:

```python
import glob, subprocess, time, requests, secrets
def sh(c): return subprocess.run(["bash","-lc",c], capture_output=True, text=True).stdout
print(sh("free -m"))
print(sh("pgrep -af 'llama-server|cloudflared' || echo '(no processes)'"))
print(sh("sudo dmesg 2>/dev/null | grep -iE 'killed process|out of memory' | tail -n 3"))

GGUF    = globals().get("GGUF")    or glob.glob("models/*.gguf")[0]
PORT    = globals().get("PORT")    or 8080     # must match the port the live tunnel points at
API_KEY = globals().get("API_KEY") or secrets.token_urlsafe(24)
ram_gib = __import__("psutil").virtual_memory().total / 2**30
SLOTS   = max(1, min(8, int((ram_gib - 3.0) // 2.0)))

subprocess.run(["bash","-lc","pkill -9 -f build/bin/llama-server || true"], capture_output=True)
time.sleep(3)
p = subprocess.Popen(["./llama.cpp/build/bin/llama-server","-m",GGUF,"-c","32768","-np",str(SLOTS),
                      "-ngl","99","--jinja","--flash-attn","auto","--host","127.0.0.1",
                      "--port",str(PORT),"--metrics","--api-key",API_KEY],
                     stdout=open("server_recover.log","w"), stderr=subprocess.STDOUT)
for _ in range(360):
    if p.poll() is not None:
        print("exited:\n" + sh("tail -n 6 server_recover.log")); break
    if requests.get(f"http://127.0.0.1:{PORT}/health", timeout=3).status_code == 200:
        print(f"SERVING slots={SLOTS} port={PORT}  ->  --workers {SLOTS}"); break
    time.sleep(2)
```

If `cloudflared` is gone too, re-run the tunnel cell instead and expect a **new** hostname.
