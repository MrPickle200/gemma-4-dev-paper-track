import os
import sys
import time
import json
from dotenv import load_dotenv

sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
sys.stderr.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)

load_dotenv(r"D:\Projects\kaggle\.env")
from kaggle.api.kaggle_api_extended import KaggleApi
import kagglesdk.kernels.types.kernels_api_service as kas

api = KaggleApi()
api.authenticate()

import argparse
parser = argparse.ArgumentParser(description="Monitor Kaggle Kernel")
parser.add_argument("--slug", type=str, default="daoviet/gemma-4-e4b-m4-sft", help="Kernel slug")
args, _ = parser.parse_known_args()

slug = args.slug
user_name = slug.split("/")[0]
kernel_slug = slug.split("/")[1]

print(f"=== Robust Monitor for Kaggle Kernel: {slug} ===")

consecutive_stream_errors = 0

while True:
    try:
        status_obj = api.kernels_status(slug)
        status_str = str(getattr(status_obj, "status", status_obj))
        curr_time = time.strftime("%X")
        print(f"[{curr_time}] Kernel status: {status_str}")

        if "COMPLETE" in status_str:
            print("=== SUCCESS: Kernel COMPLETED! ===")
            break

        if "ERROR" in status_str:
            print("=== FAILURE: Kernel encountered ERROR! ===")
            if getattr(status_obj, "failure_message", None):
                print(f"Failure message: {status_obj.failure_message}")
            sys.exit(1)

        if "RUNNING" in status_str:
            # Connect to stream
            with api.build_kaggle_client() as client:
                req = kas.ApiGetKernelSessionLogsStreamRequest()
                req.user_name = user_name
                req.kernel_slug = kernel_slug
                res = client.kernels.kernels_api_client.get_kernel_session_logs_stream(req)
                raw = getattr(res, "response", None) or res

                for line in raw.iter_lines(decode_unicode=True):
                    if not line:
                        continue
                    if line.startswith("data: "):
                        try:
                            payload = json.loads(line[6:])
                            stream_name = payload.get("stream_name", "")
                            data = payload.get("data", "")
                            if stream_name == "stdout":
                                print(f"[STDOUT] {data}", end="")
                            else:
                                if any(k in data.lower() for k in ["error", "exception", "traceback", "loss", "step", "epoch"]):
                                    print(f"[STDERR] {data}", end="")
                        except Exception:
                            pass
            consecutive_stream_errors = 0

    except Exception as e:
        consecutive_stream_errors += 1
        print(f"[{time.strftime('%X')}] Stream notice ({e}), retrying status in 10s...")
        time.sleep(10)
        if consecutive_stream_errors > 20:
            print("Too many consecutive stream errors. Checking status once more...")
            st = api.kernels_status(slug)
            print("Current status:", st)
            time.sleep(30)
            consecutive_stream_errors = 0

print("Exiting monitor.")
