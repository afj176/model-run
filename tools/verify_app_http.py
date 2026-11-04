"""Opt-in checkpoint exercise against a running Model Run server; no personal paths."""
import argparse
import json
import time
from urllib.request import Request, urlopen

parser = argparse.ArgumentParser()
parser.add_argument("url")
parser.add_argument("model_name")
parser.add_argument("--diffusion", action="store_true")
parser.add_argument("--keep-loaded", action="store_true")
args = parser.parse_args()
base = args.url.rstrip("/") + "/api/"
with urlopen(base + "session") as response: token = json.load(response)["token"]
def call(route, body=None):
    request = Request(base + route, data=None if body is None else json.dumps(body).encode(),
                      headers={"X-Model-Run-Token": token, "Content-Type": "application/json"})
    with urlopen(request, timeout=30) as response: return json.load(response)
def wait():
    deadline = time.monotonic() + 300
    while time.monotonic() < deadline:
        state = call("state")
        if state["job"]["status"] != "running":
            assert state["job"]["status"] == "completed", state
            return state
        time.sleep(.2)
    raise TimeoutError("Application job exceeded verification deadline")
catalog = call("catalog")
model = next(a for a in catalog["artifacts"] if a["name"] == args.model_name and a["format"] == "local_package")
call("load", {"id": model["id"], "device": "cuda"})
wait()
assert call("structure")["nodes"]
request = {"prompt": "The capital of France is", "elements": 256, "steps": 2, "size": 64}
call("run", request)
state = wait()
trace = call("traces/" + state["traces"][-1]["id"])
assert len(trace["events"]) > 0
if not args.diffusion:
    assert trace["tokens"] and trace["output"]["top_tokens"]
    call("run", {**request, "attention": True, "layer": 0, "head": 0})
    state = wait()
    attention = call("traces/" + state["traces"][-1]["id"])
    assert attention["kind"] == "attention_capture" and attention["values"]
    call("run", request)
    state = wait()
else:
    assert trace["preview"]["mime"] == "image/png"
if not args.keep_loaded:
    call("unload", {})
    assert wait()["loaded"] is None
print(json.dumps({"model": args.model_name, "events": len(trace["events"]), "status": "passed", "job": state["job"]}))
