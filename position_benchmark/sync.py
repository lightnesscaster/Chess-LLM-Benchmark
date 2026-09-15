"""Sync production position panels without requiring local Firebase credentials."""

import base64
import copy
import json
import os
from pathlib import Path
import shlex
import subprocess
import time
import urllib.request
import zlib

from position_benchmark.layout import CORE_POSITIONS_PATH, GAME_LIKE_POSITIONS_PATH
from position_benchmark.predictions import benchmark_result_readiness, stability_probe_readiness

WEB_URL = "https://chessbenchllm.onrender.com"
SERVICE_ID = "srv-d4mlb0m3jp1c73a172v0"


def merge_panel_record(existing: dict, panel: str, record: dict) -> dict:
    """Replace only this panel, preserving other independently acquired panels."""
    if panel not in {"core", "game_like", "stability"}:
        raise ValueError("Only production panels may be synced")
    if panel == "core":
        merged = copy.deepcopy(record)
        if "supplements" in existing:
            merged["supplements"] = copy.deepcopy(existing["supplements"])
    else:
        merged = copy.deepcopy(existing)
        merged.setdefault("supplements", {})[panel] = copy.deepcopy(record)
    return merged


def validate_panel(player_id: str, panel: str, record: dict) -> None:
    if record.get("summary", {}).get("player_id") != player_id:
        raise ValueError("Panel player ID mismatch")
    if panel == "stability":
        ready = stability_probe_readiness(record)
    elif panel in {"core", "game_like"}:
        path = CORE_POSITIONS_PATH if panel == "core" else GAME_LIKE_POSITIONS_PATH
        positions = json.loads(path.read_text())["positions"]
        ready = benchmark_result_readiness(record, positions, min_equal_positions=len(positions))
    else:
        raise ValueError("Only production panels may be synced")
    if not ready.is_ready:
        raise ValueError(f"Panel not ready: {ready.reason}")


def write_panel(db, payload: dict) -> dict:
    """Atomic per-panel update. Never writes ratings or full-game results."""
    from firebase_admin import firestore
    validate_panel(payload["player_id"], payload["panel"], payload["record"])
    ref = db.collection("benchmark_results").document(payload["player_id"])

    @firestore.transactional
    def save(tx):
        old = ref.get(transaction=tx)
        merged = merge_panel_record(old.to_dict() or {}, payload["panel"], payload["record"])
        tx.set(ref, merged)
    save(db.transaction())
    # Read back before reporting success; retries replace the same panel.
    saved = ref.get().to_dict()
    actual = saved if payload["panel"] == "core" else saved.get("supplements", {}).get(payload["panel"])
    if payload["panel"] == "core":
        actual = {k: v for k, v in actual.items() if k != "supplements"}
    if actual != payload["record"]:
        raise RuntimeError("Position sync readback mismatch")
    return {"player_id": payload["player_id"], "panel": payload["panel"], "synced": True}


def sync_via_render(payload: dict) -> dict:
    """Run the same writer with server-side credentials and verify completion."""
    source = base64.b64encode(Path(__file__).read_bytes()).decode()
    encoded = base64.b64encode(zlib.compress(json.dumps(payload).encode())).decode()
    code = (
        "import base64,json,zlib,os,urllib.request; ns={'__name__':'position_sync_worker'}; "
        f"exec(compile(base64.b64decode({source!r}),'<position-sync>','exec'),ns); "
        "from firebase_client import get_firestore_client; "
        f"receipt=ns['write_panel'](get_firestore_client(),json.loads(zlib.decompress(base64.b64decode({encoded!r})))); "
        f"req=urllib.request.Request({WEB_URL + '/api/invalidate-cache'!r},data=b'',headers={{'X-Cache-Token':os.environ['CACHE_INVALIDATE_TOKEN']}},method='POST'); "
        "urllib.request.urlopen(req,timeout=30).close(); "
        "print('POSITION_SYNC_COMPLETE '+json.dumps(receipt),flush=True)"
    )
    created = subprocess.run(["render", "jobs", "create", SERVICE_ID, "--start-command",
                              "python -c " + shlex.quote(code), "--output", "json", "--confirm"],
                             capture_output=True, text=True, timeout=90, check=True)
    job_id = json.loads(created.stdout)["id"]
    print(f"Position sync job: {job_id}", flush=True)
    deadline = time.monotonic() + 600
    while time.monotonic() < deadline:
        output = subprocess.run(["render", "logs", "--resources", job_id, "--limit", "8",
                                 "--direction", "backward", "--output", "json", "--confirm"],
                                capture_output=True, text=True, timeout=90, check=True).stdout
        decoder = json.JSONDecoder()
        while output.strip():
            entry, end = decoder.raw_decode(output.lstrip())
            output = output.lstrip()[end:]
            message = entry.get("message", "")
            if message.startswith("POSITION_SYNC_COMPLETE "):
                receipt = json.loads(message.removeprefix("POSITION_SYNC_COMPLETE "))
                if receipt.get("player_id") != payload["player_id"] or receipt.get("panel") != payload["panel"]:
                    raise ValueError("Position sync receipt mismatch")
                return receipt
        time.sleep(10)
    raise TimeoutError(f"Sync not confirmed; inspect Render job {job_id}. Retry sync, not model calls.")


def sync_panel(player_id: str, panel: str, record: dict) -> dict:
    """Use direct Firebase when configured, otherwise authenticated Render CLI."""
    validate_panel(player_id, panel, record)
    payload = {"player_id": player_id, "panel": panel, "record": record}
    from firebase_client import get_firestore_client
    try:
        db = get_firestore_client()
    except FileNotFoundError:
        return sync_via_render(payload)
    return write_panel(db, payload)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Sync saved position panels without repeating model calls")
    parser.add_argument("--player", required=True)
    parser.add_argument("--panels", nargs="+", choices=["core", "game_like", "stability"], default=["core", "game_like", "stability"])
    args = parser.parse_args()
    for panel in args.panels:
        path = Path(__file__).parent / "results" / (panel + ".json")
        print(json.dumps(sync_panel(args.player, panel, json.loads(path.read_text())[args.player])), flush=True)
