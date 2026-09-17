"""
Submit DocMind's exported MiniLM ONNX embedding model to Qualcomm AI Hub
for a real compile + profile job on physical Snapdragon silicon.

This closes the one unverified claim in the README's Snapdragon Optimization
evidence table: it turns the "Qualcomm QNN / Hexagon NPU execution" row from
"Optional, verified only if tested on real hardware or AI Hub device cloud"
into a row backed by an actual job URL and measured numbers.

What this DOES verify: the embedding stage (ONNX Runtime + QNN Execution
Provider) runs, and at what latency, on real Snapdragon X Elite/X2 silicon.
What this does NOT verify: the local LLM (llama.cpp/GGUF) stage, which AI
Hub does not profile -- be explicit about that distinction in the README
and deck (see qualcomm_ai_hub/profiling.md, "What this proves vs. what it
doesn't").

Usage:
    pip install qai-hub
    qai-hub configure --api_token YOUR_TOKEN     # from aihub.qualcomm.com
    python scripts/export_embeddings.py           # if you haven't already
    python scripts/aihub_profile.py

Writes a JSON report to qualcomm_ai_hub/profiling_reports/ and prints the
job URL + key numbers to paste directly into the README evidence table.
"""

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from config import ONNX_EMBEDDING_MODEL_DIR

ONNX_MODEL_PATH = Path(ONNX_EMBEDDING_MODEL_DIR) / "model.onnx"
REPORT_DIR = Path("qualcomm_ai_hub/profiling_reports")

# A Windows-on-Snapdragon compute device, matching the HP Omnibook class
# named in the challenge brief. Run `hub.get_devices()` to see the current
# full list if this exact name has changed by the time you run this.
TARGET_DEVICE_NAME = "Snapdragon X Elite CRD"

# Fixed shape required for a hub compile job (the app itself uses dynamic
# axes at runtime; AI Hub profiling needs one concrete shape to compile
# against). 32 tokens is a representative short-question length; bump this
# if you want a number closer to typical DocMind query lengths.
SEQUENCE_LENGTH = 32


def main() -> int:
    if not ONNX_MODEL_PATH.exists():
        print(
            f"ERROR: {ONNX_MODEL_PATH} not found.\n"
            "Run `python scripts/export_embeddings.py` first."
        )
        return 1

    try:
        import qai_hub as hub
    except ImportError:
        print("ERROR: qai-hub is not installed. Run: pip install qai-hub")
        return 1

    print(f"Compiling {ONNX_MODEL_PATH} for {TARGET_DEVICE_NAME} ...")
    device = hub.Device(TARGET_DEVICE_NAME)

    compile_job = hub.submit_compile_job(
        model=str(ONNX_MODEL_PATH),
        device=device,
        input_specs=dict(
            input_ids=((1, SEQUENCE_LENGTH), "int64"),
            attention_mask=((1, SEQUENCE_LENGTH), "int64"),
        ),
        options="--target_runtime onnx",
    )
    print(f"Compile job: {compile_job.url}")
    compile_job.wait()

    print("Submitting profile job (this runs on a real physical device in AI Hub's cloud) ...")
    profile_job = hub.submit_profile_job(
        model=compile_job.get_target_model(),
        device=device,
    )
    print(f"Profile job: {profile_job.url}")
    profile_job.wait()

    profile_data = profile_job.download_profile()

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    report_path = REPORT_DIR / "minilm_snapdragon_x_elite.json"
    with open(report_path, "w") as f:
        json.dump(
            {
                "generated_at_utc": datetime.now(timezone.utc).isoformat(),
                "device": TARGET_DEVICE_NAME,
                "sequence_length": SEQUENCE_LENGTH,
                "compile_job_url": compile_job.url,
                "profile_job_url": profile_job.url,
                "profile": profile_data,
            },
            f,
            indent=2,
            default=str,
        )

    print(f"\nSaved report -> {report_path}")
    print(
        "\nPaste this into the README's Snapdragon Optimization table and the "
        "deck:\n"
        f"  Device:      {TARGET_DEVICE_NAME}\n"
        f"  Profile job: {profile_job.url}\n"
        "  Check the report for 'execution_summary' / per-op compute unit to "
        "confirm QNN (NPU) vs CPU fallback, and the total inference latency.\n"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
