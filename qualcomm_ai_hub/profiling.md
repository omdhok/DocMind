# Profiling on Real Snapdragon Hardware (Without Owning It)

You do not need to personally own a Snapdragon X Elite/X2 Elite laptop
to get real performance numbers from one. Qualcomm AI Hub provides a
hosted device cloud: you submit a compiled model as a "job," and AI Hub
runs it on an actual physical Snapdragon device in their lab, returning
real latency and memory measurements.

## Steps

1. Create a free account at https://aihub.qualcomm.com and generate an
   API token from your account settings.
2. Install the client:
   ```
   pip install qai-hub
   qai-hub configure --api_token YOUR_TOKEN
   ```
3. Submit your exported ONNX model (e.g.
   `models/minilm-onnx/model.onnx`) as a compile job targeting a
   Snapdragon X Elite device profile, then submit the compiled model as
   a **profiling job**. AI Hub's job dashboard returns:
   - On-device inference latency
   - Peak memory usage
   - Which compute unit (NPU/GPU/CPU) each operator actually ran on
4. Save the job report (screenshot or exported JSON) into this folder,
   e.g. `qualcomm_ai_hub/profiling_reports/minilm_snapdragon_x_elite.json`,
   and reference it in your README/submission with the job URL as the
   citation for any NPU-latency claim.

## What this proves vs. what it doesn't

- **Proves:** the model, as exported and quantized, runs correctly and
  at a specific measured latency on real Snapdragon X Elite silicon.
- **Does not prove:** that DocMind's live, interactive Streamlit demo
  (as judges see it) is itself running on that hardware — that requires
  either physical/remote access to a Snapdragon Windows PC for the demo
  session, or a recorded video of the app running on one.

Be explicit in your submission about which of these two you have. Both
are legitimate; only the first is available to every participant
regardless of hardware access, and both are more credible than an
unverified performance claim.

## If you obtain temporary access to real Snapdragon hardware

Run `python scripts/benchmark.py <sample.pdf>` directly on that machine.
It measures the exact same pipeline stages (document processing,
embedding, retrieval, generation, execution provider) and writes real
numbers to `scripts/benchmark_results.json` — no manual number entry,
no estimation.
