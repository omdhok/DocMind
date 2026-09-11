"""
Downloads the quantized local LLM (GGUF format) used by DocMind's
Private Offline Mode.

This script needs internet access (to Hugging Face) the FIRST time you
run it. After the file is downloaded, DocMind's core app never needs
internet again for Private Offline Mode.

Usage:
    python scripts/download_models.py

This downloads a ~2GB file. If you prefer, you can instead manually
download any instruction-tuned Llama-3.2-3B GGUF file (Q4_K_M
quantization recommended for a good size/quality/speed balance) from
Hugging Face and place it at:
    models/llama-3.2-3b-instruct.Q4_K_M.gguf
"""

import sys
from pathlib import Path

MODEL_URL = (
    "https://huggingface.co/bartowski/Llama-3.2-3B-Instruct-GGUF/resolve/main/"
    "Llama-3.2-3B-Instruct-Q4_K_M.gguf"
)
OUTPUT_PATH = Path("models/llama-3.2-3b-instruct.Q4_K_M.gguf")


def main():
    if OUTPUT_PATH.exists():
        print(f"Model already exists at {OUTPUT_PATH} -- skipping download.")
        return

    try:
        import requests
    except ImportError:
        print("Missing 'requests'. Run: pip install requests")
        sys.exit(1)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    print(f"Downloading local LLM from:\n  {MODEL_URL}")
    print("This is a multi-GB file -- it may take a while depending on your connection.\n")

    try:
        with requests.get(MODEL_URL, stream=True, timeout=30) as response:
            response.raise_for_status()
            total = int(response.headers.get("content-length", 0))
            downloaded = 0
            tmp_path = OUTPUT_PATH.with_suffix(".tmp")
            with open(tmp_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8 * 1024 * 1024):
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total:
                        pct = downloaded / total * 100
                        print(f"\r  {downloaded / 1e6:.0f} MB / {total / 1e6:.0f} MB ({pct:.1f}%)", end="")
            tmp_path.rename(OUTPUT_PATH)
    except Exception as exc:
        print(f"\nDownload failed: {exc}")
        print(
            "You can download the file manually from the URL above and place it at:\n"
            f"  {OUTPUT_PATH}"
        )
        sys.exit(1)

    print(f"\n\nDone. Model saved to: {OUTPUT_PATH}")
    print(
        "\nNext: run 'python scripts/export_embeddings.py' if you haven't "
        "already, then 'streamlit run app.py'."
    )


if __name__ == "__main__":
    main()
