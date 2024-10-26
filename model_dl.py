#!/usr/bin/python3

import os
# import sys
from urllib import request


MODELS_URLS = {
    "llama32-1B": {
        "fp16": "https://huggingface.co/iangitonga/llama32/resolve/main/llama32-1B.fp16.bin",
    }
}


def _download_model(url, model_path):
    print("Downloading model ...")
    with request.urlopen(url) as source, open(model_path, "wb") as output:
        download_size = int(source.info().get("Content-Length"))
        download_size_mb = int(download_size / 1000_000)
        downloaded = 0
        while True:
            buffer = source.read(8192)
            if not buffer:
                break

            output.write(buffer)
            downloaded += len(buffer)
            progress_perc = int(downloaded / download_size * 100.0)
            downloaded_mb = int(downloaded / 1000_000)
            # trick to make it work with jupyter.
            print(f"\rDownload Progress [{downloaded_mb}/{download_size_mb}MB]: {progress_perc}%", end="")
    print("\n\n")


def download_model(dtype):
    model_name = "llama32-1B"
    model_path = os.path.join("models", f"{model_name}.{dtype}.bin")

    if os.path.exists(model_path):
        return
    os.makedirs("models", exist_ok=True)
    _download_model(MODELS_URLS[model_name][dtype], model_path)


# python model.py model dtype
# if len(sys.argv) != 2:
#     print(f"Args provided: {sys.argv}")
#     print("DTYPE is one of (fp16)")
#     exit(-1)


try:
    # download_model(sys.argv[1])
    download_model("fp16")
except:
    exit(-2)
