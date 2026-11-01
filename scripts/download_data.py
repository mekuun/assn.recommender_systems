"""Download original Amazon Reviews 2018 Video Games files (standard library only)."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.request import urlopen
import gzip
import hashlib
import json
import shutil

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "data" / "raw"
BASE = "https://mcauleylab.ucsd.edu/public_datasets/data/amazon_v2/"
FILES = {
    "Video_Games_5.json.gz": BASE + "categoryFilesSmall/Video_Games_5.json.gz",
    "meta_Video_Games.json.gz": BASE + "metaFiles2/meta_Video_Games.json.gz",
}


def download(entry):
    name, url = entry
    target = DEST / name
    if not target.exists():
        partial = target.with_suffix(target.suffix + ".part")
        print(f"Downloading {name}", flush=True)
        with urlopen(url, timeout=120) as response, partial.open("wb") as output:
            shutil.copyfileobj(response, output)
        # Read the entire gzip stream to check CRC before accepting the file.
        with gzip.open(partial, "rb") as stream:
            while stream.read(1024 * 1024):
                pass
        partial.replace(target)
    digest = hashlib.sha256(target.read_bytes()).hexdigest()
    print(f"Ready: {name} ({target.stat().st_size:,} bytes)", flush=True)
    return {"file": name, "url": url, "bytes": target.stat().st_size, "sha256": digest}


if __name__ == "__main__":
    DEST.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=2) as pool:
        files = list(pool.map(download, FILES.items()))
    (DEST.parent / "download_manifest.json").write_text(
        json.dumps({"source": "https://cseweb.ucsd.edu/~jmcauley/datasets/amazon_v2/",
                    "files": files}, indent=2) + "\n", encoding="utf-8"
    )
