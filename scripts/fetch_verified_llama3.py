"""Populate the pinned Llama 3 snapshot from ModelScope and verify every file."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
from pathlib import Path
import shutil
import time

import requests


URL = "https://www.modelscope.cn/api/v1/models/LLM-Research/Meta-Llama-3-8B-Instruct/repo"
MIRROR_REVISION = "6cec47ecb2d22cfcab15b1372d14b48feeef5eff"
CHUNK_BYTES = 64 * 1024**2
# Weight hashes match the Git LFS pointers at the experiment's Hugging Face revision.
FILES = {
    "config.json": (654, "61f3de03a16ca8046b05dc777bce72717022bc8522152eee61a69072272ef54b"),
    "generation_config.json": (187, "d1af74d8ba49428ab2faa65f7948165cc6bf52b740e57e840f8af772587c894c"),
    "model.safetensors.index.json": (23950, "146776fce3f6db1103aa6f249e65ee5544c5923ce6f971b092eee79aa6e5d37b"),
    "special_tokens_map.json": (73, "462d91939dbc37178aa5a3eae7068d1990ccc92e09f288cc71f42cdf139d69cc"),
    "tokenizer.json": (9085698, "e134af98b985517b4f068e3755ae90d4e9cd2d45d328325dc503f1c6b2d06cc7"),
    "tokenizer_config.json": (50977, "da0e3a7cce6e4d787e85eb1c24d548420e0d7fe2c7a214e192795c46e40d75bb"),
    "model-00001-of-00004.safetensors": (4976698672, "d8cf9c4d0dd972e1a2131bfe656235ee98221679711a3beef6d46dadf0f20b5c"),
    "model-00002-of-00004.safetensors": (4999802720, "8d4782b4a69ef03845159ce1a15e272aadaaf134dc138d68f616098e8531729c"),
    "model-00003-of-00004.safetensors": (4915916176, "3acdd690e65c24f42a24581b8467af98bd3ca357444580f8012aacd2bd607921"),
    "model-00004-of-00004.safetensors": (1168138808, "67e9ad31c8c32abf3a55ee7fc7217b3ecb35fd3c74d98a5bd233e0e4d6964f46"),
}


def verify(path, size, expected_sha):
    if not path.is_file() or path.stat().st_size != size:
        return False
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest() == expected_sha


def fetch_chunk(directory, name, size, start, stop):
    part = directory / f"{name}.{start:012d}"
    length = stop - start + 1
    if part.is_file() and part.stat().st_size == length:
        return part
    pending = part.with_suffix(part.suffix + ".partial")
    if pending.is_file() and pending.stat().st_size == length:
        pending.replace(part)
        return part
    params = {"Revision": MIRROR_REVISION, "FilePath": name}
    for attempt in range(8):
        position = start + (pending.stat().st_size if pending.exists() else 0)
        headers = {"Range": f"bytes={position}-{stop}"}
        try:
            with requests.get(URL, params=params, headers=headers, stream=True, timeout=(12, 40)) as response:
                if response.status_code == 206:
                    if response.headers.get("Content-Range") != f"bytes {position}-{stop}/{size}":
                        raise ValueError(f"Unexpected range for {name}")
                elif response.status_code != 200 or position != 0 or length != size:
                    raise ValueError(f"Unexpected HTTP {response.status_code} for {name}")
                with pending.open("ab") as stream:
                    for data in response.iter_content(1024 * 1024):
                        position += len(data)
                        if position > stop + 1:
                            raise ValueError(f"Oversized download for {name}")
                        stream.write(data)
        except (requests.RequestException, ValueError):
            if attempt == 7:
                raise
        if pending.is_file() and pending.stat().st_size == length:
            pending.replace(part)
            return part
        time.sleep(min(attempt + 1, 4))
    raise RuntimeError(f"Incomplete range for {name}: {start}-{stop}")


def populate(snapshot):
    snapshot.mkdir(parents=True, exist_ok=True)
    parts = snapshot / ".modelscope-parts"
    parts.mkdir(exist_ok=True)
    with ThreadPoolExecutor(max_workers=12) as pool:
        pending = {}
        for name, (size, expected_sha) in FILES.items():
            output = snapshot / name
            if verify(output, size, expected_sha):
                continue
            merged = parts / f"{name}.merged"
            merged.touch(exist_ok=True)
            completed = merged.stat().st_size
            if completed != size:
                completed = min(completed, size) // CHUNK_BYTES * CHUNK_BYTES
            with merged.open("r+b") as stream:
                stream.truncate(completed)
            pending[name] = [
                pool.submit(fetch_chunk, parts, name, size, start, min(start + CHUNK_BYTES, size) - 1)
                for start in range(completed, size, CHUNK_BYTES)
            ]
        for name, chunks in pending.items():
            size, expected_sha = FILES[name]
            merged = parts / f"{name}.merged"
            with merged.open("ab") as stream:
                for index, chunk in enumerate(chunks, 1):
                    part = chunk.result()
                    with part.open("rb") as source:
                        shutil.copyfileobj(source, stream, 1024 * 1024)
                    stream.flush()
                    part.unlink()
                    if index % 8 == 0:
                        print(f"{name}: {stream.tell():,}/{size:,} bytes", flush=True)
            if not verify(merged, size, expected_sha):
                merged.unlink()
                raise ValueError(f"SHA256 verification failed for {name}")
            merged.replace(snapshot / name)
            print(f"{name}: SHA256 OK", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("snapshot", type=Path)
    populate(parser.parse_args().snapshot)
