#!/usr/bin/env python3
"""
Downloads and extracts standard evaluation benchmark datasets:
1. BSD68 (68 grayscale/natural images from FFDNet GitHub repository)
2. Urban100 (100 high-resolution urban scene images from Hugging Face repository)
3. SIDD Validation / Benchmark Blocks (sRGB & RAW .mat files + paired scene directory check)
"""

import json
import os
import sys
import tarfile
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

BSD68_API_URL = "https://api.github.com/repos/cszn/FFDNet/contents/testsets/BSD68"
URBAN100_HR_URL = "https://huggingface.co/datasets/eugenesiow/Urban100/resolve/main/data/Urban100_HR.tar.gz"
SIDD_BENCH_SRGB_URL = "ftp://sidd_user:sidd_2018@130.63.97.225/SIDD_Benchmark_Data/BenchmarkNoisyBlocksSrgb.mat"
SIDD_BENCH_RAW_URL = "ftp://sidd_user:sidd_2018@130.63.97.225/SIDD_Benchmark_Data/BenchmarkNoisyBlocksRaw.mat"


def download_url(url, dest_path):
    """Downloads a single URL to a file path."""
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=30) as resp, open(dest_path, "wb") as f:
        f.write(resp.read())


def download_bsd68(target_dir="./data/BSD68"):
    """Downloads all 68 images from the BSD68 dataset."""
    os.makedirs(target_dir, exist_ok=True)
    print("Fetching BSD68 file list from GitHub...")
    
    headers = {"User-Agent": "Mozilla/5.0"}
    req = urllib.request.Request(BSD68_API_URL, headers=headers)
    with urllib.request.urlopen(req, timeout=15) as resp:
        files = json.loads(resp.read().decode("utf-8"))

    png_files = [f for f in files if f["name"].lower().endswith(".png")]
    print(f"Found {len(png_files)} images for BSD68. Downloading to {target_dir}...")

    def fetch_item(item):
        name = item["name"]
        durl = item["download_url"]
        out_path = os.path.join(target_dir, name)
        if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
            return name, "skipped"
        download_url(durl, out_path)
        return name, "downloaded"

    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(fetch_item, item) for item in png_files]
        for fut in as_completed(futures):
            name, status = fut.result()

    count = len([f for f in os.listdir(target_dir) if f.lower().endswith(".png")])
    print(f"[✓] BSD68 ready: {count} images in {target_dir}")


def download_urban100(target_dir="./data/Urban100"):
    """Downloads and extracts Urban100 High-Resolution images from Hugging Face."""
    os.makedirs(target_dir, exist_ok=True)
    tar_path = os.path.join("./data", "Urban100_HR.tar.gz")
    
    print(f"Downloading Urban100 (HR) from Hugging Face (~135 MB)...")
    download_url(URBAN100_HR_URL, tar_path)
    
    print(f"Extracting Urban100 to {target_dir}...")
    with tarfile.open(tar_path, "r:gz") as tar:
        # Extract files flattening directly or into subfolder
        members = tar.getmembers()
        for member in members:
            # Flatten or keep filename
            fname = os.path.basename(member.name)
            if fname and fname.lower().endswith(".png"):
                member.name = fname
                tar.extract(member, path=target_dir)

    # Clean up archive
    if os.path.exists(tar_path):
        os.remove(tar_path)

    count = len([f for f in os.listdir(target_dir) if f.lower().endswith(".png")])
    print(f"[✓] Urban100 ready: {count} images in {target_dir}")


def verify_or_download_sidd():
    """Verifies or downloads SIDD Benchmark .mat files and checks paired scene layout."""
    srgb_dir = "./data/SIDD_val_srgb"
    raw_dir = "./data/SIDD_val_raw"
    srgb_mat = os.path.join(srgb_dir, "BenchmarkNoisyBlocksSrgb.mat")
    raw_mat = os.path.join(raw_dir, "BenchmarkNoisyBlocksRaw.mat")

    os.makedirs(srgb_dir, exist_ok=True)
    os.makedirs(raw_dir, exist_ok=True)

    for label, path, url in [
        ("SIDD Benchmark sRGB (.mat)", srgb_mat, SIDD_BENCH_SRGB_URL),
        ("SIDD Benchmark RAW (.mat)", raw_mat, SIDD_BENCH_RAW_URL),
    ]:
        if os.path.exists(path) and os.path.getsize(path) > 0:
            print(f"[✓] {label} ready: {path}")
        else:
            print(f"Downloading {label} to {path}...")
            try:
                urllib.request.urlretrieve(url, path)
                print(f"[✓] {label} downloaded: {path}")
            except Exception as e:
                print(
                    f"[!] Could not auto-download {label} from FTP mirror ({e}).\n"
                    f"    Please place {os.path.basename(path)} in {os.path.dirname(path)}/ "
                    f"from https://www.eecs.yorku.ca/~kamel/sidd/"
                )

    paired_dir = "./data/SIDD/train/train"
    if os.path.exists(paired_dir):
        scenes = [d for d in os.listdir(paired_dir) if os.path.isdir(os.path.join(paired_dir, d))]
        print(f"[✓] SIDD Paired Scenes ready: {len(scenes)} scenes in {paired_dir}")
    else:
        print(
            f"[!] Note: For full paired SIDD training/validation, extract SIDD-Small / SIDD-Medium scenes into:\n"
            f"    {paired_dir}/<scene_folder>/*_NOISY_*.PNG & *_GT_*.PNG"
        )


def main():
    print("==================================================")
    print("Procuring BSD68, Urban100, and SIDD Benchmark Data")
    print("==================================================")
    
    download_bsd68()
    print()
    download_urban100()
    print()
    verify_or_download_sidd()
    
    print("\n==================================================")
    print("All benchmark datasets successfully downloaded and verified!")
    print("==================================================")


if __name__ == "__main__":
    main()
