#!/usr/bin/env python3
"""
Darmstadt Noise Dataset (DND) Parallel Downloader & Extractor
Logs in to noise.visinf.tu-darmstadt.de, fetches multipart archive files in parallel,
and merges/extracts them into ./data/DND.
"""

import argparse
import os
import re
import sys
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.parse import unquote, urljoin

try:
    import requests
    from requests.adapters import HTTPAdapter
    from urllib3.util.retry import Retry
except ImportError:
    print("Error: 'requests' is required. Install via: pip install requests")
    sys.exit(1)

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

BASE_URL = "https://noise.visinf.tu-darmstadt.de"
LOGIN_URL = "https://noise.visinf.tu-darmstadt.de/login/"
DOWNLOADS_URL = "https://noise.visinf.tu-darmstadt.de/downloads/"


def get_authenticated_session(username, password, retries=5):
    """Authenticates with DND and returns a session with active cookies."""
    session = requests.Session()
    retry_strategy = Retry(
        total=retries,
        backoff_factor=0.5,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["HEAD", "GET", "POST"],
    )
    adapter = HTTPAdapter(max_retries=retry_strategy, pool_connections=20, pool_maxsize=20)
    session.mount("https://", adapter)
    session.mount("http://", adapter)

    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Referer": LOGIN_URL,
    }
    session.headers.update(headers)

    print(f"Logging in to DND as '{username}'...")
    login_payload = {
        "username": username,
        "password": password,
        "redirect_to": DOWNLOADS_URL,
        "submit": "Login",
    }
    
    resp = session.post(LOGIN_URL, data=login_payload, allow_redirects=True, timeout=20)
    resp.raise_for_status()

    # Verify if login succeeded
    if "Please Login" in resp.text and "Invalid" in resp.text:
        raise ValueError("Login failed: Invalid username or password.")
    
    # Check if downloads page is accessible
    dl_resp = session.get(DOWNLOADS_URL, timeout=20)
    if "Please Login" in dl_resp.text:
        raise ValueError("Login failed: Session did not persist.")

    print("Authentication successful!")
    return session, dl_resp.text


def extract_download_links(html_content, skip_parts=()):
    """Parses all filehandler links from the downloads page, skipping specified parts."""
    links = re.findall(r'href=["\'](filehandler\.php\?fileid=[^"\']+)["\']', html_content)
    full_urls = [urljoin(BASE_URL, l) for l in set(links)]
    
    filtered_urls = []
    for u in full_urls:
        match = re.search(r'fileid=1_(\d+)', u)
        if match:
            part_num = int(match.group(1))
            if part_num in skip_parts:
                continue
            filtered_urls.append((part_num, u))
        elif "fileid=2_" in u:
            # Submission kit
            filtered_urls.append((999, u))

    filtered_urls.sort(key=lambda x: x[0])
    return [u for _, u in filtered_urls]


def download_chunk(url, session, output_dir, max_retries=5, chunk_size=1024 * 1024):
    """Downloads an individual filehandler part with resume and retry capability."""
    for attempt in range(1, max_retries + 1):
        try:
            # 1. Fetch headers to resolve filename and total size
            with session.get(url, stream=True, timeout=20) as r:
                r.raise_for_status()
                cd = r.headers.get("Content-Disposition", "")
                fname_match = re.search(r'filename\*?=(?:UTF-8\'\')?["\']?([^";\r\n]+)["\']?', cd)
                
                if fname_match:
                    filename = unquote(fname_match.group(1).strip())
                else:
                    fileid = re.search(r'fileid=([^&]+)', url)
                    filename = f"dnd_file_{fileid.group(1)}.bin" if fileid else "dnd_part.bin"

                total_size = int(r.headers.get("Content-Length", 0))

            dest_path = os.path.join(output_dir, filename)
            part_path = dest_path + ".part"

            # Check if full file already exists
            if os.path.exists(dest_path):
                existing_size = os.path.getsize(dest_path)
                if total_size > 0 and existing_size == total_size:
                    return {"status": "skipped", "filename": filename, "size": existing_size, "path": dest_path}

            # Resume support
            start_byte = 0
            headers = {}
            if os.path.exists(part_path):
                start_byte = os.path.getsize(part_path)
                if total_size > 0 and start_byte >= total_size:
                    start_byte = 0
                elif start_byte > 0:
                    headers["Range"] = f"bytes={start_byte}-"

            mode = "ab" if start_byte > 0 else "wb"
            with session.get(url, headers=headers, stream=True, timeout=60) as r:
                if start_byte > 0 and r.status_code not in (206, 200):
                    start_byte = 0
                    mode = "wb"
                    r = session.get(url, stream=True, timeout=60)
                
                r.raise_for_status()

                if HAS_TQDM:
                    pbar = tqdm(
                        total=total_size,
                        initial=start_byte,
                        unit="B",
                        unit_scale=True,
                        unit_divisor=1024,
                        desc=f"{filename[:28]:<28}",
                        leave=False
                    )
                else:
                    pbar = None

                with open(part_path, mode) as f:
                    for chunk in r.iter_content(chunk_size=chunk_size):
                        if chunk:
                            f.write(chunk)
                            if pbar:
                                pbar.update(len(chunk))

                if pbar:
                    pbar.close()

            if os.path.exists(part_path):
                os.rename(part_path, dest_path)

            return {
                "status": "success",
                "filename": filename,
                "size": os.path.getsize(dest_path),
                "path": dest_path
            }

        except Exception as e:
            if attempt == max_retries:
                return {"status": "failed", "url": url, "error": str(e)}
            time.sleep(2 ** attempt * 0.5)


def merge_and_extract(output_dir, extract_dir):
    """Merges multipart split files into a single zip and extracts."""
    os.makedirs(extract_dir, exist_ok=True)
    
    # Check for split files (e.g. dnd_2017_split.zip.001 ...)
    split_files = sorted([
        os.path.join(output_dir, f) for f in os.listdir(output_dir)
        if re.search(r'dnd_2017_split\.zip\.\d+$', f)
    ])

    if not split_files:
        print("No split files found to merge.")
        return

    # Check if parts 1 and 2 are present
    has_part1 = any("dnd_2017_split.zip.001" in f for f in split_files)
    has_part2 = any("dnd_2017_split.zip.002" in f for f in split_files)

    if not (has_part1 and has_part2):
        print("\n[!] Parts 001 and/or 002 are not yet in the parts directory.")
        print(f"Please copy your downloaded `dnd_2017_split.zip.001` and `dnd_2017_split.zip.002` into:")
        print(f"  {os.path.abspath(output_dir)}/")
        print("Then run: python3 -c 'import download_dnd; download_dnd.merge_and_extract(\"./data/DND_parts\", \"./data/DND\")'")
        return

    combined_zip_path = os.path.join(output_dir, "dnd_2017.zip")
    print(f"\nMerging {len(split_files)} parts into {combined_zip_path}...")

    with open(combined_zip_path, "wb") as outfile:
        for sfile in split_files:
            with open(sfile, "rb") as infile:
                while True:
                    buf = infile.read(10 * 1024 * 1024)
                    if not buf:
                        break
                    outfile.write(buf)

    print(f"Extracting {combined_zip_path} to {extract_dir}...")
    try:
        with zipfile.ZipFile(combined_zip_path, "r") as zip_ref:
            zip_ref.extractall(extract_dir)
        print("[✓] Extraction complete!")
    except Exception as e:
        print(f"Extraction note: {e}. You can run: unzip {combined_zip_path} -d {extract_dir}")


def main():
    parser = argparse.ArgumentParser(description="Multi-threaded batch downloader for the DND dataset (Parts 1 to 33).")
    parser.add_argument("--username", default=os.environ.get("DND_USERNAME"), help="DND portal username (or set DND_USERNAME env var)")
    parser.add_argument("--password", default=os.environ.get("DND_PASSWORD"), help="DND portal password (or set DND_PASSWORD env var)")
    parser.add_argument("--output-dir", default="./data/DND_parts", help="Directory for split part files")
    parser.add_argument("--extract-dir", default="./data/DND", help="Directory for extracted dataset")
    parser.add_argument("--threads", type=int, default=6, help="Parallel download threads (default: 6)")
    parser.add_argument("--skip-parts", type=int, nargs="*", default=[], help="Part numbers to skip if already downloaded externally")
    parser.add_argument("--merge-only", action="store_true", help="Only merge and extract existing parts")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    if args.merge_only:
        merge_and_extract(args.output_dir, args.extract_dir)
        return

    if not args.username or not args.password:
        parser.error("DND requires portal credentials. Pass --username and --password (or set DND_USERNAME and DND_PASSWORD). Register at https://noise.visinf.tu-darmstadt.de/")

    # 1. Log in and get download page
    session, dl_html = get_authenticated_session(args.username, args.password)
    skip_tuple = tuple(args.skip_parts)
    split_urls = extract_download_links(dl_html, skip_parts=skip_tuple)

    if not split_urls:
        print("Fallback: Using direct file IDs 1_1 to 1_33...")
        split_urls = [
            f"{BASE_URL}/filehandler.php?fileid=1_{i}"
            for i in range(1, 34)
            if i not in skip_tuple
        ]
        split_urls.append(f"{BASE_URL}/filehandler.php?fileid=2_0")

    print(f"\nDownloading {len(split_urls)} DND archive files in parallel...")
    print(f"Output directory: {os.path.abspath(args.output_dir)}")
    print(f"Parallel streams: {args.threads}")
    print("--------------------------------------------------\n")

    start_time = time.time()
    results = []
    
    with ThreadPoolExecutor(max_workers=args.threads) as executor:
        future_to_url = {
            executor.submit(download_chunk, url, session, args.output_dir): url
            for url in split_urls
        }

        for future in as_completed(future_to_url):
            url = future_to_url[future]
            res = future.result()
            results.append(res)
            
            if res.get("status") == "success":
                size_mb = res.get("size", 0) / (1024 * 1024)
                print(f"[SUCCESS] {res.get('filename')} ({size_mb:.2f} MB)")
            elif res.get("status") == "skipped":
                print(f"[SKIPPED] {res.get('filename')} (Already exists)")
            else:
                print(f"[FAILED]  {url} -> {res.get('error')}")

    elapsed = (time.time() - start_time) / 60
    print(f"\nDownload of parts 3 to 33 finished in {elapsed:.2f} minutes.")

    # Merge and extract
    merge_and_extract(args.output_dir, args.extract_dir)


if __name__ == "__main__":
    main()
