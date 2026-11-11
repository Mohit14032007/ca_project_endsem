#!/usr/bin/env python3
import os
import sys
import zipfile
import hashlib
import ast

def check_no_ternary():
    with open(__file__, "r", encoding="utf-8") as f:
        src = f.read()
    tree = ast.parse(src, filename=__file__)
    for node in ast.walk(tree):
        if isinstance(node, ast.IfExp):
            raise AssertionError("Ternary operator found!")
    print("[PRE-FLIGHT] Zero ternary operators verified.")

def main():
    check_no_ternary()
    zip_path = "/home/mohman/ca_project/final_submission/08_DOCUMENTATION_AND_REPRODUCIBILITY/zip/FINAL_ENDSEM_PROJECT.zip"
    checksum_file = "/home/mohman/ca_project/final_submission/08_DOCUMENTATION_AND_REPRODUCIBILITY/zip/ARCHIVE_CHECKSUM.txt"

    if not os.path.exists(zip_path):
        print(f"[FAIL] Missing {zip_path}")
        sys.exit(1)

    # 1. Verify SHA-256
    h = hashlib.sha256()
    with open(zip_path, "rb") as f:
        while True:
            chunk = f.read(65536)
            if not chunk:
                break
            h.update(chunk)
    actual_sha = h.hexdigest()

    expected_sha = None
    with open(checksum_file, "r") as f:
        for line in f:
            if line.startswith("SHA-256:"):
                expected_sha = line.split(":", 1)[1].strip()

    print(f"Calculated SHA-256: {actual_sha}")
    print(f"Recorded SHA-256:   {expected_sha}")

    if actual_sha != expected_sha:
        print("[FAIL] Checksum mismatch!")
        sys.exit(1)
    print("[PASS] SHA-256 checksum matches perfectly!")

    # 2. Inspect ZIP contents
    with zipfile.ZipFile(zip_path, "r") as z:
        namelist = z.namelist()
        print(f"[INFO] Archive contains {len(namelist)} files.")

        forbidden = [".git/", ".o", ".so", ".a", "__pycache__"]
        bad_files = []
        for name in namelist:
            for forb in forbidden:
                if forb in name:
                    bad_files.append(name)
        if bad_files:
            print(f"[FAIL] Found {len(bad_files)} forbidden build artifacts in ZIP!")
            sys.exit(1)
        print("[PASS] Clean archive verified: 0 forbidden binaries or build artifacts.")

if __name__ == "__main__":
    main()

