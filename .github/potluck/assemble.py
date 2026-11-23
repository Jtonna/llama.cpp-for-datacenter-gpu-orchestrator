#!/usr/bin/env python3
"""Assemble the Potluck llama release from the per-platform build outputs.

    assemble.py --dist DIR --out DIR --tag TAG --commit SHA --upstream-base SHA
                --repository OWNER/NAME --source-date-epoch N [--run-url URL]

DIR/<platform>/ holds llama-server, rpc-server and build-info.json (build.sh).
Writes to --out:
  potluck-llama-<tag>-<platform>.tar.gz   one archive per platform, holding
                                          <platform>/llama-server, rpc-server,
                                          SHA256SUMS and LICENSE-llama.cpp
  potluck-llama-manifest.json             the manifest Potluck reads
  SHA256SUMS                              sha256 of every other asset

Archives are deterministic: sorted entries, fixed mtime, uid/gid 0, no gzip
timestamp. Standard library only.
"""
import argparse
import gzip
import hashlib
import io
import json
import os
import re
import sys
import tarfile

PROGRAMS = ("llama-server", "rpc-server")
PLATFORMS = {
    "linux-amd64-vulkan": {"os": "linux", "arch": "amd64", "backend": "vulkan"},
    "linux-amd64-cuda": {"os": "linux", "arch": "amd64", "backend": "cuda"},
}
MANIFEST = "potluck-llama-manifest.json"
SCHEMA = "potluck-llama-release/1"


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def rpc_proto(src):
    text = open(os.path.join(src, "ggml", "include", "ggml-rpc.h"), encoding="utf-8").read()
    parts = []
    for name in ("MAJOR", "MINOR", "PATCH"):
        m = re.search(r"#define\s+RPC_PROTO_%s_VERSION\s+(\d+)" % name, text)
        if not m:
            sys.exit("ggml-rpc.h: RPC_PROTO_%s_VERSION not found" % name)
        parts.append(m.group(1))
    return ".".join(parts)


def parse_version(out):
    """Build number and short commit from `llama-server --version`.

    This fork prints "version: 1.0.0-dev (build 12539, commit cc81d31a0)";
    older llama.cpp printed "version: 12539 (cc81d31a0)".
    """
    m = re.search(r"\(build\s+(\d+),\s*commit\s+([0-9a-f]{7,40})\)", out)
    if not m:
        m = re.search(r"version:\s*(\d+)\s*\(([0-9a-f]{7,40})\)", out)
    if not m:
        return None, ""
    return int(m.group(1)), m.group(2)


def add_bytes(tar, name, data, mode, mtime):
    info = tarfile.TarInfo(name)
    info.size = len(data)
    info.mode = mode
    info.mtime = mtime
    info.uid = info.gid = 0
    info.uname = info.gname = ""
    tar.addfile(info, io.BytesIO(data))


def add_dir(tar, name, mtime):
    info = tarfile.TarInfo(name)
    info.type = tarfile.DIRTYPE
    info.mode = 0o755
    info.mtime = mtime
    info.uid = info.gid = 0
    info.uname = info.gname = ""
    tar.addfile(info)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dist", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--src", default=".")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--commit", required=True)
    ap.add_argument("--upstream-base", required=True)
    ap.add_argument("--upstream-repository", default="ggml-org/llama.cpp")
    ap.add_argument("--repository", required=True)
    ap.add_argument("--source-date-epoch", type=int, required=True)
    ap.add_argument("--run-url", default="")
    a = ap.parse_args()

    if not re.fullmatch(r"[0-9a-f]{40}", a.commit):
        sys.exit("--commit must be a full 40-character sha")
    if not re.fullmatch(r"[0-9a-f]{40}", a.upstream_base):
        sys.exit("--upstream-base must be a full 40-character sha")

    os.makedirs(a.out, exist_ok=True)
    licence = open(os.path.join(a.src, "LICENSE"), "rb").read()
    mtime = a.source_date_epoch

    platforms = {}
    build_numbers = set()
    for platform, spec in PLATFORMS.items():
        pdir = os.path.join(a.dist, platform)
        info = json.load(open(os.path.join(pdir, "build-info.json"), encoding="utf-8"))
        if info["platform"] != platform:
            sys.exit("%s: build-info.json names %s" % (platform, info["platform"]))
        number, short = parse_version(info["version_output"])
        if number is None or len(short) < 7 or not a.commit.startswith(short):
            sys.exit("%s: --version (%r) does not report commit %s" % (platform, info["version_output"], a.commit))
        build_numbers.add(number)

        files = {}
        sums = []
        for prog in PROGRAMS:
            p = os.path.join(pdir, prog)
            digest = sha256_file(p)
            files[prog] = {"sha256": digest, "bytes": os.path.getsize(p)}
            sums.append("%s  %s\n" % (digest, prog))
        sums_bytes = "".join(sums).encode()  # PROGRAMS is sorted: llama-server, rpc-server

        archive = "potluck-llama-%s-%s.tar.gz" % (a.tag, platform)
        raw = io.BytesIO()
        with tarfile.open(fileobj=raw, mode="w", format=tarfile.GNU_FORMAT) as tar:
            add_bytes(tar, "LICENSE-llama.cpp", licence, 0o644, mtime)
            add_dir(tar, platform, mtime)
            add_bytes(tar, platform + "/SHA256SUMS", sums_bytes, 0o644, mtime)
            for prog in PROGRAMS:
                add_bytes(tar, platform + "/" + prog, open(os.path.join(pdir, prog), "rb").read(), 0o755, mtime)
        apath = os.path.join(a.out, archive)
        with open(apath, "wb") as f:
            with gzip.GzipFile(filename="", mode="wb", fileobj=f, mtime=0, compresslevel=9) as gz:
                gz.write(raw.getvalue())

        platforms[platform] = dict(spec, **{
            "archive": {"name": archive, "sha256": sha256_file(apath), "bytes": os.path.getsize(apath)},
            "files": files,
            "needs": info["needs"],
            "cmake_flags": info["cmake_flags"],
            "toolchain": info["toolchain"],
            "version_output": info["version_output"],
        })

    if len(build_numbers) != 1:
        sys.exit("platforms report different build numbers: %s" % sorted(build_numbers))

    manifest = {
        "schema": SCHEMA,
        "tag": a.tag,
        "llama_build": a.commit,
        "build_number": build_numbers.pop(),
        "rpc_protocol": rpc_proto(a.src),
        "fork": {"repository": a.repository, "commit": a.commit},
        "upstream": {"repository": a.upstream_repository, "base_commit": a.upstream_base},
        "source_date_epoch": mtime,
        "workflow_run": a.run_url,
        "platforms": platforms,
    }
    with open(os.path.join(a.out, MANIFEST), "w", encoding="utf-8", newline="\n") as f:
        json.dump(manifest, f, indent=2, sort_keys=False)
        f.write("\n")

    names = sorted(n for n in os.listdir(a.out) if n != "SHA256SUMS")
    with open(os.path.join(a.out, "SHA256SUMS"), "w", encoding="utf-8", newline="\n") as f:
        for n in names:
            f.write("%s  %s\n" % (sha256_file(os.path.join(a.out, n)), n))
    print(open(os.path.join(a.out, "SHA256SUMS"), encoding="utf-8").read(), end="")


if __name__ == "__main__":
    main()
