#!/usr/bin/env bash
# Verify a downloaded Potluck llama release.
#
#   .github/potluck/verify.sh <dir>
#
# <dir> holds the release assets (SHA256SUMS, potluck-llama-manifest.json and
# the platform archives). Checks, in order:
#   1. every asset against SHA256SUMS;
#   2. each archive's sha256 and size against the manifest;
#   3. each unpacked program against its folder's SHA256SUMS and the manifest.
# Needs bash, sha256sum, tar and python3. Exits non-zero on the first mismatch.
set -euo pipefail

dir=$(cd "${1:?release dir}" && pwd)
cd "$dir"
sha256sum --strict -c SHA256SUMS

work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT

python3 - "$dir" "$work" <<'EOF'
import hashlib, json, os, subprocess, sys
d, work = sys.argv[1], sys.argv[2]
m = json.load(open(os.path.join(d, "potluck-llama-manifest.json"), encoding="utf-8"))
def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()
for name, p in m["platforms"].items():
    a = p["archive"]
    ap = os.path.join(d, a["name"])
    if sha(ap) != a["sha256"] or os.path.getsize(ap) != a["bytes"]:
        sys.exit("%s: archive does not match the manifest" % a["name"])
    subprocess.run(["tar", "-xzf", ap, "-C", work], check=True)
    pdir = os.path.join(work, name)
    subprocess.run(["sha256sum", "--strict", "-c", "SHA256SUMS"], cwd=pdir, check=True)
    for prog, f in p["files"].items():
        fp = os.path.join(pdir, prog)
        if sha(fp) != f["sha256"] or os.path.getsize(fp) != f["bytes"]:
            sys.exit("%s/%s: does not match the manifest" % (name, prog))
    print("%s: ok (%s)" % (name, ", ".join(sorted(p["files"]))))
print("release %s, llama build %s: verified" % (m["tag"], m["llama_build"]))
EOF
