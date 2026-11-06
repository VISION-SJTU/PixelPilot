#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Check selected release files for credential patterns and local private paths."""

import json
import re

from export_release import ROOT, release_files

PATTERNS = {
    "private machine path": re.compile(r"/(?:home|Users)/(?:bml|ma-user|tangpin)/"),
    "Hugging Face token": re.compile(r"hf_[A-Za-z0-9]{25,}"),
    "API token": re.compile(r"sk-(?:proj-)?[A-Za-z0-9_-]{20,}"),
    "GitHub token": re.compile(r"gh[pousr]_[A-Za-z0-9]{30,}"),
    "AWS access id": re.compile(r"AKIA[0-9A-Z]{16}"),
    "private key": re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
}


def main():
    findings = []
    files = release_files()
    for path in files:
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            for kind, pattern in PATTERNS.items():
                if pattern.search(line):
                    findings.append({"file": str(path.relative_to(ROOT)), "line": number, "kind": kind})
    print(json.dumps({"files_checked": len(files), "findings": findings}, indent=2))
    if findings:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
