#!/usr/bin/env bash
# Download the reference papers into this folder. The PDFs are git-ignored:
# they're copyrighted and this repo is public. See README.md for citations.
set -euo pipefail
cd "$(dirname "$0")"

fetch() {
  local name=$1 url=$2 sha=$3
  if [[ -f $name ]]; then
    echo "have $name"
  else
    echo "fetching $name"
    curl -fsSL -A "Mozilla/5.0" -o "$name.tmp" "$url"
    mv "$name.tmp" "$name"
  fi
  # Warn rather than fail: the hosts may re-scan or re-stamp the files.
  if ! echo "$sha  $name" | sha256sum --check --status; then
    echo "warning: $name checksum differs from the copy these notes were written against" >&2
  fi
}

fetch wallis-north-1986-measuring-the-transaction-sector.pdf \
  "https://www.nber.org/system/files/chapters/c9679/c9679.pdf" \
  5954116e285eb958a12f2bd3dc52ee149d28e9ae95d02a9e1870535dc099d789

fetch wallis-north-1988-should-transaction-costs-be-subtracted.pdf \
  "https://www.econ.umd.edu/sites/www.econ.umd.edu/files/pubs/Wallis%26North_ShouldTC_JEH_1988.pdf" \
  67f2208e6b0318c3355f13370dc6e22a4bf58de32cbd6ecfc1fcc58168587c3d
