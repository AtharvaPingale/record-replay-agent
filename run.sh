#!/usr/bin/env bash
# Wrapper so every script in this repo gets the repo root on PYTHONPATH
# (packages are flat top-level directories, not nested under src/).
#
# The LD_LIBRARY_PATH line is a sandbox-only accommodation: the machine this was
# built on had no system audio lib and couldn't run `playwright install-deps`,
# so libasound.so.2 was extracted by hand. It is only applied if that directory
# exists, so on a normal machine it's a no-op.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXTLIBS="$HOME/.local/extlibs/usr/lib/x86_64-linux-gnu"
if [ -d "$EXTLIBS" ]; then
  export LD_LIBRARY_PATH="$EXTLIBS:${LD_LIBRARY_PATH:-}"
fi
export PYTHONPATH="$HERE:${PYTHONPATH:-}"
UV="$(command -v uv || echo "$HOME/.local/bin/uv")"
exec "$UV" run --project "$HERE" "$@"
