#!/usr/bin/env bash
set -Eeuo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
trap 'echo "Start przerwany. Sprawdź błąd powyżej; ponowne uruchomienie użyje istniejącego cache." >&2' ERR
if [[ "$(uname -s)" != Linux ]]; then
  echo 'Wymagany Linux lub Ubuntu w WSL2.' >&2; exit 1
fi
missing=()
for name in python3 git curl ffmpeg; do
  command -v "$name" >/dev/null || missing+=("$name")
done
if (( ${#missing[@]} )); then
  command -v apt-get >/dev/null || { echo "Zainstaluj: ${missing[*]}"; exit 1; }
  elevate=()
  if (( EUID != 0 )); then elevate=(sudo); fi
  "${elevate[@]}" apt-get update
  "${elevate[@]}" apt-get install -y python3 git curl ffmpeg ca-certificates build-essential
fi
exec python3 bootstrap.py "$@"
