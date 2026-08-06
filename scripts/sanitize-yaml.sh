#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   ./sanitize-yaml.sh input.yaml output.yaml --sanitize
#   ./sanitize-yaml.sh input.yaml output.yaml --restore --lan-ip 192.168.1.145
#
# Examples:
#   ./sanitize-yaml.sh config.yml config.sanitized.yml --sanitize
#   ./sanitize-yaml.sh config.sanitized.yml config.restored.yml --restore --lan-ip 192.168.1.145

if [[ $# -lt 3 ]]; then
  echo "Usage: $0 input.yaml output.yaml (--sanitize | --restore --lan-ip IP)"
  exit 1
fi

input="$1"
output="$2"
mode="$3"

placeholder="REPLACE_ME_LAN_IP"

if [[ "$mode" == "--sanitize" ]]; then
  # Replace private IPv4 only, inside http/https URLs. Keeps optional :port.
  sed -E \
    "s#(https?://)(10\\.[0-9]{1,3}\\.[0-9]{1,3}\\.[0-9]{1,3}|192\\.168\\.[0-9]{1,3}\\.[0-9]{1,3}|172\\.(1[6-9]|2[0-9]|3[0-1])\\.[0-9]{1,3}\\.[0-9]{1,3})(:[0-9]+)?#\\1${placeholder}\\3#g" \
    "$input" > "$output"

elif [[ "$mode" == "--restore" ]]; then
  lan_ip=""

  # Parse --lan-ip
  if [[ "${4:-}" != "--lan-ip" ]]; then
    echo "Missing --lan-ip. Usage: $0 input.yaml output.yaml --restore --lan-ip IP"
    exit 1
  fi
  lan_ip="${5:-}"
  if [[ -z "$lan_ip" ]]; then
    echo "Missing IP after --lan-ip"
    exit 1
  fi

  # Replace placeholder with provided IP (assumes placeholder appears in the URL host part).
  sed -E "s#${placeholder}#${lan_ip}#g" "$input" > "$output"

else
  echo "Unknown mode: $mode"
  echo "Usage: $0 input.yaml output.yaml (--sanitize | --restore --lan-ip IP)"
  exit 1
fi

echo "Wrote: $output"

