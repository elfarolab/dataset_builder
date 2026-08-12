#!/usr/bin/env bash
set -euo pipefail

find ./web -type f -name "$(date +%Y%m%d)_*" -delete
find ./result -type f -name "e2e_test_dataset_*" -delete
find ./result -type f -name "test_export_*" -delete
