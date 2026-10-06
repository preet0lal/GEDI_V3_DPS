#!/usr/bin/env bash

set -euo pipefail

BASEDIR="$(cd "$(dirname "$0")" && pwd)"

mkdir -p output

echo "========================================"
echo "GEDI L4A V3 DPS"
echo "========================================"

echo "Tile   : $1"
echo "BBox   : $2 $3 $4 $5"
echo "Dates  : $6 $7"

python "${BASEDIR}/gedi_dps_wrapper.py" \
    --tile_id "$1" \
    --west "$2" \
    --south "$3" \
    --east "$4" \
    --north "$5" \
    --start_date "$6" \
    --end_date "$7" \
    --workers 4

echo "DONE"
