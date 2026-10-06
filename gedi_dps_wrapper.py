#!/usr/bin/env python3

import argparse
import tempfile
from pathlib import Path
from urllib.parse import urlparse
from concurrent.futures import ThreadPoolExecutor, as_completed

import boto3
import earthaccess
import h5py
import numpy as np
import pandas as pd

from boto3.s3.transfer import TransferConfig
from maap.maap import MAAP


# ============================================================
# ARGUMENTS
# ============================================================

parser = argparse.ArgumentParser()

parser.add_argument("--tile_id", required=True)
parser.add_argument("--west", type=float, required=True)
parser.add_argument("--south", type=float, required=True)
parser.add_argument("--east", type=float, required=True)
parser.add_argument("--north", type=float, required=True)
parser.add_argument("--start_date", required=True)
parser.add_argument("--end_date", required=True)
parser.add_argument("--workers", type=int, default=4)

args = parser.parse_args()

TILE_ID = args.tile_id
BBOX = (args.west, args.south, args.east, args.north)
GEDI_SHORT_NAME = "GEDI_L4A_AGB_Density_V3_2508"

START_TAG = args.start_date.replace("-", "")
END_TAG = args.end_date.replace("-", "")

OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_FILE = OUTPUT_DIR / f"{TILE_ID}_GEDI_L4A_V3_{START_TAG}_{END_TAG}.parquet"
FAILED_FILE = OUTPUT_DIR / f"{TILE_ID}_FAILED_GRANULES.txt"

maap = MAAP()

print("=" * 70)
print("GEDI L4A V3 — MAAP DPS")
print("=" * 70)
print("Tile      :", TILE_ID)
print("BBox      :", BBOX)
print("Period    :", args.start_date, args.end_date)
print("Workers   :", args.workers)
print("Output    :", OUTPUT_FILE)


# ============================================================
# SEARCH GEDI
# ============================================================

granules = earthaccess.search_data(
    short_name=GEDI_SHORT_NAME,
    temporal=(args.start_date, args.end_date),
    bounding_box=BBOX,
    cloud_hosted=True,
    count=-1,
)

print("Granules found:", len(granules))

urls = []
for granule in granules:
    links = [
        x
        for x in granule.data_links(access="direct")
        if x.startswith("s3://") and x.lower().endswith(".h5")
    ]
    if links:
        urls.append(links[0])

urls = sorted(set(urls))
print("Direct HDF5 links:", len(urls))


# ============================================================
# EMPTY TILE OUTPUT
# ============================================================

if not urls:
    empty = pd.DataFrame(
        columns=[
            "shot_number",
            "lat",
            "lon",
            "agbd",
            "agbd_se",
            "beam",
            "source_granule",
        ]
    )
    empty.to_parquet(OUTPUT_FILE, index=False)
    print("No GEDI granules intersect this tile.")
    print("Empty output:", OUTPUT_FILE)
    raise SystemExit(0)


# ============================================================
# EARTHDATA TEMPORARY CREDENTIALS
# ============================================================

creds = maap.aws.earthdata_s3_credentials(
    "https://data.ornldaac.earthdata.nasa.gov/s3credentials"
)

west, south, east, north = BBOX


# ============================================================
# PROCESS ONE GRANULE
# ============================================================

def process_granule(url):
    p = urlparse(url)
    granule_name = Path(p.path).name

    s3 = boto3.client(
        "s3",
        region_name="us-west-2",
        aws_access_key_id=creds["accessKeyId"],
        aws_secret_access_key=creds["secretAccessKey"],
        aws_session_token=creds["sessionToken"],
    )

    frames = []

    with tempfile.NamedTemporaryFile(suffix=".h5") as tmp:
        s3.download_file(
            p.netloc,
            p.path.lstrip("/"),
            tmp.name,
            Config=TransferConfig(use_threads=False),
        )

        with h5py.File(tmp.name, "r") as h:
            for beam in [x for x in h.keys() if x.startswith("BEAM")]:
                b = h[beam]

                required = [
                    "shot_number",
                    "lat_lowestmode",
                    "lon_lowestmode",
                    "agbd",
                    "agbd_se",
                    "l4a_quality_flag_rel3",
                    "degrade_include_flag",
                    "elev_highestreturn_outlier_flag",
                ]

                if not all(name in b for name in required):
                    continue

                shot = b["shot_number"][:]
                lat = b["lat_lowestmode"][:]
                lon = b["lon_lowestmode"][:]
                agbd = b["agbd"][:]
                agbd_se = b["agbd_se"][:]
                qflag = b["l4a_quality_flag_rel3"][:]
                degrade = b["degrade_include_flag"][:]
                elev_flag = b["elev_highestreturn_outlier_flag"][:]

                keep = (
                    np.isfinite(lat)
                    & np.isfinite(lon)
                    & np.isfinite(agbd)
                    & (lon >= west)
                    & (lon < east)
                    & (lat >= south)
                    & (lat < north)
                    & (qflag == 1)
                    & (degrade == 1)
                    & (elev_flag == 0)
                    & (agbd >= 0)
                )

                if not keep.any():
                    continue

                idx = np.flatnonzero(keep)

                frames.append(
                    pd.DataFrame(
                        {
                            "shot_number": shot[idx],
                            "lat": lat[idx].astype(np.float64),
                            "lon": lon[idx].astype(np.float64),
                            "agbd": agbd[idx].astype(np.float32),
                            "agbd_se": agbd_se[idx].astype(np.float32),
                            "beam": np.repeat(beam, idx.size),
                            "source_granule": np.repeat(granule_name, idx.size),
                        }
                    )
                )

    if not frames:
        return None

    return pd.concat(frames, ignore_index=True)


# ============================================================
# PARALLEL GRANULE PROCESSING INSIDE THIS TILE
# ============================================================

parts = []
failed = []

with ThreadPoolExecutor(max_workers=args.workers) as ex:
    futures = {ex.submit(process_granule, url): url for url in urls}

    for i, future in enumerate(as_completed(futures), start=1):
        url = futures[future]

        try:
            result = future.result()
            if result is not None:
                parts.append(result)
        except Exception as err:
            failed.append((Path(url).name, str(err)))
            print("FAILED:", Path(url).name, err)

        print(f"Processed {i}/{len(urls)}")


# ============================================================
# DO NOT SILENTLY ACCEPT A PARTIAL TILE
# ============================================================

if failed:
    with open(FAILED_FILE, "w", encoding="utf-8") as f:
        for name, err in failed:
            f.write(f"{name}\t{err}\n")

    raise RuntimeError(
        f"{len(failed)} GEDI granule(s) failed. "
        f"Tile output is intentionally not finalized; retry this DPS job. "
        f"See {FAILED_FILE}."
    )


# ============================================================
# COMBINE THIS TILE ONLY
# ============================================================

if parts:
    df = pd.concat(parts, ignore_index=True)

    before = len(df)
    df = df.drop_duplicates(subset="shot_number").reset_index(drop=True)
    print("Duplicate shots removed:", before - len(df))
else:
    df = pd.DataFrame(
        columns=[
            "shot_number",
            "lat",
            "lon",
            "agbd",
            "agbd_se",
            "beam",
            "source_granule",
        ]
    )


# ============================================================
# OUTPUT
# ============================================================

df.to_parquet(OUTPUT_FILE, index=False)

print("=" * 70)
print("DONE")
print("Tile :", TILE_ID)
print("Shots:", len(df))
print("File :", OUTPUT_FILE)
print("=" * 70)
