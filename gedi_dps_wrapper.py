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

parser.add_argument(
    "--workers",
    type=int,
    default=4
)

args = parser.parse_args()


TILE_ID = args.tile_id

BBOX = (
    args.west,
    args.south,
    args.east,
    args.north,
)

GEDI_SHORT_NAME = "GEDI_L4A_AGB_Density_V3_2508"

OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

maap = MAAP()


print("=" * 70)
print("GEDI L4A V3 — MAAP DPS")
print("=" * 70)

print("Tile      :", TILE_ID)
print("BBox      :", BBOX)
print("Period    :", args.start_date, args.end_date)
print("Workers   :", args.workers)


# ============================================================
# SEARCH GEDI
# ============================================================

granules = earthaccess.search_data(
    short_name=GEDI_SHORT_NAME,
    temporal=(
        args.start_date,
        args.end_date
    ),
    bounding_box=BBOX,
    cloud_hosted=True,
    count=-1,
)

print("Granules found:", len(granules))


urls = []

for g in granules:

    links = [
        x for x in g.data_links(access="direct")
        if (
            x.startswith("s3://")
            and x.lower().endswith(".h5")
        )
    ]

    if links:
        urls.append(links[0])


urls = sorted(set(urls))

print("Direct HDF5 links:", len(urls))


if not urls:

    # Still create an empty output
    empty = pd.DataFrame(
        columns=[
            "shot_number",
            "lat",
            "lon",
            "agbd",
            "agbd_se",
            "beam",
        ]
    )

    out = OUTPUT_DIR / f"{TILE_ID}_GEDI_L4A_V3.parquet"

    empty.to_parquet(
        out,
        index=False
    )

    print("No GEDI granules. Empty output:", out)

    raise SystemExit(0)


# ============================================================
# EARTHDATA TEMPORARY CREDENTIALS
# ============================================================

creds = maap.aws.earthdata_s3_credentials(
    "https://data.ornldaac.earthdata.nasa.gov/s3credentials"
)


# ============================================================
# ONE GRANULE
# ============================================================

west, south, east, north = BBOX


def process_granule(url):

    p = urlparse(url)

    s3 = boto3.client(
        "s3",
        region_name="us-west-2",
        aws_access_key_id=creds["accessKeyId"],
        aws_secret_access_key=creds["secretAccessKey"],
        aws_session_token=creds["sessionToken"],
    )

    frames = []

    with tempfile.NamedTemporaryFile(
        suffix=".h5"
    ) as tmp:

        s3.download_file(
            p.netloc,
            p.path.lstrip("/"),
            tmp.name,
            Config=TransferConfig(
                use_threads=False
            ),
        )

        with h5py.File(
            tmp.name,
            "r"
        ) as h:

            for beam in [
                x for x in h.keys()
                if x.startswith("BEAM")
            ]:

                b = h[beam]

                required = [
                    "lat_lowestmode",
                    "lon_lowestmode",
                    "agbd",
                    "agbd_se",
                    "l4a_quality_flag_rel3",
                    "degrade_include_flag",
                    "elev_highestreturn_outlier_flag",
                ]

                if not all(
                    x in b
                    for x in required
                ):
                    continue


                lat = b["lat_lowestmode"][:]
                lon = b["lon_lowestmode"][:]

                agbd = b["agbd"][:]
                agbd_se = b["agbd_se"][:]

                qflag = b[
                    "l4a_quality_flag_rel3"
                ][:]

                degrade = b[
                    "degrade_include_flag"
                ][:]

                elev_flag = b[
                    "elev_highestreturn_outlier_flag"
                ][:]


                keep = (
                    np.isfinite(lat)
                    & np.isfinite(lon)
                    & np.isfinite(agbd)

                    # Half-open tile boundaries prevent
                    # most cross-tile duplication
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


                idx = np.flatnonzero(
                    keep
                )


                data = {
                    "lat":
                        lat[idx].astype(
                            np.float64
                        ),

                    "lon":
                        lon[idx].astype(
                            np.float64
                        ),

                    "agbd":
                        agbd[idx].astype(
                            np.float32
                        ),

                    "agbd_se":
                        agbd_se[idx].astype(
                            np.float32
                        ),

                    "beam":
                        np.repeat(
                            beam,
                            idx.size
                        ),
                }


                if "shot_number" in b:

                    shot = b[
                        "shot_number"
                    ][:]

                    data["shot_number"] = (
                        shot[idx]
                    )

                else:

                    data["shot_number"] = (
                        np.arange(idx.size)
                    )


                frames.append(
                    pd.DataFrame(data)
                )


    if frames:

        return pd.concat(
            frames,
            ignore_index=True
        )

    return None


# ============================================================
# PARALLEL GRANULE PROCESSING INSIDE THIS TILE
# ============================================================

parts = []
failed = []

with ThreadPoolExecutor(
    max_workers=args.workers
) as ex:

    futures = {
        ex.submit(
            process_granule,
            url
        ): url
        for url in urls
    }


    for i, future in enumerate(
        as_completed(futures),
        start=1
    ):

        url = futures[future]

        try:

            result = future.result()

            if result is not None:
                parts.append(result)

        except Exception as err:

            failed.append(
                (
                    Path(url).name,
                    str(err)
                )
            )

            print(
                "FAILED:",
                Path(url).name,
                err
            )


        print(
            f"Processed {i}/{len(urls)}"
        )


# ============================================================
# COMBINE THIS TILE ONLY
# ============================================================

if parts:

    df = pd.concat(
        parts,
        ignore_index=True
    )

    if "shot_number" in df.columns:

        df = (
            df
            .drop_duplicates(
                subset="shot_number"
            )
            .reset_index(drop=True)
        )

else:

    df = pd.DataFrame(
        columns=[
            "shot_number",
            "lat",
            "lon",
            "agbd",
            "agbd_se",
            "beam",
        ]
    )


# ============================================================
# OUTPUT
# ============================================================

out = (
    OUTPUT_DIR
    / f"{TILE_ID}_GEDI_L4A_V3_2019_2023.parquet"
)

df.to_parquet(
    out,
    index=False
)


print("=" * 70)
print("DONE")
print("Tile :", TILE_ID)
print("Shots:", len(df))
print("File :", out)

if failed:
    print("Failed granules:", len(failed))

print("=" * 70)
