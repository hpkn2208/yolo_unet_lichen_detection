"""Downloads feedback images from R2 given a CSV exported from Supabase's
`feedback` table (Table Editor -> Export, or a SQL Editor query result).

Needs the original_path/overlay_path columns from that table. Reads R2
credentials the same way the app does, from .streamlit/secrets.toml.

Usage:
    python scripts/download_feedback_images.py path/to/feedback_export.csv [output_dir]
"""

import csv
import sys
from pathlib import Path

import boto3
import toml

SECRETS_PATH = Path(__file__).parent.parent / ".streamlit" / "secrets.toml"


def _client():
    secrets = toml.load(SECRETS_PATH)
    r2 = secrets["r2"]
    client = boto3.client(
        "s3",
        endpoint_url=f"https://{r2['account_id']}.r2.cloudflarestorage.com",
        aws_access_key_id=r2["access_key_id"],
        aws_secret_access_key=r2["secret_access_key"],
        region_name="auto",
    )
    return client, r2["bucket"]


def download(csv_path: str, out_dir: str = "downloaded_feedback") -> None:
    client, bucket = _client()
    out = Path(out_dir)
    out.mkdir(exist_ok=True)

    with open(csv_path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    keys = sorted({row[col] for row in rows for col in ("original_path", "overlay_path") if row.get(col)})

    for i, key in enumerate(keys, 1):
        # key looks like "research-app/<category>/<filename>.png" — drop the
        # research-app/ prefix so files land in <out_dir>/<category>/<filename>.png
        dest = out / key.split("/", 1)[1]
        dest.parent.mkdir(parents=True, exist_ok=True)
        client.download_file(bucket, key, str(dest))
        print(f"[{i}/{len(keys)}] {key}")

    print(f"Done — {len(keys)} images saved to {out}/")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python scripts/download_feedback_images.py path/to/feedback_export.csv [output_dir]")
        sys.exit(1)
    download(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "downloaded_feedback")
