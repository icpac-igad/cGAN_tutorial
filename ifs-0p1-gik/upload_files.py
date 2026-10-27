"""Copy finished files (the GAN inference output, optionally the IFS NetCDF) to
an object store, then (with --remove) delete the local copy.

    # source.coop: needs the STS credentials of `source-coop creds` in the env
    python upload_files.py --to sourcecoop:forecasts/cgan-ifs-0p1-gan-19e55e-25n15s/20261021T00z \\
        gan/GAN_IFS_20261021_00Z.nc
    # EWC Ceph: AK/SK from --env
    python upload_files.py --to ceph:cgan-ifs-0p1-gan/20261021T00z --env ../.env gan/*.nc --remove

Multipart above 64 MiB (source.coop refuses single PUTs over ~100 MB). The size
of every object is checked after upload. Credentials are never printed.
"""
from __future__ import annotations

import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import stores as S  # noqa: E402


def client(kind, env_file):
    import boto3
    from botocore.config import Config
    cfg = Config(s3={"addressing_style": "path"},
                 retries={"max_attempts": 10, "mode": "adaptive"})
    if kind == "ceph":
        if not env_file:
            sys.exit("a ceph: destination needs --env")
        ak, sk = S.load_env(env_file)
        return boto3.client("s3", endpoint_url=S.CEPH_ENDPOINT, aws_access_key_id=ak,
                            aws_secret_access_key=sk, region_name="RegionOne",
                            config=cfg), S.CEPH_BUCKET
    if "AWS_ACCESS_KEY_ID" not in os.environ:
        sys.exit("source.coop: run inside `eval \"$(source-coop creds --format env)\"`")
    return boto3.client("s3", endpoint_url=os.environ.get("AWS_ENDPOINT_URL", S.SC_ENDPOINT),
                        region_name=os.environ.get("AWS_DEFAULT_REGION", "us-east-1"),
                        config=cfg), S.SC_BUCKET


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="+")
    ap.add_argument("--to", required=True, help="sourcecoop:<prefix> or ceph:<prefix>")
    ap.add_argument("--env", default=None, help="Ceph AK/SK file (ceph: only)")
    ap.add_argument("--remove", action="store_true", help="delete the local file once copied")
    a = ap.parse_args()

    from boto3.s3.transfer import TransferConfig
    kind, prefix = S.parse(a.to)
    if kind == "local":
        sys.exit("--to must be sourcecoop: or ceph:")
    c, bucket = client(kind, a.env)
    cfg = TransferConfig(multipart_threshold=64 << 20, multipart_chunksize=64 << 20,
                         max_concurrency=8)
    for f in a.files:
        key = f"{prefix}/{os.path.basename(f)}"
        c.upload_file(f, bucket, key, Config=cfg)
        size = c.head_object(Bucket=bucket, Key=key)["ContentLength"]
        ok = size == os.path.getsize(f)
        print(f"{kind}:{bucket}/{key}  {size / 1e9:.2f} GB  {'ok' if ok else 'SIZE MISMATCH'}")
        if not ok:
            sys.exit(1)
        if a.remove:
            os.remove(f)


if __name__ == "__main__":
    main()
