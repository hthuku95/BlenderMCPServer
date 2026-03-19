"""Cloudflare R2 storage — upload renders and generate presigned download URLs."""

import os
import uuid
from pathlib import Path

import boto3
from botocore.config import Config
from dotenv import load_dotenv

load_dotenv()

_ACCOUNT_ID = os.getenv("R2_ACCOUNT_ID", "")
_ACCESS_KEY_ID = os.getenv("R2_ACCESS_KEY_ID", "")
_SECRET_ACCESS_KEY = os.getenv("R2_SECRET_ACCESS_KEY", "")
_BUCKET_NAME = os.getenv("R2_BUCKET_NAME", "blender-outputs")
_PRESIGN_EXPIRY = int(os.getenv("R2_PRESIGN_EXPIRY", "86400"))  # 24 hours


def _r2_client():
    return boto3.client(
        "s3",
        endpoint_url=f"https://{_ACCOUNT_ID}.r2.cloudflarestorage.com",
        aws_access_key_id=_ACCESS_KEY_ID,
        aws_secret_access_key=_SECRET_ACCESS_KEY,
        config=Config(signature_version="s3v4"),
        region_name="auto",
    )


def upload_render(local_path: str, prefix: str = "renders") -> str:
    """Upload a local file to R2 and return a 24-hour presigned download URL."""
    if not all([_ACCOUNT_ID, _ACCESS_KEY_ID, _SECRET_ACCESS_KEY]):
        raise RuntimeError(
            "R2 credentials not configured. Set R2_ACCOUNT_ID, R2_ACCESS_KEY_ID, "
            "R2_SECRET_ACCESS_KEY in .env"
        )

    suffix = Path(local_path).suffix
    object_key = f"{prefix}/{uuid.uuid4()}{suffix}"

    client = _r2_client()
    client.upload_file(local_path, _BUCKET_NAME, object_key)

    url = client.generate_presigned_url(
        "get_object",
        Params={"Bucket": _BUCKET_NAME, "Key": object_key},
        ExpiresIn=_PRESIGN_EXPIRY,
    )
    return url


def download_from_url(url: str, dest_path: str) -> str:
    """Download a presigned R2 URL to a local path. Returns dest_path."""
    import httpx

    with httpx.stream("GET", url, follow_redirects=True) as resp:
        resp.raise_for_status()
        with open(dest_path, "wb") as f:
            for chunk in resp.iter_bytes(chunk_size=65536):
                f.write(chunk)
    return dest_path
