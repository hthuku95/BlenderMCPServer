"""Cloudflare R2 storage — upload renders and generate presigned download URLs."""

import hashlib
import logging
import os
import tempfile
import uuid
from pathlib import Path
from urllib.parse import urlparse

import boto3
from botocore.config import Config
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

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


def _ensure_r2_credentials():
    if not all([_ACCOUNT_ID, _ACCESS_KEY_ID, _SECRET_ACCESS_KEY]):
        raise RuntimeError(
            "R2 credentials not configured. Set R2_ACCOUNT_ID, R2_ACCESS_KEY_ID, "
            "R2_SECRET_ACCESS_KEY in .env"
        )


def _presign_object_key(client, object_key: str) -> str:
    return client.generate_presigned_url(
        "get_object",
        Params={"Bucket": _BUCKET_NAME, "Key": object_key},
        ExpiresIn=_PRESIGN_EXPIRY,
    )


def _guess_suffix(source_url: str, content_type: str) -> str:
    content_type = (content_type or "").lower()
    parsed_path = Path(urlparse(source_url).path)
    path_suffix = parsed_path.suffix.lower()
    if path_suffix in {".png", ".jpg", ".jpeg", ".webp", ".gif", ".svg"}:
        return path_suffix
    if "png" in content_type:
        return ".png"
    if "webp" in content_type:
        return ".webp"
    if "gif" in content_type:
        return ".gif"
    if "svg" in content_type:
        return ".svg"
    return ".jpg"


def upload_render(local_path: str, prefix: str = "renders") -> str:
    """Upload a local file to R2 and return a 24-hour presigned download URL."""
    _ensure_r2_credentials()

    suffix = Path(local_path).suffix
    object_key = f"{prefix}/{uuid.uuid4()}{suffix}"

    client = _r2_client()
    client.upload_file(local_path, _BUCKET_NAME, object_key)

    return _presign_object_key(client, object_key)


def host_remote_asset(source_url: str, prefix: str = "assets/reference_images") -> str:
    """
    Download a remote asset, store it in R2, and return a presigned URL.

    The object key is content-addressed so duplicate reference images can be
    reused across Blender jobs and future asset workflows.
    """
    import httpx

    _ensure_r2_credentials()

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/136.0.0.0 Safari/537.36"
        ),
        "Accept": "image/avif,image/webp,image/apng,image/svg+xml,image/*,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
        "Referer": "https://www.google.com/",
    }

    with httpx.Client(
        timeout=httpx.Timeout(45.0, connect=15.0),
        follow_redirects=True,
        headers=headers,
    ) as client:
        resp = client.get(source_url)
        resp.raise_for_status()
        content = resp.content
        content_type = resp.headers.get("content-type", "")

    if not content:
        raise RuntimeError(f"Downloaded asset from {source_url} was empty")

    suffix = _guess_suffix(source_url, content_type)
    digest = hashlib.sha256(content).hexdigest()[:24]
    object_key = f"{prefix}/{digest}{suffix}"
    client = _r2_client()

    try:
        client.head_object(Bucket=_BUCKET_NAME, Key=object_key)
        logger.info("storage.host_remote_asset_reused source=%s key=%s", source_url, object_key)
        return _presign_object_key(client, object_key)
    except Exception:
        pass

    tmp_path = ""
    try:
        with tempfile.NamedTemporaryFile(
            suffix=suffix,
            delete=False,
            prefix="r2_asset_",
        ) as tmp_file:
            tmp_file.write(content)
            tmp_path = tmp_file.name

        client.upload_file(tmp_path, _BUCKET_NAME, object_key)
        logger.info(
            "storage.host_remote_asset_uploaded source=%s key=%s bytes=%s",
            source_url,
            object_key,
            len(content),
        )
        return _presign_object_key(client, object_key)
    finally:
        if tmp_path:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass


def download_from_url(url: str, dest_path: str) -> str:
    """Download a presigned R2 URL to a local path. Returns dest_path."""
    import httpx

    with httpx.stream("GET", url, follow_redirects=True) as resp:
        resp.raise_for_status()
        with open(dest_path, "wb") as f:
            for chunk in resp.iter_bytes(chunk_size=65536):
                f.write(chunk)
    return dest_path
