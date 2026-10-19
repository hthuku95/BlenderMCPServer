"""Object storage — upload renders and generate presigned download URLs.

Primary backend is Cloudflare R2. When R2 is unavailable (e.g. an account-wide
billing/entitlement outage such as `NotEntitled`) every upload transparently
falls back to real AWS S3, mirroring the Rust `StorageRouter` in
`src/storage.rs`. The env contract matches the Rust side so a single set of
credentials configures both processes:

    S3_FALLBACK_BUCKET          enable the fallback (empty/unset = R2 only)
    S3_REGION | AWS_REGION      fallback region (default us-east-1)
    S3_FALLBACK_ACCESS_KEY_ID   optional long-lived signing key
    S3_FALLBACK_SECRET_ACCESS_KEY
    AWS_ACCESS_KEY_ID           used when S3_FALLBACK_* is not set
    AWS_SECRET_ACCESS_KEY
"""

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

# --- R2 (primary) ----------------------------------------------------------
_ACCOUNT_ID = os.getenv("R2_ACCOUNT_ID", "")
_ACCESS_KEY_ID = os.getenv("R2_ACCESS_KEY_ID", "")
_SECRET_ACCESS_KEY = os.getenv("R2_SECRET_ACCESS_KEY", "")
_BUCKET_NAME = os.getenv("R2_BUCKET_NAME", "blender-outputs")
_PRESIGN_EXPIRY = int(os.getenv("R2_PRESIGN_EXPIRY", "86400"))  # 24 hours

# --- AWS S3 (fallback) -----------------------------------------------------
_S3_BUCKET = os.getenv("S3_FALLBACK_BUCKET", "")
_S3_REGION = (
    os.getenv("S3_REGION") or os.getenv("AWS_REGION") or "us-east-1"
).strip() or "us-east-1"
_S3_STATIC_KEY = os.getenv("S3_FALLBACK_ACCESS_KEY_ID", "").strip()
_S3_STATIC_SECRET = os.getenv("S3_FALLBACK_SECRET_ACCESS_KEY", "").strip()

# Last backend used, for observability in callers/logs.
LAST_BACKEND = "r2"


def _r2_configured() -> bool:
    return all([_ACCOUNT_ID, _ACCESS_KEY_ID, _SECRET_ACCESS_KEY])


def _s3_enabled() -> bool:
    return bool(_S3_BUCKET.strip())


def _s3_credentials_present() -> bool:
    if _S3_STATIC_KEY and _S3_STATIC_SECRET:
        return True
    # boto3 falls back to the default provider chain (env, IMDS, task role).
    return bool(
        (os.getenv("AWS_ACCESS_KEY_ID") and os.getenv("AWS_SECRET_ACCESS_KEY"))
        or os.getenv("AWS_CONTAINER_CREDENTIALS_RELATIVE_URI")
        or os.getenv("AWS_CONTAINER_CREDENTIALS_FULL_URI")
    )


def _r2_client():
    return boto3.client(
        "s3",
        endpoint_url=f"https://{_ACCOUNT_ID}.r2.cloudflarestorage.com",
        aws_access_key_id=_ACCESS_KEY_ID,
        aws_secret_access_key=_SECRET_ACCESS_KEY,
        config=Config(signature_version="s3v4"),
        region_name="auto",
    )


def _s3_client():
    kwargs = {"region_name": _S3_REGION}
    if _S3_STATIC_KEY and _S3_STATIC_SECRET:
        kwargs["aws_access_key_id"] = _S3_STATIC_KEY
        kwargs["aws_secret_access_key"] = _S3_STATIC_SECRET
    return boto3.client("s3", **kwargs)


def _presign(client, bucket: str, object_key: str) -> str:
    return client.generate_presigned_url(
        "get_object",
        Params={"Bucket": bucket, "Key": object_key},
        ExpiresIn=_PRESIGN_EXPIRY,
    )


def _presign_object_key(client, object_key: str) -> str:
    return _presign(client, _BUCKET_NAME, object_key)


def _upload_r2(local_path: str, object_key: str, content_type: str = "") -> str:
    client = _r2_client()
    extra = {"ContentType": content_type} if content_type else None
    if extra:
        client.upload_file(local_path, _BUCKET_NAME, object_key, ExtraArgs=extra)
    else:
        client.upload_file(local_path, _BUCKET_NAME, object_key)
    return _presign_object_key(client, object_key)


def _upload_s3(local_path: str, object_key: str, content_type: str = "") -> str:
    client = _s3_client()
    extra = {"ContentType": content_type} if content_type else None
    if extra:
        client.upload_file(local_path, _S3_BUCKET, object_key, ExtraArgs=extra)
    else:
        client.upload_file(local_path, _S3_BUCKET, object_key)
    return _presign(client, _S3_BUCKET, object_key)


def _upload_with_fallback(
    local_path: str, object_key: str, content_type: str = ""
) -> str:
    """Upload to R2 first, then AWS S3. Returns a presigned URL.

    Re-raises only when every configured backend fails, so callers never see a
    silently-lost render.
    """
    global LAST_BACKEND
    errors: list[str] = []

    if _r2_configured():
        try:
            url = _upload_r2(local_path, object_key, content_type)
            LAST_BACKEND = "r2"
            logger.info("storage.upload backend=r2 key=%s", object_key)
            return url
        except Exception as exc:  # noqa: BLE001 - fall through to S3
            errors.append(f"r2: {exc}")
            logger.warning(
                "storage.r2_upload_failed key=%s err=%s — trying S3 fallback",
                object_key,
                exc,
            )
    else:
        errors.append("r2: not configured")

    if _s3_enabled() and _s3_credentials_present():
        try:
            url = _upload_s3(local_path, object_key, content_type)
            LAST_BACKEND = "s3"
            logger.info("storage.upload backend=s3 key=%s", object_key)
            return url
        except Exception as exc:  # noqa: BLE001
            errors.append(f"s3: {exc}")
            logger.error("storage.s3_upload_failed key=%s err=%s", object_key, exc)
    else:
        errors.append(
            "s3: not enabled (set S3_FALLBACK_BUCKET + credentials to enable)"
        )

    raise RuntimeError(
        "object storage upload failed — " + "; ".join(errors)
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


def _existing_url(object_key: str) -> str | None:
    """Return a presigned URL if the key exists in R2 or S3, else None."""
    global LAST_BACKEND

    if _r2_configured():
        try:
            client = _r2_client()
            client.head_object(Bucket=_BUCKET_NAME, Key=object_key)
            LAST_BACKEND = "r2"
            return _presign_object_key(client, object_key)
        except Exception:  # noqa: BLE001 - miss or R2 outage
            pass

    if _s3_enabled() and _s3_credentials_present():
        try:
            client = _s3_client()
            client.head_object(Bucket=_S3_BUCKET, Key=object_key)
            LAST_BACKEND = "s3"
            return _presign(client, _S3_BUCKET, object_key)
        except Exception:  # noqa: BLE001
            pass

    return None


def upload_render(local_path: str, prefix: str = "renders") -> str:
    """Upload a local file and return a 24-hour presigned download URL.

    Uses R2 when healthy and transparently falls back to AWS S3 otherwise.
    """
    suffix = Path(local_path).suffix
    object_key = f"{prefix}/{uuid.uuid4()}{suffix}"
    return _upload_with_fallback(local_path, object_key)


def host_remote_asset(source_url: str, prefix: str = "assets/reference_images") -> str:
    """
    Download a remote asset, store it, and return a presigned URL.

    The object key is content-addressed so duplicate reference images can be
    reused across Blender jobs and future asset workflows. Reuse checks both
    backends, so an asset uploaded during an R2 outage stays reachable.
    """
    import httpx

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

    reused = _existing_url(object_key)
    if reused:
        logger.info(
            "storage.host_remote_asset_reused source=%s key=%s backend=%s",
            source_url,
            object_key,
            LAST_BACKEND,
        )
        return reused

    tmp_path = ""
    try:
        with tempfile.NamedTemporaryFile(
            suffix=suffix,
            delete=False,
            prefix="asset_",
        ) as tmp_file:
            tmp_file.write(content)
            tmp_path = tmp_file.name

        url = _upload_with_fallback(tmp_path, object_key, content_type)
        logger.info(
            "storage.host_remote_asset_uploaded source=%s key=%s bytes=%s backend=%s",
            source_url,
            object_key,
            len(content),
            LAST_BACKEND,
        )
        return url
    finally:
        if tmp_path:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass


def download_from_url(url: str, dest_path: str) -> str:
    """Download a presigned URL (R2 or S3) to a local path. Returns dest_path."""
    import httpx

    with httpx.stream("GET", url, follow_redirects=True) as resp:
        resp.raise_for_status()
        with open(dest_path, "wb") as f:
            for chunk in resp.iter_bytes(chunk_size=65536):
                f.write(chunk)
    return dest_path
