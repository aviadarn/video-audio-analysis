import os
import pytest
from celebvision.storage import BlobStore
from celebvision.config import Settings

pytestmark = pytest.mark.requires_stack


async def test_put_get_file_streams_large_file(tmp_path):
    s = Settings.from_env()
    store = BlobStore(s.minio_endpoint, s.minio_access_key, s.minio_secret_key)
    await store.ensure_bucket("test-bucket")
    src = tmp_path / "big.bin"
    src.write_bytes(os.urandom(5 * 1024 * 1024))   # 5 MB
    await store.put_file("test-bucket", "big.bin", str(src))
    dst = tmp_path / "out.bin"
    await store.get_file("test-bucket", "big.bin", str(dst))
    assert dst.read_bytes() == src.read_bytes()
