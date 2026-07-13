import pytest
from celebvision.storage import BlobStore
from celebvision.config import Settings

pytestmark = pytest.mark.requires_stack


async def test_put_get_round_trip():
    s = Settings.from_env()
    store = BlobStore(s.minio_endpoint, s.minio_access_key, s.minio_secret_key)
    await store.ensure_bucket("test-bucket")
    ref = await store.put_bytes("test-bucket", "hello.txt", b"hi there")
    assert ref == "test-bucket/hello.txt"
    assert await store.get_bytes("test-bucket", "hello.txt") == b"hi there"
