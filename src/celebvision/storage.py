import aioboto3
from botocore.config import Config
from botocore.exceptions import ClientError


class BlobStore:
    def __init__(self, endpoint: str, access_key: str, secret_key: str) -> None:
        self._endpoint = endpoint
        self._access_key = access_key
        self._secret_key = secret_key
        self._session = aioboto3.Session()

    def _client(self):
        return self._session.client(
            "s3", endpoint_url=self._endpoint,
            aws_access_key_id=self._access_key,
            aws_secret_access_key=self._secret_key,
            config=Config(signature_version="s3v4"),
        )

    async def ensure_bucket(self, bucket: str) -> None:
        async with self._client() as s3:
            try:
                await s3.head_bucket(Bucket=bucket)
            except ClientError:
                await s3.create_bucket(Bucket=bucket)

    async def put_bytes(self, bucket: str, key: str, data: bytes) -> str:
        async with self._client() as s3:
            await s3.put_object(Bucket=bucket, Key=key, Body=data)
        return f"{bucket}/{key}"

    async def get_bytes(self, bucket: str, key: str) -> bytes:
        async with self._client() as s3:
            resp = await s3.get_object(Bucket=bucket, Key=key)
            async with resp["Body"] as stream:
                return await stream.read()

    async def put_file(self, bucket: str, key: str, path: str) -> str:
        with open(path, "rb") as f:
            return await self.put_bytes(bucket, key, f.read())

    async def get_file(self, bucket: str, key: str, dest: str) -> str:
        data = await self.get_bytes(bucket, key)
        with open(dest, "wb") as f:
            f.write(data)
        return dest
