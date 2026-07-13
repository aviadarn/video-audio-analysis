import pytest
from httpx import AsyncClient, ASGITransport
from celebvision.api.app import create_app

pytestmark = pytest.mark.requires_stack


async def test_metrics_endpoint_returns_prometheus_text():
    app = create_app()
    async with app.router.lifespan_context(app):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as c:
            r = await c.get("/metrics")
            assert r.status_code == 200
            assert "celebvision_stage_processed_total" in r.text
