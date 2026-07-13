import asyncio
from celebvision.config import Settings
from celebvision.bus import MessageBus
from celebvision.db import Database
from celebvision.coordinator import run_coordinator


async def main() -> None:
    settings = Settings.from_env()
    db = Database(settings.postgres_dsn)
    await db.connect()
    bus = MessageBus(settings.kafka_bootstrap)
    await bus.start()
    try:
        await run_coordinator(bus, db)
    finally:
        await bus.stop()
        await db.close()


if __name__ == "__main__":
    asyncio.run(main())
