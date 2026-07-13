import os
import pytest


def pytest_collection_modifyitems(config, items):
    if os.environ.get("CELEBVISION_STACK") == "1":
        return
    skip = pytest.mark.skip(reason="needs docker stack; set CELEBVISION_STACK=1")
    for item in items:
        if "requires_stack" in item.keywords:
            item.add_marker(skip)
