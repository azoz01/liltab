from pathlib import Path
from pytest import fixture


@fixture(scope="session")
def resources_path():
    return Path("test/resources")
