import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
# The participant agent is a flat set of modules (shipped as a folder to participants).
for extra in (HERE.parent / "participant_agent", HERE.parent):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))


import shutil

import pytest

V4_REFERENCE_DIR = HERE.parent / "reference" / "v4"


@pytest.fixture(scope="session")
def v4_reference_dir(tmp_path_factory):
    """A temp copy of challenge/reference/v4 with its products generated (about 5 s).

    The reference configs are the v4 author's sample configs (public sample seed, legacy
    seed streams), so the author's regression anchors still hold. They are never a card.
    """
    from challenge import v4_catalog_generator, v4_weather_simulator

    root = tmp_path_factory.mktemp("v4_reference")
    for path in V4_REFERENCE_DIR.glob("*.json"):
        shutil.copyfile(path, root / path.name)
    v4_catalog_generator.generate_catalog(root / "v4_catalog_config.json", root / "products")
    v4_weather_simulator.generate(root / "v4_weather_config.json", root / "products")
    v4_weather_simulator.generate(root / "v4_weather_stress_config.json", root / "products" / "stress")
    return root
