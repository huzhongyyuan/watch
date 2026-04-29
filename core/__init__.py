import os
from pathlib import Path

PROJ_ROOT = Path(__file__).resolve().parents[1]
PROJ_CORE = Path(__file__).resolve().parents[0]


def os_chdir_to_proj_root():
    """useful for running notebooks in different directories."""
    os.chdir(PROJ_ROOT)
