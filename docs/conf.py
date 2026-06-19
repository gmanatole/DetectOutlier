"""Sphinx configuration for the OutlierDetect documentation."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

project = "OutlierDetect"
author = "Anatole Gros-Martial, Fabien Roquet"
copyright = "2026, Anatole Gros-Martial, Fabien Roquet"

extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
]

source_suffix = {
    ".md": "markdown",
}

myst_enable_extensions = [
    "colon_fence",
    "deflist",
]
myst_heading_anchors = 3

autosummary_generate = True
autoclass_content = "both"
autodoc_typehints = "description"
autodoc_member_order = "bysource"
autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "private-members": True,
    "show-inheritance": True,
}

templates_path: list[str] = []
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]
html_static_path: list[str] = []
html_title = "OutlierDetect"

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", {}),
    "numpy": ("https://numpy.org/doc/stable/", {}),
}

try:
    from outlierdetect import __version__

    release = __version__
except Exception:  # pragma: no cover - docs should still build from source tree
    release = "0.1.0"

version = release
