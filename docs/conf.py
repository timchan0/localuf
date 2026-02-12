from __future__ import annotations

import os
import sys
from datetime import date
from pathlib import Path


# Ensure the package is importable when building docs.
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))


project = "localuf"
copyright = f"{date.today().year}"  # noqa: A001
author = "Tim Chan"


extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
]


autosummary_generate = True

autodoc_default_options = {
    "members": True,
    "undoc-members": False,
    "show-inheritance": True,
}
autodoc_member_order = "bysource"
autodoc_typehints = "description"


templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]


html_theme = "alabaster"
html_static_path = ["_static"]


# If your docs environment does not have all runtime deps installed,
# you can add module names here to avoid import failures during autodoc.
autodoc_mock_imports: list[str] = []


# Be forgiving about missing references by default.
nitpicky = False


# Make sure Sphinx doesn't treat warnings as errors unless user opts in.
os.environ.setdefault("SPHINX_DISABLE_PROMPT", "1")
