"""Make ``src/`` importable when running a script directly.

Importing this module has the side effect of putting ``<repo>/src`` on the
import path, so ``import oulad`` works from ``python scripts/whatever.py``
without needing to install the package first.

If you would rather do it the standard way, ``pip install -e .`` installs the
package properly and makes this file unnecessary. It is kept so that a fresh
clone runs with no setup step beyond installing dependencies.
"""

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
