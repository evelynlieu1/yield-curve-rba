"""pytest setup: put src/ on the import path so tests can 'import data'.

Run from the project folder:
    python -m pytest tests
pytest is in requirements.txt (installed with the other packages).
"""

import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
