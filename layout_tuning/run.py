"""Entry point: python -m layout_tuning.run <experiment>"""

import sys

from .runner import main

if __name__ == "__main__":
    sys.exit(main())
