"""Define the nrtk package"""

import importlib
import importlib.metadata
import sys
import warnings
from datetime import datetime

__version__ = importlib.metadata.version(__name__)


# Time delay deprecation
if datetime.now() > datetime(2025, 5, 1):
    sys.exit(
        "*\n*** Please install the `nrtk` package. `nrtk-jatic` has been "
        "deprecated and functionality has been moved to `nrtk.interop.maite` (`nrtk` v0.20.0+).",
    )
else:
    warnings.warn(
        "`nrtk-jatic` has been deprecated and will fail to import on 2025/05/01. "
        "Please switch to using `nrtk.interop.maite` (`nrtk` v0.20.0+)",
        DeprecationWarning,
        stacklevel=2,
    )

    for module in ["interop", "api", "utils"]:
        nrtk_module = importlib.import_module(f"nrtk.interop.maite.{module}")
        sys.modules[f"nrtk_jatic.{module}"] = nrtk_module
        setattr(sys.modules["nrtk_jatic"], module, nrtk_module)
