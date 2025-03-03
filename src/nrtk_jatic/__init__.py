"""Define the nrtk package"""

import importlib
import importlib.metadata
import sys
import warnings

__version__ = importlib.metadata.version(__name__)

warnings.warn(
    "nrtk-jatic has been deprecated and will soon be scheduled for deletion. "
    "Please switch to using nrtk.interop.maite",
    DeprecationWarning,
    stacklevel=2,
)

for module in ["interop", "api", "utils"]:
    nrtk_module = importlib.import_module(f"nrtk.interop.maite.{module}")
    sys.modules[f"nrtk_jatic.{module}"] = nrtk_module
    setattr(sys.modules["nrtk_jatic"], module, nrtk_module)
