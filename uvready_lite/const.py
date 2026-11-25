# SPDX-License-Identifier: GPL-3.0-or-later
"""Names in one place. Product name and id come from blender_manifest.toml."""
import os
import re
import tomllib

HERE = os.path.dirname(os.path.abspath(__file__))


def _manifest():
    try:
        with open(os.path.join(HERE, "blender_manifest.toml"), "rb") as fh:
            return tomllib.load(fh)
    except (OSError, tomllib.TOMLDecodeError):
        return {}


_M = _manifest()
ADDON_ID = _M.get("id", "uvready_lite")
STORE_NAME = _M.get("name", "UVReady Lite - One Click Auto UV Unwrap (Free)")     # long keyword name (manifest); never shown in the UI
ADDON_NAME = "UVReady Lite"                     # short name: sidebar tab, panel title, messages
VERSION = _M.get("version", "?")
_KEY = re.sub(r"[^a-z0-9_]", "_", ADDON_ID.lower())
TAB = ADDON_NAME
ACTION = "Auto UV"                        # the button and the one undo step
SETTINGS = _KEY + "_settings"             # bpy.types.Scene.<SETTINGS>
RESULT = _KEY + "_result"                 # bpy.types.Scene.<RESULT>: JSON of the last run
OPERATOR = _KEY                           # bpy.ops.<OPERATOR>.unwrap
CLASS_PREFIX = _KEY.upper()
UV_NAME = "UVReady"                        # name of the UV map created in "new map" mode
MAX_UV_MAPS = 8                           # Blender's limit

# Lite (set by tools/build_lite.py of the full version; core.run enforces them whatever the caller passes)
LITE_MARGIN = 4                          # px
LITE_TEX = 2048                          # texture size the margin is measured on
LITE_MAX_FACES = 200000                    # objects over this are skipped
FULL_URL = "https://mendbox.gumroad.com/l/uvready"
LITE_LINE = ("Lite unwraps one object per click with the default settings. The full version unwraps all selected "
             "objects at once with one texel density, and has margin, texture size, rotate-to-axes, replace-UV-map "
             "and face-limit settings.")
