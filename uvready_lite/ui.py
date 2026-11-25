# SPDX-License-Identifier: GPL-3.0-or-later
"""Sidebar tab (3D Viewport > N panel), Object Mode."""
import json
import textwrap

from bpy.types import Panel

from . import const


def wrapped(layout, context, text, icon="NONE", inset=70):
    """A label cuts long text with "..."; this breaks it into lines that fit the sidebar. Icon on the first line.
    `inset` = pixels (at scale 1) the layout loses to margins, icon and boxes."""
    scale = context.preferences.system.ui_scale
    chars = max(16, int((context.region.width - inset * scale) / (6.5 * scale)))
    for n, line in enumerate(textwrap.wrap(text, chars) or [""]):
        layout.label(text=line, icon=icon if n == 0 else "BLANK1")


class UVREADYLITE_PT_main(Panel):
    bl_label = const.ADDON_NAME
    bl_idname = const.CLASS_PREFIX + "_PT_main"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = const.TAB

    def draw(self, context):
        col = self.layout.column()
        if context.mode != 'OBJECT':
            col.label(text="Switch to Object Mode", icon='INFO')
        col.operator(const.OPERATOR + ".unwrap", icon='UV')
        col.label(text="One object, margin %d px on %d, new UV map" % (const.LITE_MARGIN, const.LITE_TEX),
                  icon="SETTINGS")
        box = self.layout.box()
        wrapped(box, context, const.LITE_LINE, "INFO", inset=100)
        box.operator("wm.url_open", text="Get the full version", icon="URL").url = const.FULL_URL
        raw = getattr(context.scene, const.RESULT, "")
        try:
            rep = json.loads(raw) if raw else None
        except ValueError:
            rep = None
        if isinstance(rep, dict):
            b = self.layout.box()
            b.label(text=str(rep.get("head", "")), icon='CHECKMARK' if rep.get("ok") else 'ERROR')
            for line in rep.get("lines", []):
                wrapped(b, context, str(line), inset=100)


classes = (UVREADYLITE_PT_main,)
