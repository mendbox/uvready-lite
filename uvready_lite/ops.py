# SPDX-License-Identifier: GPL-3.0-or-later
import json

import bpy
from bpy.types import Operator

from . import const, core


class UVREADYLITE_OT_unwrap(Operator):
    bl_idname = const.OPERATOR + ".unwrap"
    bl_label = const.ACTION
    bl_description = ("Unwrap the active mesh object: seams, islands, packing, in one step "
                      "(Lite: one object per click, default settings)")
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        return context.mode == 'OBJECT' and bool(context.selected_objects)

    def execute(self, context):
        wm = context.window_manager
        wm.progress_begin(0, 100)
        try:
            res = core.run(context, progress=lambda n, k: wm.progress_update(int(100 * n / max(k, 1))))
        finally:
            wm.progress_end()
        lines = list(res["messages"])
        if res["objects"]:
            head = "%d object%s, %d islands, stretch %s (%.1f%%)" % (
                res["objects"], "" if res["objects"] == 1 else "s", res["islands"], res["label"], res["stretch_pct"])
        else:
            head = "Nothing to unwrap."
        setattr(context.scene, const.RESULT, json.dumps(dict(head=head, lines=lines, ok=bool(res["objects"]))))
        for m in lines:
            self.report({'WARNING'}, m)
        self.report({'INFO'} if res["objects"] else {'ERROR'}, head)
        return {'FINISHED'} if res["objects"] else {'CANCELLED'}


classes = (UVREADYLITE_OT_unwrap,)


def register_props():
    setattr(bpy.types.Scene, const.RESULT, bpy.props.StringProperty())


def unregister_props():
    delattr(bpy.types.Scene, const.RESULT)
