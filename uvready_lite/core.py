# SPDX-License-Identifier: GPL-3.0-or-later
"""The work, without any UI: unwrap the given mesh objects with the shared AssetBake auto-UV code.

    results = run(context)          # Lite: one object, the full version's default settings

Runs synchronously inside the operator (one undo step). Blender's own UI is blocked meanwhile: that is why there
is a face limit per object (about 13-25 s at 150k triangles, 4 minutes at 875k) instead of a worker process."""
import math

import bpy
import numpy as np

from . import const
from .pipeline import autouv, uvlab as L


def stretch_label(angle_mean):
    """angle_mean is the area-weighted mean of sigma1/sigma2 (1.0 = no shear). Returns (percent over 1, word)."""
    pct = (angle_mean - 1.0) * 100.0
    return pct, ("low" if pct < 15 else "medium" if pct < 35 else "high")  # calibrated 2026-11-19: sphere/Suzanne unwraps measure 10-13 %


def world_scale(ob):
    """Area scale factor of the object's transform and whether it is non-uniform."""
    s = ob.matrix_world.to_scale()
    det = abs(s.x * s.y * s.z)
    area = det ** (2.0 / 3.0) if det > 0 else 0.0
    uniform = max(s) - min(s) <= 0.01 * max(max(s), 1e-12)
    return area, uniform


def _free_name(me, base):
    if base not in me.uv_layers:
        return base
    i = 1
    while "%s.%03d" % (base, i) in me.uv_layers:
        i += 1
    return "%s.%03d" % (base, i)


def run(context, progress=None):
    """Returns dict(done=[per object dict], messages=[str], islands, stretch_pct, label, flips).
    Objects other than meshes, meshes without faces, meshes over 200000 faces, library data and duplicates of one
    mesh datablock are skipped with a message.
    Lite: no options (the full version's defaults are written in: 4 px margin on 2048), and only ONE mesh is
    unwrapped (the active one when it qualifies, else the first by name); the others are named in a message."""
    msgs = []
    sel = list(context.selected_objects)
    active = context.view_layer.objects.active
    targets, seen = [], set()
    notmesh = [o.name for o in sel if o.type != 'MESH']
    if notmesh:
        msgs.append("Ignored, not a mesh: " + ", ".join(notmesh))
    for ob in sel:
        if ob.type != 'MESH':
            continue
        me = ob.data
        if me.library or ob.library:
            msgs.append("Skipped %s: linked from a library." % ob.name)
        elif not len(me.polygons):
            msgs.append("Skipped %s: the mesh has no faces." % ob.name)
        elif len(me.polygons) > 200000:
            msgs.append("Skipped %s: %d faces is over the Lite limit of %d (decimate first; the full version "
                        "lets you raise the limit)."
                        % (ob.name, len(me.polygons), 200000))
        elif me.as_pointer() in seen:
            msgs.append("%s shares its mesh with another selected object: unwrapped once." % ob.name)
        else:
            seen.add(me.as_pointer())
            targets.append(ob)
    if len(targets) > 1:
        keep = active if active in targets else min(targets, key=lambda o: o.name)
        others = sorted(o.name for o in targets if o is not keep)
        targets = [keep]
        msgs.append("Lite unwraps one object per click: %s. Not unwrapped: %s. The full version unwraps all selected "
                    "objects at once." % (keep.name, ", ".join(others)))
    done = []
    try:
        for n, ob in enumerate(targets):
            if progress:
                progress(n, len(targets))
            me = ob.data
            _, uniform = world_scale(ob)
            if not uniform:
                msgs.append("%s has a non-uniform scale: apply it (Ctrl+A) or the texture will look stretched." % ob.name)
            if len(me.uv_layers) >= const.MAX_UV_MAPS:
                name = me.uv_layers.active.name
                msgs.append("%s has %d UV maps already: the active one was replaced." % (ob.name, const.MAX_UV_MAPS))
            else:
                name = _free_name(me, const.UV_NAME)
            st = autouv.auto_uv(ob, tex=2048, gap_px=4, uv_name=name, pack=True)
            done.append(dict(ob=ob, uv=name, stats=st))
    finally:
        for o in context.selected_objects:
            o.select_set(False)
        for o in sel:
            if o.name in context.view_layer.objects:
                o.select_set(True)
        context.view_layer.objects.active = active
    # measure
    infos = []
    for d in done:
        me = d["ob"].data
        m, ex = L.metrics(me, raster=0)
        a3 = float(ex["T"].fa.sum()) * (world_scale(d["ob"])[0] or 1.0)
        dens = math.sqrt(m["fill_sum"] / a3) if a3 > 0 and m["fill_sum"] > 0 else 0.0
        infos.append((m, dens))
    islands, flips, wsum, awsum = 0, 0, 0.0, 0.0
    for d, (m, _) in zip(done, infos):
        d["islands"] = m["islands"]
        d["angle"] = m["angle_mean"]
        d["flipped_share"] = m["flipped_share"]
        d["flips_left"] = d["stats"]["flips_left"]
        islands += m["islands"]
        flips += d["flips_left"]
        wsum += d["stats"]["faces"]
        awsum += m["angle_mean"] * d["stats"]["faces"]
    angle = awsum / wsum if wsum else 1.0
    pct, label = stretch_label(angle)
    if flips:
        msgs.append("%d faces could not be unfolded without a fold-over." % flips)
    return dict(done=done, messages=msgs, objects=len(done), islands=islands, stretch_pct=pct, label=label, flips=flips,
                method=done[0]["stats"]["method"] if done else "")
