"""Headless test of UVReady Lite, through its real operator and its real core.

  blender -b --python tests/test_lite.py -- <out_dir>          (tests/run_all_versions.sh runs it on 4.2, 4.5, 5.2)

The installed extension is tested (bl_ext.user_default.uvready_lite). When the paid UVReady is installed in the same
user folder, the coexistence checks run too (both add-ons on at once). UV quality is measured by metrics.py, which
does not use the add-on's code.
Prints one CHECK line per check and a final line `CASE lite <blender> PASS|FAIL n/m`.
"""
import addon_utils
import bmesh
import bpy
import importlib
import json
import math
import os
import sys
import traceback
import types

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import metrics  # noqa: E402

OUT = sys.argv[sys.argv.index("--") + 1]
os.makedirs(OUT, exist_ok=True)
LITE_ID = "uvready_lite"
MOD = "bl_ext.user_default." + LITE_ID
PAID = "bl_ext.user_default.uvready"
FULL_URL = "https://mendbox.gumroad.com/l/uvready"
VERSION = "%d.%d.%d" % bpy.app.version
results = []


def check(name, ok, note=""):
    results.append(bool(ok))
    print("CHECK %02d %-66s %s %s" % (len(results), name, "OK" if ok else "FAIL", note), flush=True)


def finish():
    print("CASE lite %s %s %d/%d" % (VERSION, "PASS" if all(results) else "FAIL", sum(results), len(results)), flush=True)
    sys.stdout.flush()
    os._exit(0 if all(results) else 1)


lite_at_start = addon_utils.check(MOD)[1]
paid_installed = os.path.isdir(os.path.join(bpy.utils.user_resource("EXTENSIONS"), "user_default", "uvready"))
both_at_start = lite_at_start and addon_utils.check(PAID)[1] and hasattr(bpy.types.Scene, "uvready_lite_result") \
    and hasattr(bpy.types.Scene, "uvready_settings")


def reset(paid=False):
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.preferences.addon_enable(module=MOD)
    if paid:
        bpy.ops.preferences.addon_enable(module=PAID)
    return sys.modules[MOD + ".core"]


def obj_from_bm(name, bm, loc=(0, 0, 0)):
    me = bpy.data.meshes.new(name)
    bm.to_mesh(me)
    bm.free()
    ob = bpy.data.objects.new(name, me)
    ob.location = loc
    bpy.context.scene.collection.objects.link(ob)
    return ob


def sphere(name="Sphere", sub=4, r=1.0, loc=(0, 0, 0), noise=0.15):
    bm = bmesh.new()
    bmesh.ops.create_icosphere(bm, subdivisions=sub, radius=r)
    for v in bm.verts:
        v.co *= 1 + noise * (math.sin(5 * v.co.x) * math.cos(4 * v.co.y) + 0.3 * math.sin(7 * v.co.z))
    return obj_from_bm(name, bm, loc)


def hardcube(name="Crate", size=2.0, loc=(0, 0, 0)):
    bm = bmesh.new()
    bmesh.ops.create_cube(bm, size=size)
    bmesh.ops.bevel(bm, geom=list(bm.edges), offset=0.12, segments=2, affect="EDGES")
    return obj_from_bm(name, bm, loc)


def suzanne(name="Suzanne", loc=(0, 0, 0)):
    bpy.ops.mesh.primitive_monkey_add(location=loc)
    ob = bpy.context.active_object
    ob.name = name
    return ob


def select(objs, active=None):
    if bpy.context.mode != 'OBJECT':
        bpy.ops.object.mode_set(mode='OBJECT')
    for o in bpy.context.view_layer.objects:
        o.select_set(False)
    for o in objs:
        o.select_set(True)
    bpy.context.view_layer.objects.active = active or objs[0]


def maps(ob):
    return [l.name for l in ob.data.uv_layers]


def uv_hash(ob):
    out = {}
    for l in ob.data.uv_layers:
        a = np.empty(len(ob.data.loops) * 2)
        l.data.foreach_get("uv", a)
        out[l.name] = round(float(a.sum()), 6)
    return out


def quality(ob, label, angle=None):
    m = metrics.compute(ob.data, 1024)
    a = np.empty(len(ob.data.loops) * 2)
    ob.data.uv_layers.active.data.foreach_get("uv", a)
    inside = m["uv_bbox_w"] <= 1.0001 and m["uv_bbox_h"] <= 1.0001 and a.min() >= -1e-6 and a.max() <= 1 + 1e-6
    ok = m["flipped_faces"] == 0 and m["overlap_pixel_share"] < 0.002 and inside and m["zero_uv_tris"] == 0
    if angle is not None:
        ok = ok and m["angle_mean"] < angle
    check("%s: 0 flips, no overlap, inside 0-1, no zero-area UV%s" % (label, "" if angle is None else ", angle < %g" % angle),
          ok, "flips %d overlap %.5f bbox %.3fx%.3f zero %d angle %.2f" % (
              m["flipped_faces"], m["overlap_pixel_share"], m["uv_bbox_w"], m["uv_bbox_h"], m["zero_uv_tris"], m["angle_mean"]))
    return m


def unwrap():
    """The Lite operator; background mode turns an ERROR report into RuntimeError."""
    try:
        return bpy.ops.uvready_lite.unwrap()
    except RuntimeError as ex:
        return {"CANCELLED"} if "Nothing to unwrap" in str(ex) else {"EXC: %s" % ex}


def result():
    return json.loads(bpy.context.scene.uvready_lite_result or "{}")


class Layout:
    """Records what a panel draws (background mode cannot draw panels)."""

    def __init__(self, log):
        self.log, self.enabled, self.scale_y = log, True, 1.0

    def _sub(self, *a, **k):
        return Layout(self.log)

    box = column = row = _sub

    def label(self, text="", icon="NONE"):
        self.log.append(("label", text))

    def prop(self, data, name, **k):
        self.log.append(("prop", name))

    def operator(self, idname, **k):
        self.log.append(("operator", idname))
        ref = types.SimpleNamespace()
        self.log.append(("ref", ref))
        return ref

    def separator(self, **k):
        pass


def draw_panel(ui, width=320):
    log = []
    panel = types.SimpleNamespace(layout=Layout(log))
    ctx = types.SimpleNamespace(mode=bpy.context.mode, scene=bpy.context.scene, view_layer=bpy.context.view_layer,
                                region=types.SimpleNamespace(width=width),
                                preferences=types.SimpleNamespace(system=types.SimpleNamespace(ui_scale=1.0)))
    ui.UVREADYLITE_PT_main.draw(panel, ctx)
    return log


def run_all():
    check("Lite enabled from the user folder at startup", lite_at_start)
    core = reset()
    const = sys.modules[MOD + ".const"]
    ops = sys.modules[MOD + ".ops"]
    ui = sys.modules[MOD + ".ui"]

    # ------------------------------------------------------------------------------------------ identity
    check("extension id, version, names", const.ADDON_ID == LITE_ID and const.VERSION == "1.0.0" and
          const.STORE_NAME == "UVReady Lite - One Click Auto UV Unwrap (Free)" and const.ADDON_NAME == "UVReady Lite"
          and const.TAB == "UVReady Lite", "%s %s %s" % (const.ADDON_ID, const.VERSION, const.STORE_NAME))
    man = open(os.path.join(os.path.dirname(const.__file__), "blender_manifest.toml"), encoding="utf-8").read()
    check("manifest: GPL-3.0-or-later, 4 platforms, website", 'license = ["SPDX:GPL-3.0-or-later"]' in man and
          'platforms = ["macos-arm64", "macos-x64", "linux-x64", "windows-x64"]' in man and
          "github.com/mendbox/uvready-lite" in man)
    check("own names: operator, result property, panel, tab", const.OPERATOR == "uvready_lite" and
          const.RESULT == "uvready_lite_result" and hasattr(bpy.ops.uvready_lite, "unwrap") and
          hasattr(bpy.types.Scene, "uvready_lite_result") and hasattr(bpy.types, "UVREADY_LITE_PT_main") and
          bpy.types.UVREADY_LITE_PT_main.bl_category == "UVReady Lite")
    check("no settings: no Scene settings property, no PropertyGroup", not hasattr(bpy.types.Scene, "uvready_lite_settings")
          and not hasattr(ops, "Settings") and ops.classes == (ops.UVREADYLITE_OT_unwrap,))
    check("Lite limits are the full version's defaults", (const.LITE_MARGIN, const.LITE_TEX, const.LITE_MAX_FACES) ==
          (4, 2048, 200000) and const.FULL_URL == FULL_URL and const.UV_NAME == "UVReady")
    check("operator undoable", "UNDO" in ops.UVREADYLITE_OT_unwrap.bl_options)

    # ------------------------------------------------------------------------------------------ panel
    ob = sphere()
    select([ob])
    log = draw_panel(ui)
    labels = " ".join(t for k, t in log if k == "label")
    props = [t for k, t in log if k == "prop"]
    opers = [t for k, t in log if k == "operator"]
    url = [r for (k, r) in log if k == "ref" and getattr(r, "url", None)]
    check("panel: Auto UV button, no settings to change", "uvready_lite.unwrap" in opers and props == [], str(props))
    check("panel: fixed settings line and the Lite note", "margin 4 px on 2048" in labels and
          "Lite unwraps one object per click" in labels and "texel density" in labels, labels[:240])
    check("panel: Get the full version opens the Gumroad page", "wm.url_open" in opers and url and url[0].url == FULL_URL)
    narrow = draw_panel(ui, width=200)
    check("panel: long lines wrapped in a narrow sidebar", all(len(t) <= 30 for k, t in narrow if k == "label" and t and not t.startswith("One object"))
          and len([1 for k, t in narrow if k == "label"]) > len([1 for k, t in log if k == "label"]))

    # ------------------------------------------------------------------------------------------ one object, operator
    ob.data.uv_layers.new(name="Orig")
    before = uv_hash(ob)
    bpy.ops.ed.undo_push(message="before")
    res = unwrap()
    check("operator: one sphere unwrapped", res == {"FINISHED"}, str(res))
    after = uv_hash(ob)
    check("new UV map 'UVReady' active, existing map kept", maps(ob) == ["Orig", "UVReady"] and
          ob.data.uv_layers.active.name == "UVReady" and after["Orig"] == before["Orig"], str(maps(ob)))
    rep = result()
    check("result line: 1 object, islands, stretch", rep.get("ok") and rep["head"].startswith("1 object,") and
          "islands" in rep["head"] and "stretch" in rep["head"], str(rep))
    quality(ob, "sphere")
    bpy.ops.ed.undo_push(message="after")
    bpy.ops.ed.undo()
    ob = bpy.data.objects["Sphere"]
    check("one undo restores the UV maps", uv_hash(ob) == before, str(uv_hash(ob)))

    # ------------------------------------------------------------------------------------------ quality
    core = reset()
    s = suzanne()
    select([s])
    unwrap()
    quality(s, "Suzanne")
    c = hardcube()
    select([c])
    unwrap()
    m = quality(c, "bevelled crate", angle=5.0)
    r = core.run(bpy.context)
    check("second run on the same object adds UVReady.001", maps(c) == ["UVReady", "UVReady.001"], str(maps(c)))

    # ------------------------------------------------------------------------------------------ the limits hold in core
    core = reset()
    ob = sphere("Opts", sub=3)
    ob.data.uv_layers.new(name="Only")
    select([ob])
    refused = []
    for kw in (dict(margin_px=40), dict(tex=512), dict(match_density=True), dict(align_axes=True), dict(replace=True),
               dict(max_faces=100)):
        try:
            core.run(bpy.context, **kw)
        except TypeError:
            refused.append(list(kw)[0])
    check("core.run takes no paid option; nothing unwrapped by the refused calls", len(refused) == 6 and
          maps(ob) == ["Only"], "%s %s" % (refused, maps(ob)))
    src = open(core.__file__, encoding="utf-8").read()
    check("paid code is not in the Lite core", not any(w in src for w in (
        "_pack_axis", "align_axes", "match_density", "max_faces", "margin_px", "const.LITE_")) and
        not hasattr(core, "_pack_axis"))
    r = core.run(bpy.context)
    ref = sphere("Ref", sub=3, loc=(4, 0, 0))
    select([ref])
    saved = (const.LITE_MARGIN, const.LITE_TEX, const.LITE_MAX_FACES)
    const.LITE_MARGIN, const.LITE_TEX, const.LITE_MAX_FACES = 40, 512, 100          # a script changing the constants
    try:
        r2 = core.run(bpy.context)
    finally:
        const.LITE_MARGIN, const.LITE_TEX, const.LITE_MAX_FACES = saved
    check("changed module constants change nothing: not skipped, same UVs as the defaults",
          r["objects"] == 1 and r2["objects"] == 1 and maps(ob) == ["Only", "UVReady"] and
          uv_hash(ob)["UVReady"] == uv_hash(ref)["UVReady"], "%s %s" % (uv_hash(ob)["UVReady"], uv_hash(ref)["UVReady"]))
    sc = bpy.context.scene
    sc["uvready_lite_settings"] = {"margin": 40, "max_faces": 2000000, "replace": True}     # scene data set by a script
    sc["uvready_settings"] = {"margin": 40, "replace": True}
    extra = sphere("Extra", sub=3, loc=(8, 0, 0))
    extra.data.uv_layers.new(name="Mine")
    select([extra])
    try:
        bad = bpy.ops.uvready_lite.unwrap(margin=40)
    except TypeError:
        bad = "refused"
    res = unwrap()
    op_props = [p.identifier for p in bpy.ops.uvready_lite.unwrap.get_rna_type().properties if p.identifier != "rna_type"]
    check("operator has no properties; scene data set by a script changes nothing", bad == "refused" and
          res == {"FINISHED"} and maps(extra) == ["Mine", "UVReady"] and
          uv_hash(extra)["UVReady"] == uv_hash(ref)["UVReady"] and
          op_props == [] and not getattr(ops.UVREADYLITE_OT_unwrap, "__annotations__", {}),
          "%s %s %s %s" % (bad, res, maps(extra), op_props))

    # ------------------------------------------------------------------------------------------ several selected
    core = reset()
    a = sphere("A", sub=3)
    b = hardcube("B", loc=(6, 0, 0))
    cc = suzanne("C", loc=(12, 0, 0))
    select([a, b, cc], active=b)
    res = unwrap()
    rep = result()
    lines = " ".join(rep.get("lines", []))
    check("three selected: only the active one unwrapped", res == {"FINISHED"} and maps(b) == ["UVReady"] and
          maps(a) == [] and maps(cc) == ["UVMap"], "%s %s %s" % (maps(a), maps(b), maps(cc)))
    check("three selected: message names the others and the full version", "Lite unwraps one object per click: B" in lines
          and "A, C" in lines and "full version" in lines, lines[:200])
    check("three selected: selection and active restored", all(o.select_get() for o in (a, b, cc)) and
          bpy.context.view_layer.objects.active is b)
    e = bpy.data.objects.new("Empty", None)
    bpy.context.scene.collection.objects.link(e)
    select([e, cc, a], active=e)
    r = core.run(bpy.context)
    check("active not a mesh: first mesh by name unwrapped, non-mesh named", r["objects"] == 1 and maps(a) == ["UVReady"]
          and maps(cc) == ["UVMap"] and any("Ignored, not a mesh: Empty" in x for x in r["messages"]), str(r["messages"]))

    # ------------------------------------------------------------------------------------------ edge cases
    select([e])
    check("only a non-mesh selected: cancelled", unwrap() == {"CANCELLED"})
    bm = bmesh.new()
    for i in range(5):
        bm.verts.new((i, 0, 0))
    nf = obj_from_bm("NoFaces", bm)
    select([nf])
    check("mesh without faces: cancelled, untouched", unwrap() == {"CANCELLED"} and len(nf.data.uv_layers) == 0)
    core = reset()
    ob = sphere("Mod", sub=3)
    m1 = ob.modifiers.new("Sub", "SUBSURF")
    m1.levels = 1
    ob.modifiers.new("Mir", "MIRROR")
    nv = len(ob.data.vertices)
    select([ob])
    unwrap()
    check("modifier stack and base mesh kept", [m.name for m in ob.modifiers] == ["Sub", "Mir"] and m1.levels == 1 and
          len(ob.data.vertices) == nv)
    full = sphere("Full", sub=2)
    for i in range(8):
        full.data.uv_layers.new(name="M%d" % i)
    select([full])
    r = core.run(bpy.context)
    check("8 UV maps: the active one replaced, with a message", len(full.data.uv_layers) == 8 and
          any("8 UV maps" in x for x in r["messages"]), str(r["messages"]))
    bpy.ops.mesh.primitive_grid_add(x_subdivisions=451, y_subdivisions=451, size=4)
    big = bpy.context.active_object
    big.name = "Big"
    select([big])
    const.LITE_MAX_FACES = 2000000
    try:
        r = core.run(bpy.context)
    finally:
        const.LITE_MAX_FACES = 200000
    check("over 200,000 faces: skipped (also with the constant raised), message names the full version", r["objects"] == 0 and len(big.data.uv_layers) <= 1
          and any("over the Lite limit of 200000" in x and "full version" in x for x in r["messages"]),
          "%d faces %s" % (len(big.data.polygons), r["messages"]))

    # ------------------------------------------------------------------------------------------ coexistence
    if paid_installed:
        check("both enabled from the user folder at startup", both_at_start)
        core = reset(paid=True)
        check("both on: both result properties, both operators, both panels in their own tabs",
              hasattr(bpy.types.Scene, "uvready_settings") and hasattr(bpy.types.Scene, "uvready_lite_result") and
              hasattr(bpy.ops.uvready, "unwrap") and hasattr(bpy.ops.uvready_lite, "unwrap") and
              hasattr(bpy.types, "UVREADY_PT_main") and hasattr(bpy.types, "UVREADY_LITE_PT_main") and
              bpy.types.UVREADY_PT_main.bl_category != bpy.types.UVREADY_LITE_PT_main.bl_category)
        check("both on: the paid settings are there with their defaults", bpy.context.scene.uvready_settings.margin == 4
              and bpy.context.scene.uvready_settings.match_density)
        a = sphere("PA", sub=3)
        b = hardcube("PB", loc=(6, 0, 0))
        select([a, b])
        res = bpy.ops.uvready.unwrap()
        prep = json.loads(bpy.context.scene.uvready_result or "{}")
        check("both on: the paid operator unwraps both objects", res == {"FINISHED"} and maps(a) == ["UVReady"] and
              maps(b) == ["UVReady"] and prep.get("head", "").startswith("2 objects"), str(prep))
        x = sphere("LX", sub=3, loc=(0, 6, 0))
        select([x])
        res = unwrap()
        check("both on: Lite works, its result separate from the paid one", res == {"FINISHED"} and
              result()["head"].startswith("1 object,") and
              json.loads(bpy.context.scene.uvready_result)["head"].startswith("2 objects"))
        bpy.ops.preferences.addon_disable(module=MOD)
        check("Lite switched off: the paid add-on untouched", not hasattr(bpy.types.Scene, "uvready_lite_result") and
              hasattr(bpy.types.Scene, "uvready_settings") and hasattr(bpy.types, "UVREADY_PT_main"))
        bpy.ops.preferences.addon_enable(module=MOD)
        check("Lite switched on again", hasattr(bpy.types.Scene, "uvready_lite_result") and
              hasattr(bpy.types, "UVREADY_LITE_PT_main"))
    else:
        print("NOTE paid UVReady not installed: coexistence checks skipped", flush=True)


try:
    run_all()
except Exception:
    traceback.print_exc()
    check("no exception", False)
finish()
