# SPDX-License-Identifier: GPL-3.0-or-later
# SHARED WITH ASSETBAKE: copied by tools/sync_shared.py from aiasset/ai_asset_maker/pipeline/autouv.py - do not edit here.
"""Automatic UV unwrap of the low mesh.

    stats = autouv.auto_uv(obj, tex=2048, gap_px=8)

Steps:
  1. face normals smoothed with a bilateral filter (6 passes, sigma 40 degrees)
  2. hard surface (uvlab.Hard): in the shells that are hard surface, creases become chart borders, a flat or
     developable region (flat face, cylinder or cone wall) becomes one chart as it is, large flat patches are
     never cut and never share a chart with a vertex that cannot be flattened. Organic shells are not touched.
  3. charts: bottom-up face clustering (uvlab.cluster_hier), border smoothing, small charts merged
  4. seams on chart borders, then every island gets the best of: the ANGLE_BASED solver, the MINIMUM_STRETCH
     solver (Blender 4.3+; Blender 4.2 has ANGLE_BASED only), its exact rigid unfolding when it is developable
     (uvlab.unfold), and the projection on its mean plane when the solvers fail on it (uvlab.project)
  5. every island is checked (angle and area distortion, flipped faces, self-overlap, length); bad ones are
     split and unwrapped again, at most `rounds` times
  6. inverted faces are turned back (uvlab.untangle); what cannot be turned back becomes its own island
  7. average island scale, pack (concave shapes, fixed pixel gap; the best of several kinds of rotation)

The winding of the mesh is taken as it is: the caller makes the normals consistent before.
"""
import time

import bpy
import numpy as np

from . import uvlab as L


def unwrap_method():
    """The second solver next to ANGLE_BASED: MINIMUM_STRETCH where Blender has it (4.3+). Blender 4.2 does not,
    and every island is solved with ANGLE_BASED alone there."""
    methods = [i.identifier for i in bpy.ops.uv.unwrap.get_rna_type().properties["method"].enum_items]
    return "MINIMUM_STRETCH" if "MINIMUM_STRETCH" in methods else "ANGLE_BASED"

DEFAULTS = dict(
    smooth_iters=6, smooth_sigma=40.0,   # bilateral normal smoothing
    rmin=0.55,                           # mean resultant length of RAW normals a chart must keep (no closed tubes)
    cone=160.0,                          # conservative cone bound on smoothed normals, degrees
    w_shape=0.2,                         # compactness weight in the merge cost
    border_passes=3, min_share=0.004,    # border smoothing; charts under 0.4 % of the area are merged away
    max_angle=1.3, max_area=0.35,        # island limits: mean sigma1 / sigma2, mean |log2 area ratio|
    split_min_share=0.003,               # islands under 0.3 % of the area are split only when broken
    overlap_share=0.004,                 # self-overlap above 0.4 % of the island's pixels counts as broken
    rounds=6,
    # hard surface (uvlab.Hard)
    hard=True,
    sharp=40.0,                          # a crease of this many degrees is a chart border in a hard-surface shell
    shell_share=0.33,                    # a shell is hard surface with this share of its area in whole regions and rigid patches
    flat_tol=1.5, flat_share=0.002, flat_shell_share=0.04,   # rigid patch: coplanar within 1.5 degrees, 0.2 % of the mesh or 4 % of its shell
    cone_k=0.2,                          # angle defect (radians) that makes a vertex a cone
    region_k=0.15, disk_sum=1.5, tube_sum=4.0,   # a region is taken whole: no vertex above region_k, total defect under these
    k_merge=0.05,                        # a small chart crosses a crease when the vertices taken in carry less defect than this
    speck_share=0.0002, speck_faces=3,   # a crumb of one or two faces under 0.02 % of the area joins a neighbour anyway
    # solvers and direct constructions
    abf_area=0.005,                      # ANGLE_BASED may cost an island this much area distortion (mean |log2|)
    flip_slack=0.003,                    # flipped area a direct construction may add (needle faces, repaired at the end)
    direct_gain=0.01,                    # a projected island must be this much less distorted than the solver's
    # packing
    pack_tries=3, pack_tries_faces=100000,   # rotation kinds tried (one on a mesh with more faces than this)
    pack_any_gain=1.08,                  # what free rotation must gain to be taken for a hard-surface mesh
)


def _edit():
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.select_all(action='SELECT')


def _unwrap(method):
    kw = dict(method=method, fill_holes=False, correct_aspect=False, margin_method='SCALED', margin=0.001)
    if method == 'MINIMUM_STRETCH':
        kw['iterations'] = 10
    bpy.ops.uv.unwrap(**kw)


def unwrap_collapsed(me, T):
    """True when the unwrap gave one island that is a line or a point and as long as the whole layout. The
    MINIMUM_STRETCH solver does that to a chart that is pinched at a vertex (TRELLIS meshes have such charts);
    the unwrap operator then scales every other island down to nothing next to it."""
    uv = L._uv_array(me)
    if not len(uv):
        return False
    isl, _ = L.uv_islands(T, uv)
    n = int(isl.max()) + 1
    corners = T.ls[:, None] + np.arange(3)
    tri = uv[corners]
    d1, d2 = tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]
    area = np.bincount(isl, weights=0.5 * np.abs(d1[:, 0] * d2[:, 1] - d1[:, 1] * d2[:, 0]), minlength=n)
    lo, hi = np.full((n, 2), np.inf), np.full((n, 2), -np.inf)
    which = np.repeat(isl, 3)
    np.minimum.at(lo, which, tri.reshape(-1, 2))
    np.maximum.at(hi, which, tri.reshape(-1, 2))
    extent = (hi - lo).max(1)
    whole = float((uv.max(0) - uv.min(0)).max())
    return bool(((area < 1e-7 * extent * extent) & (extent > 0.5 * whole)).any()) and n > 1


def unwrap_best(me, T, state, p):
    """Unwrap with the seams that are set, every island with the better of two solvers.
    ANGLE_BASED keeps angles and is exact on a chart that can be flattened (flat faces, cylinder and cone
    walls); MINIMUM_STRETCH (Blender 4.3+) keeps areas better on a curved chart but shears flat faces a little.
    An island takes the ANGLE_BASED result when that has no more flipped area, no more angle distortion, at most
    `abf_area` more area distortion (mean |log2|) and no more triangles that are off by 2x in area than the
    other. Returns the share of the area that took ANGLE_BASED. state['ms'] turns False for the rest of the mesh
    when MINIMUM_STRETCH collapses an island; state['islands'] gets (island per face, cut per pair) or None."""
    state['islands'] = None
    _edit()
    _unwrap('ANGLE_BASED')
    bpy.ops.object.mode_set(mode='OBJECT')
    if not state['ms']:
        return 1.0
    uv_b = L._uv_array(me)
    _edit()
    _unwrap('MINIMUM_STRETCH')
    bpy.ops.object.mode_set(mode='OBJECT')
    uv_a = L._uv_array(me)
    if unwrap_collapsed(me, T):
        state['ms'] = False                    # TRELLIS meshes have charts that this solver pinches to a line
        me.uv_layers.active.data.foreach_set('uv', uv_b.ravel())
        return 1.0
    cut_a, cut_b = (L.uv_cut(T, uv_a), L.uv_cut(T, uv_b))
    if not np.array_equal(cut_a[1], cut_b[1]):
        return 0.0                             # the two do not agree on the islands: keep MINIMUM_STRETCH
    islands = L.uv_islands(T, uv_a, cut=cut_a)
    state['islands'] = islands
    ra = L.island_report(L.metrics(me, raster=0, T=T, uv=uv_a, islands=islands)[1])
    rb = L.island_report(L.metrics(me, raster=0, T=T, uv=uv_b, islands=islands)[1])
    pick = (rb['flip'] <= ra['flip']) & (rb['angle'] <= ra['angle']) & (rb['area'] <= ra['area'] + p['abf_area']) & (rb['far'] <= ra['far']) & (rb['uvarea'] > 0)
    pick |= (ra['uvarea'] <= 0) & (rb['uvarea'] > 0)
    pick = pick[islands[0]]
    if pick.any():
        uv = np.where(pick[T.lp][:, None], uv_b, uv_a)
        me.uv_layers.active.data.foreach_set('uv', uv.ravel())
    return float(T.fa[pick].sum() / max(T.fa.sum(), 1e-30))


def _cost(rep):
    """One number per island for 'how distorted': log2 of the angle measure plus the area measure."""
    return np.log2(np.maximum(rep['angle'], 1)) + rep['area']


def improve_islands(me, T, hard, p, state):
    """Replaces the solver's result on the islands where a direct construction is better:
      - in the hard-surface shells, an island that is developable gets its exact rigid unfolding (uvlab.unfold),
        and one that is close to developable gets it when that is less distorted;
      - any island gets the projection on its mean plane (uvlab.project) when that is less distorted by at least
        `direct_gain`. This saves the nearly flat islands of CAD meshes, where needle faces defeat both solvers.
    A replacement has no more flipped area than what it replaces, give or take `flip_slack` of the island (needle
    faces, which uvlab.untangle turns back at the end). Returns the area share replaced."""
    uv = L._uv_array(me)
    if state.get('islands') is None:
        state['islands'] = L.uv_islands(T, uv)
    islands = state['islands'] + (L.uv_cut(T, uv)[0],)
    isl = islands[0]
    ni = int(isl.max()) + 1
    rep = L.island_report(L.metrics(me, raster=0, T=T, uv=uv, islands=islands)[1])
    cost = _cost(rep)
    flip = rep['flip']
    done = np.zeros(ni, bool)
    if hard is not None and hard.hard_face.any():
        new, exact, cand = L.unfold(T, uv, islands, hard.hard_face)
        take = exact
        if cand.any():
            r = L.island_report(L.metrics(me, raster=0, T=T, uv=new, islands=islands)[1])
            c = _cost(r)
            ok = cand & (r['flip'] <= flip + p['flip_slack']) & (c < cost) & (r['uvarea'] > 0)
            cost = np.where(ok, c, cost)
            flip = np.where(ok, r['flip'], flip)
            take = take | ok
        if take.any():
            uv = np.where(take[isl][T.lp][:, None], new, uv)
            cost = np.where(exact, 0.0, cost)
            flip = np.where(exact, 0.0, flip)
            done |= take
    new = L.project(T, uv, islands)
    r = L.island_report(L.metrics(me, raster=0, T=T, uv=new, islands=islands)[1])
    ok = (r['flip'] <= flip + p['flip_slack']) & (_cost(r) < cost - p['direct_gain']) & (r['uvarea'] > 0)
    if ok.any():
        uv = np.where(ok[isl][T.lp][:, None], new, uv)
        done |= ok
    if not done.any():
        return 0.0
    me.uv_layers.active.data.foreach_set('uv', uv.ravel())
    return float(T.fa[done[isl]].sum() / max(T.fa.sum(), 1e-30))


def auto_uv(ob, tex=2048, gap_px=8, method=None, pack=True, uv_name='UVMap', **over):
    """Unwrap and pack `ob` (active, object mode). Returns a dict with counts and timings:
    method (the solver used next to ANGLE_BASED, or ANGLE_BASED alone), faces, charts (before any split), rounds,
    splits, abf_share and direct_share (area that took ANGLE_BASED / a direct construction in the last round),
    flips_left (0 unless a fold could not be repaired), pack (rotation kind taken), hard_share (area in
    hard-surface shells), tubes (regions cut open as tubes), tm (seconds per step)."""
    p = dict(DEFAULTS)
    p.update(over)
    method = method or unwrap_method()
    tm = {}
    t0 = t = time.perf_counter()
    me = ob.data
    for o in bpy.context.selected_objects:
        o.select_set(False)
    bpy.context.view_layer.objects.active = ob
    ob.select_set(True)
    if uv_name not in me.uv_layers:
        me.uv_layers.new(name=uv_name)
    me.uv_layers.active = me.uv_layers[uv_name]
    me.uv_layers[uv_name].active_render = True
    T = L.Topo(me)
    sn = T.smoothed_normals(p['smooth_iters'], p['smooth_sigma'])
    hard = L.Hard(T, p['flat_tol'], p['flat_share'], p['flat_shell_share'], p['sharp'], p['cone_k'], p['region_k'], p['disk_sum'], p['tube_sum'], p['shell_share']) if p['hard'] else None
    if hard is not None and not hard.any:
        hard = None
    start = hard.start_labels() if hard else None
    atoms = hard.atom_labels() if hard else None
    cuts = hard.seams if hard else None
    labels = L.cluster_hier(T, sn, rmin=p['rmin'], w_shape=p['w_shape'], cone=p['cone'], labels=start, hard=hard)
    labels, _ = L.smooth_boundaries(T, labels, p['border_passes'], hard=hard)
    labels = L.merge_small(T, sn, labels, min_share=p['min_share'], hard=hard, k_merge=p['k_merge'], speck_share=p['speck_share'], speck_faces=p['speck_faces'])
    tm['charts'] = time.perf_counter() - t
    charts0 = int(len(np.unique(labels)))
    t = time.perf_counter()
    splits = 0
    rounds = 0
    state = dict(ms=method == 'MINIMUM_STRETCH')
    abf_share = 1.0
    for rnd in range(p['rounds'] + 1):
        L.labels_to_seams(T, labels, cuts)
        abf_share = unwrap_best(me, T, state, p)
        direct_share = improve_islands(me, T, hard, p, state)
        rounds = rnd
        if rnd == p['rounds']:
            break
        m, ex = L.metrics(me, raster=0, T=T, islands=state['islands'])
        rep = L.island_report(ex)
        isl = ex['isl']
        cnt = np.bincount(isl)
        cross = L.boundary_crossings(T, L._uv_array(me), isl, ex['cut'])
        for i in np.nonzero(cross > 0)[0].tolist():
            if L.island_overlap_share(ex['td']['Q'][ex['ti'] == i]) < p['overlap_share']:
                cross[i] = 0
        big = rep['area3'] >= p['split_min_share'] * rep['area3'].sum()
        # an island longer than the texture is wide (at the scale all islands share) is cut in two
        ext = L.island_extent(L._uv_array(me), T, isl, len(cnt)) * np.sqrt(rep['area3'] / np.maximum(rep['uvarea'], 1e-30))
        long_ = ext > L.long_side(T)
        severe = (rep['angle'] > 2 * p['max_angle'] - 1) | (rep['area'] > 2 * p['max_area']) | (rep['flip'] > 0.1) | (rep['uvarea'] <= 0) | (cross > 0)
        mild = (rep['angle'] > p['max_angle']) | (rep['area'] > p['max_area']) | (rep['flip'] > 0.03)
        if rnd >= p['rounds'] - 2:
            mild[:] = False
        bad = np.nonzero((mild & big | severe) & (cnt >= 24) | long_ & (cnt >= 2))[0]
        if not len(bad):
            break
        labels = isl.astype(np.int64).copy()
        nxt = int(labels.max()) + 1
        done = 0
        for i in bad.tolist():
            mask = isl == i
            target = 4 if severe[i] and cnt[i] >= 96 else 2
            sub = L.cluster_hier(T, sn, w_shape=p['w_shape'], target=target, restrict=mask, labels=atoms)
            parts = np.unique(sub[mask]).tolist()
            if len(parts) < 2 and atoms is not None:     # one rigid patch: it has to be cut after all
                sub = L.cluster_hier(T, sn, w_shape=p['w_shape'], target=target, restrict=mask)
                parts = np.unique(sub[mask]).tolist()
            for x in parts[1:]:
                labels[mask & (sub == x)] = nxt
                nxt += 1
            done += len(parts) > 1
        splits += done
        if not done:
            break
    # faces that are still inverted after the repair (a real fold, not a needle) become islands of their own
    for _ in range(2):
        uv, still = L.untangle(T, L._uv_array(me))
        me.uv_layers.active.data.foreach_set('uv', uv.ravel())
        if not still.any():
            break
        isl = L.uv_islands(T, L._uv_array(me))[0].astype(np.int64)
        k = still[T.pf0] & still[T.pf1] & (isl[T.pf0] == isl[T.pf1])
        group = L._components_pairs(T.nf, T.pf0[k], T.pf1[k])
        labels = np.where(still, int(isl.max()) + 1 + group, isl)
        L.labels_to_seams(T, labels, cuts)
        unwrap_best(me, T, state, p)
        improve_islands(me, T, hard, p, state)
        splits += 1
    tm['unwrap_and_fix'] = time.perf_counter() - t
    t = time.perf_counter()
    pack_how = ''
    if pack:
        # Blender's packer with several kinds of rotation; the layout that leaves the islands largest wins.
        # Hard-surface islands are axis-aligned already: they are packed unrotated or turned by 90 degrees, and
        # free rotation is taken only when it gains a lot (`pack_any_gain`).
        _edit()
        bpy.ops.uv.select_all(action='SELECT')
        bpy.ops.uv.average_islands_scale()
        bpy.ops.object.mode_set(mode='OBJECT')
        scaled = L._uv_array(me)
        if hard is not None and hard.hard_share >= 0.5:
            kinds = [('NONE', 1.0), ('CARDINAL', 1.005), ('ANY', p['pack_any_gain'])]
        else:
            kinds = [('ANY', 1.0), ('CARDINAL', 1.005)]
        best = None
        for how, gain in kinds[:p['pack_tries'] if T.nf <= p['pack_tries_faces'] else 1]:
            me.uv_layers.active.data.foreach_set('uv', scaled.ravel())
            _edit()
            bpy.ops.uv.select_all(action='SELECT')
            kw = dict(rotate=False) if how == 'NONE' else dict(rotate=True, rotate_method=how)
            bpy.ops.uv.pack_islands(shape_method='CONCAVE', margin_method='ADD', margin=gap_px / (2.0 * tex), **kw)
            bpy.ops.object.mode_set(mode='OBJECT')
            uv = L._uv_array(me)
            area = float(L.tri_data(T, uv)['uva'].sum())
            if best is None or area > best[0] * gain:
                best = (area, uv, how)
        uv = best[1]
        pack_how = best[2]
        me.uv_layers.active.data.foreach_set('uv', uv.ravel())
    tm['pack'] = time.perf_counter() - t
    t = time.perf_counter()
    uv, still = L.untangle(T, L._uv_array(me))           # last: the packer's scale must not bring a flip back
    me.uv_layers.active.data.foreach_set('uv', uv.ravel())
    flips_left = int(still.sum())
    tm['untangle'] = time.perf_counter() - t
    tm['total'] = time.perf_counter() - t0
    stats = dict(method=method if state['ms'] else 'ANGLE_BASED', faces=T.nf, charts=charts0, rounds=rounds, splits=splits,
                 abf_share=round(abf_share, 3), direct_share=round(direct_share, 3), flips_left=flips_left, pack=pack_how,
                 hard_share=round(hard.hard_share, 3) if hard else 0.0, tubes=hard.tubes if hard else 0)
    stats['tm'] = {k: round(v, 2) for k, v in tm.items()}
    return stats
