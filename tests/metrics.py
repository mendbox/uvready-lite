"""Independent UV metrics (bpy + numpy only, nothing from the add-on). All on the ACTIVE uv layer.
Per triangle (loop_triangles): angle deviation, area ratio, flip. Per polygon loops: seams and islands."""
import numpy as np, bpy, math

def _arrays(me):
    nl, nt = len(me.loops), len(me.loop_triangles)
    co = np.empty(len(me.vertices) * 3); me.vertices.foreach_get("co", co); co = co.reshape(-1, 3)
    lv = np.empty(nl, np.int32); me.loops.foreach_get("vertex_index", lv)
    uv = np.empty(nl * 2); me.uv_layers.active.data.foreach_get("uv", uv); uv = uv.reshape(-1, 2)
    tl = np.empty(nt * 3, np.int32); me.loop_triangles.foreach_get("loops", tl); tl = tl.reshape(-1, 3)
    tp = np.empty(nt, np.int32); me.loop_triangles.foreach_get("polygon_index", tp)
    ls = np.empty(len(me.polygons), np.int32); me.polygons.foreach_get("loop_start", ls)
    tot = np.empty(len(me.polygons), np.int32); me.polygons.foreach_get("loop_total", tot)
    le = np.empty(nl, np.int32); me.loops.foreach_get("edge_index", le)
    return co, lv, uv, tl, tp, ls, tot, le

def _angles(P):
    """P (n,3,d) corner positions -> (n,3) interior angles in degrees."""
    out = np.empty(P.shape[:2])
    for i in range(3):
        a, b, c = P[:, i], P[:, (i + 1) % 3], P[:, (i + 2) % 3]
        u, v = b - a, c - a
        nu, nv = np.linalg.norm(u, axis=1), np.linalg.norm(v, axis=1)
        d = (u * v).sum(1) / np.maximum(nu * nv, 1e-30)
        out[:, i] = np.degrees(np.arccos(np.clip(d, -1, 1)))
    return out

def rasterize(U, flip, n=1024):
    """Pixel-centre rasterisation of UV triangles U (m,3,2) in [0,1]^2 on an n x n grid. Returns (count, flipcount)."""
    cnt = np.zeros((n, n), np.int16); fl = np.zeros((n, n), np.int16)
    X = U * n
    x0 = np.floor(X[:, :, 0].min(1) - 0.5).astype(int).clip(0, n - 1); x1 = np.ceil(X[:, :, 0].max(1) + 0.5).astype(int).clip(0, n)
    y0 = np.floor(X[:, :, 1].min(1) - 0.5).astype(int).clip(0, n - 1); y1 = np.ceil(X[:, :, 1].max(1) + 0.5).astype(int).clip(0, n)
    for k in range(len(X)):
        if x1[k] <= x0[k] or y1[k] <= y0[k]: continue
        (ax, ay), (bx, by), (cx, cy) = X[k]
        det = (by - cy) * (ax - cx) + (cx - bx) * (ay - cy)
        if abs(det) < 1e-12: continue
        px = (np.arange(x0[k], x1[k]) + 0.5)[None, :]; py = (np.arange(y0[k], y1[k]) + 0.5)[:, None]
        l1 = ((by - cy) * (px - cx) + (cx - bx) * (py - cy)) / det
        l2 = ((cy - ay) * (px - cx) + (ax - cx) * (py - cy)) / det
        l3 = 1 - l1 - l2
        m = (l1 > 1e-9) & (l2 > 1e-9) & (l3 > 1e-9)
        cnt[y0[k]:y1[k], x0[k]:x1[k]] += m
        if flip[k]: fl[y0[k]:y1[k], x0[k]:x1[k]] += m
    return cnt, fl

def compute(me, raster=1024, want_raster=False):
    co, lv, uv, tl, tp, ls, tot, le = _arrays(me)
    nt = len(tl)
    P = co[lv[tl]]; U = uv[tl]
    d1, d2 = P[:, 1] - P[:, 0], P[:, 2] - P[:, 0]
    area3 = 0.5 * np.linalg.norm(np.cross(d1, d2), axis=1)
    e1, e2 = U[:, 1] - U[:, 0], U[:, 2] - U[:, 0]
    uva_s = 0.5 * (e1[:, 0] * e2[:, 1] - e1[:, 1] * e2[:, 0])
    uva = np.abs(uva_s)
    ok3 = area3 > 1e-9 * max(area3.max(), 1e-30)
    out = dict(tris=nt, faces=len(ls), degenerate_3d_tris=int((~ok3).sum()))
    # angle deviation: mean absolute difference of the three corner angles 3D vs UV (degrees), per triangle
    uv_ok = ok3 & (uva > 1e-14)
    dev = np.abs(_angles(P) - _angles(np.concatenate([U, np.zeros((nt, 3, 1))], 2))).mean(1)
    dev = np.where(uv_ok, dev, 60.0)                       # a triangle with no UV area is as bad as a regular triangle squashed to a line
    out["angle_mean"] = float(dev[ok3].mean()); out["angle_p95"] = float(np.percentile(dev[ok3], 95))
    # area distortion after global normalisation
    gs = uva[ok3].sum() / area3[ok3].sum()
    ratio = np.maximum(uva[ok3], 1e-12 * gs * area3[ok3]) / area3[ok3] / gs
    lr = np.log(ratio)
    out["area_logstd"] = float(lr.std())
    w = area3[ok3]; mw = (lr * w).sum() / w.sum(); out["area_logstd_w"] = float(math.sqrt((w * (lr - mw) ** 2).sum() / w.sum()))
    out["area_worse2x_share"] = float(((ratio > 2) | (ratio < 0.5)).mean())
    # flips (UV winding against the 3D winding; every input has consistent normals) and zero-area UV triangles
    out["flipped_faces"] = int((uva_s[ok3] < -1e-14).sum()); out["zero_uv_tris"] = int((ok3 & (uva <= 1e-14)).sum())
    # texel density
    dens = np.sqrt(uva[ok3] / area3[ok3]); mean_d = (dens * w).sum() / w.sum()
    out["texel_cv_face"] = float(math.sqrt((w * (dens - mean_d) ** 2).sum() / w.sum()) / mean_d)
    # edges: seams and islands via loop pairs per edge
    ne = len(me.edges)
    ev = np.empty(ne * 2, np.int32); me.edges.foreach_get("vertices", ev); ev = ev.reshape(-1, 2)
    elen = np.linalg.norm(co[ev[:, 0]] - co[ev[:, 1]], axis=1)
    pol = np.repeat(np.arange(len(ls)), tot)                 # polygon of each loop
    idx = np.arange(len(lv)); nxt = ls[pol] + (idx - ls[pol] + 1) % tot[pol]
    order = np.argsort(le, kind="stable"); cntE = np.bincount(le, minlength=ne)
    two = np.nonzero(cntE == 2)[0]
    starts = np.concatenate([[0], np.cumsum(cntE)[:-1]])
    a = order[starts[two]]; b = order[starts[two] + 1]
    # loop a goes v(a)->v(nxt a); the same edge in loop b: match corners by vertex
    same = lv[a] == lv[b]                                    # same direction (inconsistent winding) -> match a with b
    ub_for_a = np.where(same[:, None], uv[b], uv[nxt[b]]); ub_for_na = np.where(same[:, None], uv[nxt[b]], uv[b])
    cut = (np.abs(uv[a] - ub_for_a).max(1) > 1e-5) | (np.abs(uv[nxt[a]] - ub_for_na).max(1) > 1e-5)
    out["seam_len_share"] = float(elen[two][cut].sum() / elen.sum())
    out["seam_edge_share"] = float(cut.sum() / max(ne, 1))
    # islands: union-find over polygons joined through uncut 2-loop edges
    par = np.arange(len(ls))
    def find(x):
        while par[x] != x:
            par[x] = par[par[x]]; x = par[x]
        return x
    for pa, pb in zip(pol[a[~cut]].tolist(), pol[b[~cut]].tolist()):
        ra, rb = find(pa), find(pb)
        if ra != rb: par[ra] = rb
    roots = np.array([find(i) for i in range(len(ls))]); _, isl = np.unique(roots, return_inverse=True)
    n_isl = int(isl.max()) + 1; out["islands"] = n_isl
    # island density uniformity (area-weighted over islands)
    ti = isl[tp]
    a3 = np.bincount(ti[ok3], weights=area3[ok3], minlength=n_isl); au = np.bincount(ti[ok3], weights=uva[ok3], minlength=n_isl)
    good = a3 > 0; di = np.sqrt(au[good] / a3[good]); wi = a3[good]; mi = (di * wi).sum() / wi.sum()
    out["texel_cv_island"] = float(math.sqrt((wi * (di - mi) ** 2).sum() / wi.sum()) / mi)
    # layout in 0-1
    cen = U.mean(1); inside = ((cen >= 0) & (cen <= 1)).all(1)
    out["uv_area_sum"] = float(uva.sum()); out["out_of_01_area_share"] = float(uva[~inside].sum() / max(uva.sum(), 1e-30))
    lo, hi = uv.min(0), uv.max(0); out["uv_bbox_w"] = float(hi[0] - lo[0]); out["uv_bbox_h"] = float(hi[1] - lo[1])
    out["bbox_fill"] = float(uva.sum() / max((hi[0] - lo[0]) * (hi[1] - lo[1]), 1e-30))
    cnt = fl = None
    if raster:
        cnt, fl = rasterize(U[inside | True], uva_s < -1e-14, raster)
        cov = cnt > 0
        out["pack_fill_01"] = float(cov.mean())                          # packing efficiency: share of the 0-1 square covered
        out["overlap_pixel_share"] = float((cnt > 1).sum() / max(cov.sum(), 1))   # of the covered pixels, share covered more than once
        out["overlap_area_01"] = float((cnt > 1).mean())                 # same, as part of the whole 0-1 square
    if want_raster: return out, dict(cnt=cnt, fl=fl, U=U, uva_s=uva_s, isl=ti)
    return out
