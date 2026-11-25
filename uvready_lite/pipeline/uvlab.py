# SPDX-License-Identifier: GPL-3.0-or-later
# SHARED WITH ASSETBAKE: copied by tools/sync_shared.py from aiasset/ai_asset_maker/pipeline/uvlab.py - do not edit here.
"""Numpy tools for the automatic UV unwrap (used by autouv.py). No scipy, works in Blender 4.2 to 5.x.

Parts:
  Topo(me)            numpy view of a mesh: face normals and areas, manifold face pairs, edge lengths
  cluster_hier(...)   chart segmentation: bottom-up face clustering on smoothed normals
  smooth_boundaries, merge_small, labels_to_seams
  metrics(me)         UV islands, distortion per triangle, optional raster fill and overlaps
  island_report(...)  per-island quality after an unwrap

Notes on cluster_hier (the heart of the unwrap):
  every cluster keeps the sum of area * smoothed normal, the sum of area * raw normal, its area and its perimeter.
  R = |sum| / area is the mean resultant length of the normals: 1 = flat, 0.64 = half cylinder, 0.5 = hemisphere,
  0 = closed tube. Two clusters may merge while R of the RAW normals stays above `rmin` (a chart must not wrap around
  a tube) and a conservative bound on the widest angle between a member's SMOOTHED normal and the chart mean stays
  under `cone` degrees. The merge cost is (1 - R_smoothed) + w_shape * relative loss of compactness.
"""
import heapq
import math

import numpy as np


class Topo:
    """Numpy view of a mesh (object mode)."""

    def __init__(self, me):
        self.me = me
        nv, ne, nl, nf = (len(me.vertices), len(me.edges), len(me.loops), len(me.polygons))
        self.nv, self.ne, self.nl, self.nf = (nv, ne, nl, nf)
        co = np.empty(nv * 3, np.float64)
        me.vertices.foreach_get('co', co)
        self.co = co.reshape(-1, 3)
        self.lv = np.empty(nl, np.int32)
        me.loops.foreach_get('vertex_index', self.lv)
        self.le = np.empty(nl, np.int32)
        me.loops.foreach_get('edge_index', self.le)
        self.ls = np.empty(nf, np.int32)
        me.polygons.foreach_get('loop_start', self.ls)
        self.lt = np.empty(nf, np.int32)
        me.polygons.foreach_get('loop_total', self.lt)
        self.lp = np.repeat(np.arange(nf, dtype=np.int32), self.lt)
        n = np.empty(nf * 3, np.float64)
        me.polygons.foreach_get('normal', n)
        self.fn = n.reshape(-1, 3)
        self.fa = np.empty(nf, np.float64)
        me.polygons.foreach_get('area', self.fa)
        c = np.empty(nf * 3, np.float64)
        me.polygons.foreach_get('center', c)
        self.fc = c.reshape(-1, 3)
        ev = np.empty(ne * 2, np.int32)
        me.edges.foreach_get('vertices', ev)
        self.ev = ev.reshape(-1, 2)
        self.elen = np.linalg.norm(self.co[self.ev[:, 0]] - self.co[self.ev[:, 1]], axis=1)
        order = np.argsort(self.le, kind='stable')
        cnt = np.bincount(self.le, minlength=ne)
        start = np.concatenate(([0], np.cumsum(cnt)[:-1]))
        self.ecnt = cnt
        m = np.nonzero(cnt == 2)[0]
        self.pe = m
        self.pl0 = order[start[m]]
        self.pl1 = order[start[m] + 1]
        self.pf0 = self.lp[self.pl0]
        self.pf1 = self.lp[self.pl1]
        self.plen = self.elen[m]
        d = np.einsum('ij,ij->i', self.fc[self.pf1] - self.fc[self.pf0], self.fn[self.pf0])
        self.pconcave = d > 0
        self.pcos = np.einsum('ij,ij->i', self.fn[self.pf0], self.fn[self.pf1])
        self.fper = np.bincount(self.lp, weights=self.elen[self.le], minlength=nf)
        # needle faces: so thin that their normal means nothing (CAD and STL meshes are full of them)
        longest = np.zeros(nf)
        np.maximum.at(longest, self.lp, self.elen[self.le])
        self.sliver = self.fa < 1e-4 * longest * longest

    def smoothed_normals(self, iters, sigma_deg=40.0):
        """Area-weighted neighbour averaging of face normals. Bilateral: the weight of a neighbour falls with the
        dihedral angle of the shared edge (exp(-(angle/sigma)^2)), so thin rods and hard edges are not blurred away.
        sigma_deg=0 gives plain averaging."""
        n = self.fn * self.fa[:, None]
        if sigma_deg > 0:
            ang = np.arccos(np.clip(self.pcos, -1, 1))
            w = np.exp(-(ang / math.radians(sigma_deg)) ** 2)[:, None]
        else:
            w = 1.0
        for _ in range(iters):
            acc = n.copy()
            np.add.at(acc, self.pf0, n[self.pf1] * w)
            np.add.at(acc, self.pf1, n[self.pf0] * w)
            l = np.linalg.norm(acc, axis=1)
            l[l == 0] = 1
            n = acc / l[:, None] * self.fa[:, None]
        l = np.linalg.norm(n, axis=1)
        l[l == 0] = 1
        return n / l[:, None]

    def shells(self):
        """Connected components of the faces, computed once."""
        if getattr(self, '_shells', None) is None:
            self._shells = self.components()
        return self._shells

    def components(self, labels=None):
        """Connected components of faces; with labels, only same-label neighbours connect."""
        parent = list(range(self.nf))

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x
        f0, f1 = (self.pf0, self.pf1)
        if labels is not None:
            k = labels[f0] == labels[f1]
            f0, f1 = (f0[k], f1[k])
        for a, b in zip(f0.tolist(), f1.tolist()):
            ra, rb = (find(a), find(b))
            if ra != rb:
                parent[rb] = ra
        root = np.array([find(i) for i in range(self.nf)])
        _, comp = np.unique(root, return_inverse=True)
        return comp.astype(np.int32)


def long_side(T, fill=0.55, part=0.9):
    """The longest an island should be, in mesh units: `part` of the side of the texture square when the layout
    fills `fill` of it. A longer island has to lie on the diagonal and shrinks everything else."""
    return part * math.sqrt(float(T.fa.sum()) / fill)


def island_extent(uv, T, isl, ni):
    """Length of every UV island along its own long axis (principal axis of its corners), in UV units."""
    il = isl[T.lp]
    n = np.maximum(np.bincount(il, minlength=ni), 1)
    mean = np.stack((np.bincount(il, weights=uv[:, 0], minlength=ni), np.bincount(il, weights=uv[:, 1], minlength=ni)), 1) / n[:, None]
    d = uv - mean[il]
    cxx = np.bincount(il, weights=d[:, 0] * d[:, 0], minlength=ni)
    cyy = np.bincount(il, weights=d[:, 1] * d[:, 1], minlength=ni)
    cxy = np.bincount(il, weights=d[:, 0] * d[:, 1], minlength=ni)
    th = 0.5 * np.arctan2(2 * cxy, cxx - cyy)
    t = d[:, 0] * np.cos(th)[il] + d[:, 1] * np.sin(th)[il]
    lo = np.full(ni, np.inf)
    hi = np.full(ni, -np.inf)
    np.minimum.at(lo, il, t)
    np.maximum.at(hi, il, t)
    return np.where(hi > lo, hi - lo, 0.0)


def angle_defect(T):
    """2 pi minus the sum of the face angles at every vertex (discrete Gaussian curvature); 0 at open or
    non-manifold edges and at unused vertices."""
    nxt = np.arange(T.nl) + 1
    last = T.ls + T.lt - 1
    nxt[last] = T.ls
    prv = np.arange(T.nl) - 1
    prv[T.ls] = last
    p = T.co[T.lv]
    u = p[nxt] - p
    v = p[prv] - p
    d = np.linalg.norm(u, axis=1) * np.linalg.norm(v, axis=1)
    ang = np.arccos(np.clip(np.einsum('ij,ij->i', u, v) / np.maximum(d, 1e-30), -1, 1))
    total = np.bincount(T.lv, weights=ang, minlength=T.nv)
    skip = np.bincount(T.lv, minlength=T.nv) == 0
    skip[T.ev[T.ecnt != 2].ravel()] = True
    return np.where(skip, 0.0, 2 * math.pi - total), ang


def flat_patches(T, tol_deg=1.5, min_share=0.002, min_shell_share=0.04):
    """Rigid patches: connected groups of coplanar faces (every normal within tol_deg of the normal of the
    largest face of the group) that cover at least min_share of the mesh area, or min_shell_share of the area of
    their own shell. One triangle is never a patch. Returns the patch id per face, -1 outside."""
    out = np.full(T.nf, -1, np.int64)
    ct = math.cos(math.radians(tol_deg))
    # a polygon with more than 3 corners counts only when it is planar itself
    nxt = np.arange(T.nl) + 1
    last = T.ls + T.lt - 1
    nxt[last] = T.ls
    prv = np.arange(T.nl) - 1
    prv[T.ls] = last
    p = T.co[T.lv]
    cn = np.cross(p[nxt] - p, p[prv] - p)
    cl = np.linalg.norm(cn, axis=1)
    dev = np.abs(np.einsum('ij,ij->i', cn, T.fn[T.lp])) / np.maximum(cl, 1e-30)
    dev[cl < 1e-12 * max(float(T.fa.max()), 1e-30)] = 1.0
    planar = np.ones(T.nf, bool)
    np.logical_and.at(planar, T.lp, dev >= ct)
    wild = T.sliver                      # a needle face joins the patch around it whatever its normal says
    planar |= wild
    k = ((T.pcos >= ct) | wild[T.pf0] | wild[T.pf1]) & planar[T.pf0] & planar[T.pf1]
    f0, f1 = (T.pf0[k], T.pf1[k])
    shell = T.shells()
    shell_area = np.bincount(shell, weights=T.fa)
    floor = np.minimum(min_share * T.fa.sum(), min_shell_share * shell_area[shell])   # per face: the area a patch needs
    # candidates: faces in a pairwise-coplanar component that is large enough, and large single polygons
    comp = _components_pairs(T.nf, f0, f1)
    carea = np.bincount(comp, weights=T.fa)
    cand = planar & (carea[comp] >= floor) & ((np.bincount(comp, minlength=len(carea))[comp] > 1) | (T.lt > 3))
    if not cand.any():
        return out
    nb = {}
    for a, b in zip(f0.tolist(), f1.tolist()):
        nb.setdefault(a, []).append(b)
        nb.setdefault(b, []).append(a)
    fn = T.fn
    wild = wild.tolist()
    fa = T.fa.tolist()
    fl = floor.tolist()
    order = np.nonzero(cand & ~T.sliver)[0]
    order = order[np.argsort(-T.fa[order], kind='stable')]
    seen = np.zeros(T.nf, bool)
    npatch = 0
    budget = 20 * T.nf                   # face visits; a dense smooth mesh would otherwise flood again from every seed
    for s in order.tolist():
        if seen[s]:
            continue
        if budget <= 0:
            break
        n0 = fn[s]
        seen[s] = True
        members = [s]
        stack = [s]
        area = fa[s]
        while stack:
            f = stack.pop()
            for g in nb.get(f, ()):
                if not seen[g] and (wild[g] or fn[g] @ n0 >= ct):
                    seen[g] = True
                    members.append(g)
                    stack.append(g)
                    area += fa[g]
        budget -= len(members)
        if area >= fl[s] and (len(members) > 1 or T.lt[s] > 3):
            out[members] = npatch
            npatch += 1
        else:
            seen[members[1:]] = False          # the seed stays used; the others may join a later patch
    return out


def _components_pairs(n, f0, f1):
    """Connected components of n nodes joined by the pairs (f0, f1): a label per node, 0 .. count - 1."""
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    for a, b in zip(f0.tolist(), f1.tolist()):
        ra, rb = (find(a), find(b))
        if ra != rb:
            parent[rb] = ra
    root = np.array([find(i) for i in range(n)], dtype=np.int64) if n else np.zeros(0, np.int64)
    _, inv = np.unique(root, return_inverse=True)
    return inv.reshape(-1)


class Hard:
    """What the chart builder needs to treat the hard-surface parts of a mesh as hard surface.
      block    per manifold face pair: a crease of at least `sharp` degrees. Charts never merge across it, so seams
               follow the sharp edges.
      region   per face: the smooth region it lies in (faces joined through edges that are not blocked)
      whole    per face: its region becomes one chart as it is, because it can be flattened with next to no
               stretch: a flat region of any shape, a gently curved region with one border, or a tube wall
               (two borders: side of a cylinder, a cone, a barrel), which is cut open along `seams`
      patch    rigid flat patch per face (-1 = none) in the other regions; a patch is never cut and is the unit
               the clustering starts from
      cones    vertices whose angle defect is above `cone_k` radians, cone_faces: the faces around each of them.
               A chart with such a vertex inside cannot be flattened without stretch, so a chart with a rigid
               patch never gets one inside.
    All of it only in the shells that are mostly hard surface (hard_face). On an organic mesh everything is empty
    and the clustering works as without it."""

    def __init__(self, T, tol_deg=1.5, min_share=0.002, min_shell_share=0.04, sharp=40.0, cone_k=0.2,
                 region_k=0.15, disk_sum=1.5, tube_sum=4.0, shell_share=0.5, cad_soft=15.0, cad_soft_max=0.1, cad_sharp_min=0.03):
        self.T = T
        self.block = (T.pcos <= math.cos(math.radians(sharp))) & ~(T.sliver[T.pf0] | T.sliver[T.pf1])
        self.seams = np.zeros(T.ne, bool)
        self.whole = np.zeros(T.nf, bool)
        self.tubes = 0
        K, _ = angle_defect(T)
        K = np.abs(K)
        self.region = np.zeros(T.nf, np.int64)
        if self.block.any():
            self._regions(K, region_k, disk_sum, tube_sum)
        self.patch = flat_patches(T, tol_deg, min_share, min_shell_share)
        self.patch[self.whole] = -1
        # Only a shell (connected part) that is hard surface is treated as one. It is when at least shell_share of
        # its area lies in whole regions and rigid patches, or when its edges are of two kinds only, as on a CAD
        # part: creases (at least cad_sharp_min of the edge length) and smooth tessellation, with at most
        # cad_soft_max of the length between cad_soft degrees and a crease. The other shells get the plain
        # clustering, with no creases.
        shell = T.shells()
        ns = int(shell.max()) + 1
        hs = self.whole | (self.patch >= 0)
        share = np.bincount(shell, weights=T.fa * hs, minlength=ns) / np.maximum(np.bincount(shell, weights=T.fa, minlength=ns), 1e-30)
        ps = shell[T.pf0]
        elen = np.maximum(np.bincount(ps, weights=T.plen, minlength=ns), 1e-30)
        soft = ~self.block & (T.pcos <= math.cos(math.radians(cad_soft)))
        cad = (np.bincount(ps, weights=T.plen * self.block, minlength=ns) >= cad_sharp_min * elen) & (np.bincount(ps, weights=T.plen * soft, minlength=ns) <= cad_soft_max * elen)
        self.hard_face = ((share >= shell_share) | cad)[shell]
        self.hard_share = float(T.fa[self.hard_face].sum() / max(float(T.fa.sum()), 1e-30))
        off = ~self.hard_face
        self.whole[off] = False
        self.patch[off] = -1
        self.block &= self.hard_face[T.pf0]
        edge_off = np.zeros(T.ne, bool)
        edge_off[T.le[off[T.lp]]] = True
        self.seams[edge_off] = False
        self.rigid = self.patch >= 0
        self.cones = np.zeros(0, np.int64)
        self.cone_faces = []
        self.face_cone = np.zeros(T.nf, bool)
        self.any = bool(self.block.any() or self.rigid.any())
        self.K = K
        l = np.nonzero((K > cone_k)[T.lv])[0] if self.rigid.any() else []
        if len(l):
            o = np.argsort(T.lv[l], kind='stable')
            l = l[o]
            v = T.lv[l]
            cut = np.nonzero(np.diff(v))[0] + 1
            self.cones = v[np.concatenate(([0], cut))]
            self.cone_faces = np.split(T.lp[l], cut)
            self.face_cone[T.lp[l]] = True

    def _regions(self, K, region_k, disk_sum, tube_sum):
        T = self.T
        max_len = long_side(T)
        keep = ~self.block
        reg = _components_pairs(T.nf, T.pf0[keep], T.pf1[keep])
        self.region = reg
        nr = int(reg.max()) + 1
        if nr == T.nf:
            return
        cnt = np.bincount(reg, minlength=nr)
        A = np.bincount(reg, weights=T.fa, minlength=nr)
        N = np.zeros((nr, 3))
        np.add.at(N, reg, T.fn * T.fa[:, None])
        R = np.linalg.norm(N, axis=1) / np.maximum(A, 1e-30)
        # nodes = (region, vertex); border edges of a region: to another region, open or non-manifold
        rl = reg[T.lp]
        key = rl * T.nv + T.lv
        nodes, node_of_loop = np.unique(key, return_inverse=True)
        node_of_loop = node_of_loop.reshape(-1)
        node_reg = nodes // T.nv
        nV = np.bincount(node_reg, minlength=nr)
        ekey = np.unique(rl * T.ne + T.le)
        nE = np.bincount(ekey // T.ne, minlength=nr)
        chi = nV - nE + cnt
        nxt = np.arange(T.nl) + 1
        last = T.ls + T.lt - 1
        nxt[last] = T.ls
        eb = np.ones(T.ne, bool)                      # edge is a border of the regions of its faces
        inner = reg[T.pf0] == reg[T.pf1]
        eb[T.pe[inner]] = False
        bl = np.nonzero(eb[T.le])[0]                  # loops on a border edge
        na, nb = (node_of_loop[bl], node_of_loop[nxt[bl]])
        on_border = np.zeros(len(nodes), bool)
        on_border[na] = True
        on_border[nb] = True
        comp = _components_pairs(len(nodes), na, nb)
        ck = np.unique(node_reg[on_border] * len(nodes) + comp[on_border])
        loops = np.bincount(ck // len(nodes), minlength=nr)
        Kn = K[nodes % T.nv]
        Kn[on_border] = 0
        ksum = np.bincount(node_reg, weights=Kn, minlength=nr)
        kmax = np.zeros(nr)
        np.maximum.at(kmax, node_reg, Kn)
        smooth = (kmax < region_k) & (cnt >= 2)
        flat = (R > 0.9995) & (cnt >= 2)
        disk = smooth & (loops == 1) & (chi == 1) & (ksum <= disk_sum)
        tube = smooth & (loops == 2) & (chi == 0) & (ksum <= tube_sum) & ~flat
        whole = flat | disk
        for r in np.nonzero(tube)[0].tolist():
            path = self._tube_cut(r, reg, node_reg, nodes, comp, on_border, max_len)
            if path is not None:
                self.seams[path] = True
                whole[r] = True
                self.tubes += 1
        self.whole = whole[reg]

    def _tube_cut(self, r, reg, node_reg, nodes, comp, on_border, max_len):
        """Edges that cut tube region r open: the shortest path between its two borders (Dijkstra over its
        edges). A tube longer around than max_len gets several cuts, evenly spaced, so that no strip is longer
        than the texture is wide."""
        T = self.T
        faces = np.nonzero(reg == r)[0]
        loops = np.concatenate([np.arange(T.ls[f], T.ls[f] + T.lt[f]) for f in faces.tolist()])
        e = np.unique(T.le[loops])
        k = (node_reg == r) & on_border
        bv = nodes[k] % T.nv
        first = comp[k] == comp[k][0]
        src = set(bv[first].tolist())
        dst = set(bv[~first].tolist())
        if not src or not dst:
            return None
        adj = {}
        ring = {}
        around = 0.0
        for i, (a, b), w in zip(e.tolist(), T.ev[e].tolist(), T.elen[e].tolist()):
            adj.setdefault(a, []).append((b, w, i))
            adj.setdefault(b, []).append((a, w, i))
            if a in src and b in src:
                ring.setdefault(a, []).append((b, w))
                ring.setdefault(b, []).append((a, w))
                around += w
        n = max(1, min(8, int(math.ceil(around / max(max_len, 1e-30)))))
        starts = [next(iter(src))]
        if n > 1:                                   # walk the first border, drop a start every around / n
            v = starts[0]
            seen = {v}
            run = 0.0
            while len(starts) < n:
                step = [(u, w) for u, w in ring.get(v, ()) if u not in seen]
                if not step:
                    break
                v, w = step[0]
                seen.add(v)
                run += w
                if run >= around * len(starts) / n:
                    starts.append(v)
        path = []
        for s0 in starts:
            dist = {s0: 0.0}
            prev = {}
            heap = [(0.0, s0)]
            end = None
            while heap:
                d, v = heapq.heappop(heap)
                if d > dist.get(v, math.inf):
                    continue
                if v in dst:
                    end = v
                    break
                for u, w, i in adj.get(v, ()):
                    nd = d + w
                    if nd < dist.get(u, math.inf) and u not in src:
                        dist[u] = nd
                        prev[u] = (v, i)
                        heapq.heappush(heap, (nd, u))
            if end is None:
                return None
            while end in prev:
                end, i = prev[end]
                path.append(i)
        return path

    def atom_labels(self):
        """The smallest units a chart is made of: one per rigid patch, one per other face."""
        lab = np.where(self.rigid, self.patch, -1)
        n = int(self.patch.max()) + 1
        free = np.nonzero(~self.rigid)[0]
        lab[free] = n + np.arange(len(free))
        return lab

    def start_labels(self):
        """Clusters the hierarchy starts from: the atoms, and every `whole` region as one cluster."""
        lab = self.atom_labels()
        if self.whole.any():
            lab[self.whole] = int(lab.max()) + 1 + self.region[self.whole]
            lab = relabel(lab)
        return lab

    def rigid_charts(self, labels):
        """Boolean per label: the chart holds a rigid patch."""
        r = np.zeros(int(labels.max()) + 1, bool)
        r[labels[self.rigid]] = True
        return r

    def cone_sets(self, labels):
        """-> (set of charts around every cone vertex, chart -> indices of the cone vertices it touches)."""
        cs = [set(labels[f].tolist()) for f in self.cone_faces]
        by = {}
        for i, st in enumerate(cs):
            for x in st:
                by.setdefault(x, []).append(i)
        return cs, by


def cluster_hier(T, sn, rmin=0.55, w_shape=0.2, max_share=1.0, w_concave=0.0, labels=None, target=None, restrict=None, cone=120.0, hard=None):
    """Bottom-up clustering (Garland-style hierarchical face clustering, simplified).
    Each cluster keeps: sum of area*smoothed normal, sum of area*raw normal, area, perimeter.
    R = |sum|/area is the mean resultant length of normals (1 = flat, 0.64 = half cylinder, 0.5 = hemisphere, 0 = closed tube).
    A merge is allowed while
      R of the RAW normals of the merged chart >= rmin  (stops charts that wrap around a tube or close on themselves), and
      a conservative bound on the largest angle between a member's SMOOTHED normal and the chart mean <= cone degrees
      (bound: dev(m) <= max(dev(a) + angle(Na, Nm), dev(b) + angle(Nb, Nm)); real cones are narrower than the bound).
    Cost = (1 - R_smoothed) + w_shape * relative loss of compactness (+ w_concave * share of the common border that is not a concave crease).
    labels: start from an existing partition. target: merge down to this many clusters, ignoring the limits.
    restrict: boolean face mask, only these faces take part (others get label -1).
    hard: a Hard object (blocked creases, rigid flat patches, cone vertices). With it, no merge crosses a blocked
    crease, and a chart that holds a rigid patch never gets a cone vertex in its interior, so it stays developable
    and the unwrap leaves its flat faces undistorted. Not used with `target` or `restrict`."""
    nf = T.nf
    lab0 = np.arange(nf, dtype=np.int64) if labels is None else labels.astype(np.int64)
    ncl = int(lab0.max()) + 1
    N = np.zeros((ncl, 3))
    np.add.at(N, lab0, sn * T.fa[:, None])
    NR = np.zeros((ncl, 3))
    np.add.at(NR, lab0, T.fn * T.fa[:, None])
    A = np.bincount(lab0, weights=T.fa, minlength=ncl)
    f0, f1, pl, conc, pc = (T.pf0, T.pf1, T.plen, T.pconcave, T.pcos)
    if restrict is not None:
        k = restrict[f0] & restrict[f1]
        f0, f1, pl, conc, pc = (f0[k], f1[k], pl[k], conc[k], pc[k])
    if hard is not None and restrict is None:
        k = ~hard.block
        f0, f1, pl, conc, pc = (f0[k], f1[k], pl[k], conc[k], pc[k])
    a, b = (lab0[f0], lab0[f1])
    k = a != b
    a, b, plk = (a[k], b[k], pl[k])
    crease_len = plk * (conc[k] & (pc[k] < 0.82))
    P = np.bincount(lab0, weights=T.fper, minlength=ncl)
    same = lab0[T.pf0] == lab0[T.pf1]
    np.subtract.at(P, lab0[T.pf0][same], 2 * T.plen[same])
    adj = [None] * ncl
    for x, y, s, w in zip(a.tolist(), b.tolist(), plk.tolist(), crease_len.tolist()):
        dx = adj[x]
        if dx is None:
            dx = adj[x] = {}
        dy = adj[y]
        if dy is None:
            dy = adj[y] = {}
        e = dx.get(y)
        if e is None:
            e = [0.0, 0.0]
            dx[y] = e
            dy[x] = e
        e[0] += s
        e[1] += w
    Nl = N.tolist()
    NRl = NR.tolist()
    Al = (A + 1e-30).tolist()
    Pl = P.tolist()
    nn = np.linalg.norm(N, axis=1)
    nn[nn == 0] = 1
    cosd = np.einsum('ij,ij->i', sn, (N / nn[:, None])[lab0])
    dmin = np.ones(ncl)
    np.minimum.at(dmin, lab0, cosd)
    Dl = np.arccos(np.clip(dmin, -1, 1)).tolist()
    conelim = math.radians(cone)
    acos = math.acos
    sqrt = math.sqrt
    Atot = float(T.fa.sum()) if restrict is None else float(T.fa[restrict].sum())
    amax = max_share * Atot
    ver = [0] * ncl
    alive = [adj[i] is not None for i in range(ncl)]
    four_pi = 4 * math.pi
    free = target is not None
    use_hard = hard is not None and not free and bool(hard.rigid.any())
    if use_hard:
        has_r = np.zeros(ncl, bool)
        has_r[lab0[hard.rigid]] = True
        has_r = has_r.tolist()
        cs = [set(lab0[f].tolist()) for f in hard.cone_faces]      # clusters around every cone vertex
        ic = [0] * ncl                                             # cone vertices inside a cluster
        cv = [None] * ncl                                          # cone vertices on the border of a cluster
        for i, st in enumerate(cs):
            if len(st) == 1:
                ic[next(iter(st))] += 1
            else:
                for x in st:
                    if cv[x] is None:
                        cv[x] = set()
                    cv[x].add(i)

        def encloses(x, y):
            """True when the merged cluster would hold a rigid patch and a cone vertex in its interior."""
            if not (has_r[x] or has_r[y]):
                return False
            if ic[x] or ic[y]:
                return True
            sx, sy = (cv[x], cv[y])
            if not sx or not sy:
                return False
            if len(sx) > len(sy):
                sx, sy = (sy, sx)
            for i in sx:
                if len(cs[i]) == 2 and i in sy:
                    return True
            return False

    def cost(x, y, e):
        """-> (cost, allowed, cone bound of the merged cluster)"""
        nx, ny = (Nl[x], Nl[y])
        mx, my, mz = (nx[0] + ny[0], nx[1] + ny[1], nx[2] + ny[2])
        am = Al[x] + Al[y]
        lm = sqrt(mx * mx + my * my + mz * mz)
        lx = sqrt(nx[0] * nx[0] + nx[1] * nx[1] + nx[2] * nx[2]) * lm + 1e-30
        ly = sqrt(ny[0] * ny[0] + ny[1] * ny[1] + ny[2] * ny[2]) * lm + 1e-30
        dx_ = Dl[x] + acos(max(-1.0, min(1.0, (nx[0] * mx + nx[1] * my + nx[2] * mz) / lx)))
        dy_ = Dl[y] + acos(max(-1.0, min(1.0, (ny[0] * mx + ny[1] * my + ny[2] * mz) / ly)))
        dev = dx_ if dx_ > dy_ else dy_
        ok = True
        if not free:
            rx, ry = (NRl[x], NRl[y])
            qx, qy, qz = (rx[0] + ry[0], rx[1] + ry[1], rx[2] + ry[2])
            ok = dev <= conelim and am <= amax and (sqrt(qx * qx + qy * qy + qz * qz) / am >= rmin)
            if ok and use_hard and encloses(x, y):
                ok = False
        pm = Pl[x] + Pl[y] - 2 * e[0]
        gm = pm * pm / (four_pi * am) + 1e-30
        gx = Pl[x] * Pl[x] / (four_pi * Al[x])
        gy = Pl[y] * Pl[y] / (four_pi * Al[y])
        shape = max((gm - max(gx, gy)) / gm, -1.0)
        c = 1 - lm / am + w_shape * shape
        if w_concave:
            c += w_concave * (1 - e[1] / e[0]) if e[0] > 0 else 0.0
        return (c, ok, dev)
    heap = []
    for x in range(ncl):
        dx = adj[x]
        if dx is None:
            continue
        for y, e in dx.items():
            if y > x:
                c, ok, dv = cost(x, y, e)
                if ok:
                    heap.append((c, x, y, 0, 0))
    heapq.heapify(heap)
    nclusters = sum(alive)
    parent = list(range(ncl))
    while heap:
        if free and nclusters <= target:
            break
        c, x, y, vx, vy = heapq.heappop(heap)
        if not (alive[x] and alive[y]) or ver[x] != vx or ver[y] != vy:
            continue
        if len(adj[x]) < len(adj[y]):
            x, y = (y, x)
        e = adj[x].pop(y)
        del adj[y][x]
        Dl[x] = cost(x, y, e)[2]
        nx, ny = (Nl[x], Nl[y])
        Nl[x] = [nx[0] + ny[0], nx[1] + ny[1], nx[2] + ny[2]]
        nx, ny = (NRl[x], NRl[y])
        NRl[x] = [nx[0] + ny[0], nx[1] + ny[1], nx[2] + ny[2]]
        Pl[x] = Pl[x] + Pl[y] - 2 * e[0]
        Al[x] += Al[y]
        if use_hard:
            has_r[x] = has_r[x] or has_r[y]
            ic[x] += ic[y]
            if cv[y]:
                if cv[x] is None:
                    cv[x] = set()
                sx = cv[x]
                for i in cv[y]:
                    st = cs[i]
                    st.discard(y)
                    st.add(x)
                    if len(st) == 1:
                        ic[x] += 1
                        sx.discard(i)
                    else:
                        sx.add(i)
                cv[y] = None
        dx = adj[x]
        for z, ez in adj[y].items():
            dz = adj[z]
            del dz[y]
            ex = dx.get(z)
            if ex is None:
                dx[z] = ez
                dz[x] = ez
            else:
                ex[0] += ez[0]
                ex[1] += ez[1]
        adj[y] = None
        alive[y] = False
        parent[y] = x
        nclusters -= 1
        ver[x] += 1
        vx = ver[x]
        for z, ez in dx.items():
            c, ok, dv = cost(x, z, ez)
            if ok:
                heapq.heappush(heap, (c, x, z, vx, ver[z]))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    roots = np.array([find(i) for i in range(ncl)], dtype=np.int64)
    out = roots[lab0]
    if restrict is not None:
        out = np.where(restrict, out, -1)
    return out


def relabel(labels):
    _, inv = np.unique(labels, return_inverse=True)
    return inv.astype(np.int64)


def smooth_boundaries(T, labels, iters=3, hard=None):
    """A face whose border is mostly shared with one other chart moves to that chart (shortens seams).
    With `hard`: faces of rigid patches stay, nothing moves across a blocked crease, and a face at a cone vertex
    does not move into a chart that holds a rigid patch."""
    labels = labels.copy()
    f0, f1, pl = (T.pf0, T.pf1, T.plen)
    use_hard = hard is not None and hard.any
    if use_hard:
        k = ~hard.block
        f0, f1, pl = (f0[k], f1[k], pl[k])
        rc = hard.rigid_charts(labels)
        stay = hard.rigid | hard.whole
        fcone = hard.face_cone
    nb = [[] for _ in range(T.nf)]
    for a, b, s in zip(f0.tolist(), f1.tolist(), pl.tolist()):
        nb[a].append((b, s))
        nb[b].append((a, s))
    moved_total = 0
    for _ in range(iters):
        diff = labels[f0] != labels[f1]
        cand = np.unique(np.concatenate((f0[diff], f1[diff])))
        if use_hard:
            cand = cand[~stay[cand]]
        L = labels.tolist()
        moved = 0
        for f in cand.tolist():
            own = L[f]
            acc = {}
            for g, s in nb[f]:
                acc[L[g]] = acc.get(L[g], 0.0) + s
            best = max(acc, key=acc.get)
            if use_hard and fcone[f] and rc[best]:
                continue
            if best != own and acc[best] > 1.2 * acc.get(own, 0.0) and (acc[best] > 0.5 * sum(acc.values())):
                labels[f] = best
                L[f] = best
                moved += 1
        moved_total += moved
        if not moved:
            break
    return (labels, moved_total)


def merge_small(T, sn, labels, min_share=0.004, min_faces=8, rfloor=0.5, hard=None, k_merge=0.05, speck_share=0.0002, speck_faces=3):
    """Connected charts smaller than min_share of the area (or min_faces) join the neighbour with the
    longest common border, as long as the merged chart keeps R (raw normals) >= rfloor, so two halves of a
    thin rod are not merged back into a closed tube. Tiny separate shells stay.
    With `hard`: a chart with a rigid patch or a whole region is small only by its area; no merge puts a cone
    vertex inside a chart with a rigid patch; and a merge across a blocked crease is taken only when the two
    charts stay a bent sheet: the vertices that end up inside carry no more than k_merge radians of angle defect
    together (a chamfer strip joins the face next to it, a cylinder cap does not join the wall). A speck (under
    speck_faces faces and under speck_share of the area) joins a neighbour whatever the crease between them: a
    crumb of one or two faces is not worth an island."""
    labels = T.components(labels).astype(np.int64)
    Atot = T.fa.sum()
    use_hard = hard is not None and hard.any
    if use_hard:
        vorder = np.argsort(T.lv, kind='stable')
        vstart = np.searchsorted(T.lv[vorder], np.arange(T.nv + 1))
        vfaces = T.lp[vorder]
        Kv = hard.K
    for _ in range(4):
        ncl = int(labels.max()) + 1
        A = np.bincount(labels, weights=T.fa, minlength=ncl)
        cnt = np.bincount(labels, minlength=ncl)
        N = np.zeros((ncl, 3))
        np.add.at(N, labels, T.fn * T.fa[:, None])
        small = (A < min_share * Atot) | (cnt < min_faces)
        a, b = (labels[T.pf0], labels[T.pf1])
        k = a != b
        border = {}
        crease = {}                       # (x, y), x < y -> vertices of their common border when it has a crease
        if use_hard:
            rc = hard.rigid_charts(labels)
            rc[labels[hard.whole]] = True
            small &= ~rc | (A < min_share * Atot)
            cs, cby = hard.cone_sets(labels)
            inner = np.zeros(ncl, bool)                  # charts with a cone vertex inside already
            for st in cs:
                if len(st) == 1:
                    inner[next(iter(st))] = True
            speck = small & (cnt < speck_faces) & (A < speck_share * Atot)
            kb = k & hard.block & (small[a] | small[b])
            for x, y, e in zip(a[kb].tolist(), b[kb].tolist(), T.pe[kb].tolist()):
                crease.setdefault((x, y) if x < y else (y, x), []).append(e)
        if not small.any():
            break
        a, b, s = (a[k], b[k], T.plen[k])
        for x, y, l in zip(a.tolist(), b.tolist(), s.tolist()):
            border.setdefault(x, {})
            border.setdefault(y, {})
            border[x][y] = border[x].get(y, 0.0) + l
            border[y][x] = border[y].get(x, 0.0) + l
        lab = labels

        def bent_sheet(x, y, edges):
            """The union of x and y across these crease edges takes in vertices with next to no angle defect."""
            total = 0.0
            for v in np.unique(T.ev[edges]).tolist():
                kv = Kv[v]
                if kv > 0:
                    lf = lab[vfaces[vstart[v]:vstart[v + 1]]]
                    if ((lf == x) | (lf == y)).all():
                        total += kv
                        if total > k_merge:
                            return False
            return True
        target = np.arange(ncl)
        changed = 0
        order = np.argsort(A, kind="stable")
        done = set()
        for x in order.tolist():
            if not small[x] or x not in border or x in done:
                continue
            best = None
            bl = 0.0
            for y, l in border[x].items():
                if y in done:
                    continue
                if l <= bl:
                    continue
                if use_hard and speck[x]:
                    best, bl = (y, l)
                    continue
                m = N[x] + N[y]
                r = np.sqrt(m @ m) / (A[x] + A[y])
                if r >= rfloor:
                    if use_hard and (rc[x] or rc[y]) and (inner[x] or inner[y] or any(cs[i] <= {x, y} for i in cby.get(x, ()))):
                        continue
                    edges = crease.get((x, y) if x < y else (y, x))
                    if edges is not None and not bent_sheet(x, y, edges):
                        continue
                    best, bl = (y, l)
            if best is not None:
                target[x] = best
                done.add(x)
                done.add(best)
                changed += 1
        if not changed:
            break
        labels = relabel(target[labels])
    return labels


def labels_to_seams(T, labels, extra=None):
    """Seams on the chart borders, plus the `extra` edges (boolean per edge: cuts inside a chart)."""
    seam = np.zeros(T.ne, dtype=bool) if extra is None else extra.copy()
    seam[T.pe[labels[T.pf0] != labels[T.pf1]]] = True
    T.me.edges.foreach_set('use_seam', seam)
    return int(seam.sum())


def _uv_array(me):
    uvl = me.uv_layers.active
    uv = np.empty(len(me.loops) * 2, np.float64)
    uvl.data.foreach_get('uv', uv)
    return uv.reshape(-1, 2)


def uv_cut(T, uv):
    """-> (UV vertex id per loop: loops of one vertex that share a UV position, boolean per manifold pair: the
    two faces are joined in UV). All numpy; uv_islands builds the islands from it."""
    q = np.round(uv * 100000.0).astype(np.int64)
    key = np.stack((T.lv.astype(np.int64), q[:, 0], q[:, 1]), axis=1)
    _, uvid = np.unique(key, axis=0, return_inverse=True)
    uvid = uvid.reshape(-1)
    nxt = np.arange(T.nl) + 1
    last = T.ls + T.lt - 1
    nxt[last] = T.ls
    a = uvid
    b = uvid[nxt]
    lo = np.minimum(a, b)
    hi = np.maximum(a, b)
    joined = (lo[T.pl0] == lo[T.pl1]) & (hi[T.pl0] == hi[T.pl1])
    return uvid, joined


def uv_islands(T, uv, want_ids=False, cut=None):
    """Island id per face from UV connectivity, and the boolean 'cut' flag per manifold pair.
    want_ids: also the UV vertex id per loop. cut: (uvid, joined) from uv_cut, when the caller has it."""
    uvid, joined = uv_cut(T, uv) if cut is None else cut
    isl = _components_pairs(T.nf, T.pf0[joined], T.pf1[joined])
    if want_ids:
        return (isl, ~joined, uvid)
    return (isl, ~joined)


def boundary_crossings(T, uv, isl, cut):
    """Number of proper self-intersections of each island's UV boundary. With no flipped faces, an island overlaps
    itself only if its boundary crosses itself, so this is a cheap overlap test (no rasterisation)."""
    bl = np.zeros(T.nl, bool)
    bl[T.pl0[cut]] = True
    bl[T.pl1[cut]] = True
    bl[T.ecnt[T.le] != 2] = True
    l = np.nonzero(bl)[0]
    nxt = l + 1
    last = (T.ls + T.lt - 1)[T.lp[l]]
    nxt = np.where(l == last, T.ls[T.lp[l]], nxt)
    A = uv[l]
    B = uv[nxt]
    ii = isl[T.lp[l]]
    order = np.argsort(ii, kind='stable')
    A, B, ii = (A[order], B[order], ii[order])
    ni = int(isl.max()) + 1
    out = np.zeros(ni, np.int64)
    bounds = np.searchsorted(ii, np.arange(ni + 1))
    for i in range(ni):
        s0, s1 = (bounds[i], bounds[i + 1])
        n = s1 - s0
        if n < 4:
            continue
        if n > 3000:                     # the n x n test would take gigabytes: say "maybe", the caller rasterises
            out[i] = 1
            continue
        a = A[s0:s1]
        b = B[s0:s1]
        d = b - a

        def cross(o, dv, pts):
            return dv[:, None, 0] * (pts[None, :, 1] - o[:, None, 1]) - dv[:, None, 1] * (pts[None, :, 0] - o[:, None, 0])
        c1 = cross(a, d, a)
        c2 = cross(a, d, b)
        eps = 1e-12
        straddle = (c1 > eps) & (c2 < -eps) | (c1 < -eps) & (c2 > eps)
        hit = straddle & straddle.T
        out[i] = int(np.triu(hit, 1).sum())
    return out


def _lay_out(T, uv, w, isl, cut, good):
    """Shared end of unfold and project: the islands `good` take the complex positions w, turned so that their
    border runs along the axes (length-weighted direction of the border edges, period 90 degrees), moved to the
    origin and scaled by 1 / sqrt(mesh area). They lie on top of each other there until the packer places them
    (kept near the origin on purpose: a UV layer is float32, and far from 0 it would lose the fine detail)."""
    ni = len(good)
    il = isl[T.lp]
    nxt = np.arange(T.nl) + 1
    nxt[T.ls + T.lt - 1] = T.ls
    bl = np.zeros(T.nl, bool)
    bl[T.pl0[cut]] = True
    bl[T.pl1[cut]] = True
    bl[T.ecnt[T.le] != 2] = True
    l = np.nonzero(bl)[0]
    e = w[nxt[l]] - w[l]
    ln = np.abs(e)
    dirs = np.where(ln > 0, (e / np.maximum(ln, 1e-30)) ** 4 * ln, 0)
    acc = np.bincount(il[l], weights=dirs.real, minlength=ni) + 1j * np.bincount(il[l], weights=dirs.imag, minlength=ni)
    rot = np.exp(-0.25j * np.angle(np.where(np.abs(acc) > 0, acc, 1)))
    w = w * rot[il]
    xy = np.stack((w.real, w.imag), 1)
    lo = np.full((ni, 2), np.inf)
    np.minimum.at(lo, il, xy)
    scale = 1.0 / max(math.sqrt(float(T.fa.sum())), 1e-30)
    new = uv.copy()
    k = good[il]
    new[k] = (xy[k] - lo[il][k]) * scale
    return new


def project(T, uv, islands):
    """Every island projected on the plane across its mean normal (what Smart UV Project does with a group of
    faces). No solver, so needle faces do no harm; right for an island that is nearly flat. Returns the uv."""
    isl, cut, _ = islands
    ni = int(isl.max()) + 1
    N = np.zeros((ni, 3))
    np.add.at(N, isl, T.fn * T.fa[:, None])
    ln = np.linalg.norm(N, axis=1)
    N = np.where(ln[:, None] > 0, N / np.maximum(ln, 1e-30)[:, None], np.array([0.0, 0.0, 1.0]))
    ref = np.where((np.abs(N[:, 2]) < 0.9)[:, None], np.array([0.0, 0.0, 1.0]), np.array([1.0, 0.0, 0.0]))
    ex = np.cross(ref, N)
    ex /= np.maximum(np.linalg.norm(ex, axis=1), 1e-30)[:, None]
    ey = np.cross(N, ex)
    il = isl[T.lp]
    p = T.co[T.lv]
    w = np.einsum('ij,ij->i', p, ex[il]) + 1j * np.einsum('ij,ij->i', p, ey[il])
    return _lay_out(T, uv, w, isl, cut, np.ones(ni, bool))


def unfold(T, uv, islands, faces_ok, tol=1e-4, loose=0.05):
    """Rigid unfolding of UV islands: flat regions, bent sheets, cylinder and cone walls. Every face is laid flat
    in its own plane and hinged to a neighbour that is already placed (breadth first from the largest face), so
    needle faces cannot flip and no solver is involved. The copies of a UV vertex that the faces around it give
    are then averaged. `err` is how far those copies lie apart, relative to the island size: under `tol` the
    island is developable and the result is exact; up to `loose` it is close to developable and the result is a
    candidate that the caller compares with the solver's. The long border edges of every island are turned
    parallel to the U or V axis. Islands are those of `uv`; only islands whose faces are all in `faces_ok` are
    tried. `islands` = (island per face, cut per pair, UV vertex per loop) of `uv`, from uv_islands.
    Returns (uv with every tried island replaced, exact per island, candidate per island)."""
    isl, cut, uvid = islands
    ni = int(isl.max()) + 1
    okc = np.bincount(isl, weights=~faces_ok, minlength=ni) == 0
    none = np.zeros(ni, bool)
    if not okc.any():
        return uv, none, none
    nl = T.nl
    nxt = np.arange(nl) + 1
    last = T.ls + T.lt - 1
    nxt[last] = T.ls
    # every polygon flat in its own frame (complex numbers): origin = first corner, x = first edge
    p = T.co[T.lv]
    o = p[T.ls][T.lp]
    ex = p[T.ls + 1][T.lp] - o
    ex /= np.maximum(np.linalg.norm(ex, axis=1), 1e-30)[:, None]
    ey = np.cross(T.fn[T.lp], ex)
    d = p - o
    z = np.einsum('ij,ij->i', d, ex) + 1j * np.einsum('ij,ij->i', d, ey)
    j = ~cut & okc[isl[T.pf0]]
    la, lb = (T.pl0[j], T.pl1[j])
    opposite = T.lv[la] == T.lv[nxt[lb]]            # consistent winding: the edge runs the other way in the neighbour
    nb = [[] for _ in range(T.nf)]
    for a, b, f, g, op in zip(la.tolist(), lb.tolist(), T.lp[la].tolist(), T.lp[lb].tolist(), opposite.tolist()):
        if op:
            nb[f].append((g, a, b))
            nb[g].append((f, b, a))
    zl = z.tolist()
    nx = nxt.tolist()
    out = [0j] * nl
    placed = np.zeros(T.nf, bool)
    ls, lt = (T.ls.tolist(), T.lt.tolist())
    seeds = np.nonzero(okc[isl])[0]
    seeds = seeds[np.argsort(-T.fa[seeds], kind='stable')]
    for s0 in seeds.tolist():
        if placed[s0]:
            continue
        placed[s0] = True
        for l in range(ls[s0], ls[s0] + lt[s0]):
            out[l] = zl[l]
        queue = [s0]
        head = 0
        while head < len(queue):
            f = queue[head]
            head += 1
            for g, a, b in nb[f]:
                if placed[g]:
                    continue
                # edge a -> nx[a] of f is edge nx[b] -> b of g
                den = zl[b] - zl[nx[b]]
                num = out[nx[a]] - out[a]
                if den == 0 or num == 0:
                    continue
                c = (num / den)
                c /= abs(c)
                t = out[nx[a]] - c * zl[b]
                for l in range(ls[g], ls[g] + lt[g]):
                    out[l] = c * zl[l] + t
                out[b] = out[nx[a]]
                out[nx[b]] = out[a]
                placed[g] = True
                queue.append(g)
    w = np.array(out, np.complex128)
    il = isl[T.lp]
    # consistency: the loops of one UV vertex must have landed on one point
    nid = int(uvid.max()) + 1
    cnt = np.maximum(np.bincount(uvid, minlength=nid), 1)
    mean = (np.bincount(uvid, weights=w.real, minlength=nid) + 1j * np.bincount(uvid, weights=w.imag, minlength=nid)) / cnt
    err = np.zeros(ni)
    np.maximum.at(err, il, np.abs(w - mean[uvid]))
    lo = np.full((ni, 2), np.inf)
    hi = np.full((ni, 2), -np.inf)
    xy = np.stack((w.real, w.imag), 1)
    np.minimum.at(lo, il, xy)
    np.maximum.at(hi, il, xy)
    size = np.maximum((hi - lo).max(1), 1e-30)
    allp = np.bincount(isl, weights=~placed, minlength=ni) == 0
    ok = okc & allp & (size > 1e-20)
    exact = ok & (err <= tol * size)
    cand = ok & ~exact & (err <= loose * size)
    good = exact | cand
    if not good.any():
        return uv, none, none
    return _lay_out(T, uv, mean[uvid], isl, cut, good), exact, cand


def untangle(T, uv, passes=5):
    """Turns single inverted UV triangles back by moving one UV vertex across the opposite edge (or, when there is
    no room, onto it: a needle face may end with no UV area, never with a negative one). A move is taken when it
    leaves fewer inverted triangles around the vertex and inverts none that was right. Needle triangles and the
    odd folded quad are repaired this way; a real fold of many faces is left (the caller cuts those faces out).
    Everything is judged on the float32 values a UV layer stores: a needle that is right in double precision can
    be inverted by the rounding alone. Returns (uv, boolean per face: still holds an inverted triangle)."""
    uvid = uv_cut(T, uv)[0]
    me = T.me
    me.calc_loop_triangles()
    nt = len(me.loop_triangles)
    tl = np.empty(nt * 3, np.int32)
    me.loop_triangles.foreach_get('loops', tl)
    tl = tl.reshape(-1, 3)
    tp = np.empty(nt, np.int32)
    me.loop_triangles.foreach_get('polygon_index', tp)
    tv = uvid[tl]
    nid = int(uvid.max()) + 1
    rep_loop = np.zeros(nid, np.int64)
    rep_loop[uvid] = np.arange(T.nl)
    pos = uv[rep_loop].astype(np.float32).astype(np.float64)       # one position per UV vertex, as stored

    def areas(idx):
        q = pos[tv[idx]]
        d1, d2 = (q[:, 1] - q[:, 0], q[:, 2] - q[:, 0])
        return d1[:, 0] * d2[:, 1] - d1[:, 1] * d2[:, 0]
    P = T.co[T.lv[tl]]
    a3 = np.linalg.norm(np.cross(P[:, 1] - P[:, 0], P[:, 2] - P[:, 0]), axis=1)
    live = a3 > 1e-9 * max(float(a3.max()), 1e-30)
    bad = np.nonzero((areas(np.arange(nt)) < 0) & live)[0]
    still = np.zeros(T.nf, bool)
    if not len(bad):
        return pos[uvid], still
    if len(bad) > max(2000, nt // 50):   # not needles but a wrecked layout: no repair by hand, report them
        still[tp[bad]] = True
        return pos[uvid], still
    order = np.argsort(tv.ravel(), kind='stable')
    start = np.searchsorted(tv.ravel()[order], np.arange(nid + 1))
    for _ in range(passes):
        moved = 0
        for t in bad.tolist():
            if areas(np.array([t]))[0] >= 0:
                continue
            best = None
            for c in range(3):
                v, v1, v2 = (int(tv[t, c]), int(tv[t, (c + 1) % 3]), int(tv[t, (c + 2) % 3]))
                e = pos[v2] - pos[v1]
                el = float(np.hypot(e[0], e[1]))
                if el <= 0:
                    continue
                n = np.array([-e[1], e[0]]) / el          # the side of the edge v1 -> v2 the vertex belongs on
                h = float((pos[v] - pos[v1]) @ n)
                around = order[start[v]:start[v + 1]] // 3
                lv_ = live[around]
                was_bad = (areas(around) < 0) & lv_
                old = pos[v].copy()
                for step in (-2 * h, -h + 1e-6 * el, -h + 4e-7, -h + 2e-6, -h + 1e-5):
                    new = (old + n * step).astype(np.float32).astype(np.float64)
                    pos[v] = new
                    now_bad = (areas(around) < 0) & lv_
                    if int(now_bad.sum()) < int(was_bad.sum()) and not (now_bad & ~was_bad).any():
                        gain = int(was_bad.sum()) - int(now_bad.sum())
                        if best is None or (gain, -abs(step)) > (best[0], -abs(best[1])):
                            best = (gain, step, v, new)
                        break
                pos[v] = old
            if best is not None:
                pos[best[2]] = best[3]
                moved += 1
        if not moved:
            break
    still[tp[(areas(np.arange(nt)) < 0) & live]] = True
    return pos[uvid], still


def tri_data(T, uv):
    me = T.me
    me.calc_loop_triangles()
    nt = len(me.loop_triangles)
    tl = np.empty(nt * 3, np.int32)
    me.loop_triangles.foreach_get('loops', tl)
    tl = tl.reshape(-1, 3)
    tp = np.empty(nt, np.int32)
    me.loop_triangles.foreach_get('polygon_index', tp)
    P = T.co[T.lv[tl]]
    Q = uv[tl]
    e1 = P[:, 1] - P[:, 0]
    e2 = P[:, 2] - P[:, 0]
    x1 = np.linalg.norm(e1, axis=1)
    x1s = np.where(x1 > 0, x1, 1)
    x2 = np.einsum('ij,ij->i', e2, e1) / x1s
    y2 = np.linalg.norm(np.cross(e1, e2), axis=1) / x1s
    ok = (x1 > 1e-12) & (y2 > 1e-12)
    y2s = np.where(ok, y2, 1)
    d1 = Q[:, 1] - Q[:, 0]
    d2 = Q[:, 2] - Q[:, 0]
    ja = d1 / x1s[:, None]
    jb = (d2 - d1 * (x2 / x1s)[:, None]) / y2s[:, None]
    E = (ja * ja).sum(1)
    F = (ja * jb).sum(1)
    G = (jb * jb).sum(1)
    disc = np.sqrt(np.maximum((E - G) ** 2 + 4 * F * F, 0))
    s1 = np.sqrt(np.maximum((E + G + disc) / 2, 0))
    s2 = np.sqrt(np.maximum((E + G - disc) / 2, 0))
    det = ja[:, 0] * jb[:, 1] - ja[:, 1] * jb[:, 0]
    area3 = x1 * y2 / 2
    uva = np.abs(d1[:, 0] * d2[:, 1] - d1[:, 1] * d2[:, 0]) / 2
    return dict(tl=tl, tp=tp, Q=Q, s1=s1, s2=s2, det=det, area3=area3, uva=uva, ok=ok)


def rasterize(Q, isl_t, n=1024):
    """Pixel-centre rasterisation. Returns count map, island-id map, inter-island conflict map."""
    cnt = np.zeros((n, n), np.uint8)
    ids = np.full((n, n), -1, np.int32)
    conflict = np.zeros((n, n), bool)
    q = Q * n
    mn = np.floor(q.min(1) - 0.5).astype(np.int64) + 1
    mx = np.floor(q.max(1) - 0.5).astype(np.int64)
    mn = np.clip(mn, 0, n - 1)
    mx = np.clip(mx, -1, n - 1)
    x0 = q[:, 0, 0]
    y0 = q[:, 0, 1]
    x1 = q[:, 1, 0]
    y1 = q[:, 1, 1]
    x2 = q[:, 2, 0]
    y2 = q[:, 2, 1]
    den = (y1 - y2) * (x0 - x2) + (x2 - x1) * (y0 - y2)
    ar = np.arange(n) + 0.5
    for i in np.nonzero((mx[:, 0] >= mn[:, 0]) & (mx[:, 1] >= mn[:, 1]) & (np.abs(den) > 1e-12))[0].tolist():
        ax, bx = (mn[i, 0], mx[i, 0] + 1)
        ay, by = (mn[i, 1], mx[i, 1] + 1)
        px = ar[ax:bx][None, :]
        py = ar[ay:by][:, None]
        d = den[i]
        w0 = ((y1[i] - y2[i]) * (px - x2[i]) + (x2[i] - x1[i]) * (py - y2[i])) / d
        w1 = ((y2[i] - y0[i]) * (px - x2[i]) + (x0[i] - x2[i]) * (py - y2[i])) / d
        m = (w0 >= 0) & (w1 >= 0) & (w0 + w1 <= 1)
        if not m.any():
            continue
        sub_ids = ids[ay:by, ax:bx]
        sub_cnt = cnt[ay:by, ax:bx]
        conflict[ay:by, ax:bx] |= m & (sub_ids >= 0) & (sub_ids != isl_t[i])
        sub_cnt[m] += 1
        sub_ids[m] = isl_t[i]
    return (cnt, ids, conflict)


def island_overlap_share(Q, n=192):
    """Self-overlap of ONE island: share of its covered pixels that are covered twice (island scaled to its bbox)."""
    lo = Q.reshape(-1, 2).min(0)
    hi = Q.reshape(-1, 2).max(0)
    size = max(float((hi - lo).max()), 1e-30)
    q = (Q - lo) / size * 0.98 + 0.01
    cnt, _, _ = rasterize(q, np.zeros(len(Q), np.int32), n)
    cov = int((cnt > 0).sum())
    return float((cnt > 1).sum()) / max(cov, 1)


def wpercentile(x, w, p):
    o = np.argsort(x, kind="stable")
    cw = np.cumsum(w[o])
    if cw[-1] <= 0:
        return float('nan')
    return float(x[o][min(np.searchsorted(cw, p / 100 * cw[-1]), len(x) - 1)])


def island_gap(ids, kmax=8):
    """Smallest pixel distance (Chebyshev) at which two different islands meet, up to kmax."""
    n = ids.shape[0]
    for k in range(1, kmax + 1):
        for dx, dy in ((k, 0), (0, k), (k, k), (k, -k)):
            a = ids[max(dy, 0):n + min(dy, 0), dx:]
            b = ids[max(-dy, 0):n + min(-dy, 0), :n - dx]
            if ((a >= 0) & (b >= 0) & (a != b)).any():
                return k
    return kmax + 1


def metrics(me, raster=1024, T=None, uv=None, islands=None):
    """islands: (island per face, cut per pair, ...) when the caller knows them already (same seams, other uv)."""
    T = T or Topo(me)
    if uv is None:
        uv = _uv_array(me)
    isl, cut = uv_islands(T, uv) if islands is None else islands[:2]
    ni = int(isl.max()) + 1
    td = tri_data(T, uv)
    ok = td['ok'] & (td['s2'] > 1e-12)
    w = td['area3']
    ti = isl[td['tp']]
    ia3 = np.bincount(ti, weights=td['area3'], minlength=ni)
    iuv = np.bincount(ti, weights=td['uva'], minlength=ni)
    iscale = iuv / np.maximum(ia3, 1e-30)
    gscale = iuv.sum() / ia3.sum()
    s = td['s1'] * td['s2']
    with np.errstate(divide='ignore', invalid='ignore'):
        la = np.abs(np.log2(np.where(ok, s / iscale[ti], 1)))
        lg = np.abs(np.log2(np.where(ok, s / gscale, 1)))
        conf = np.where(ok, td['s1'] / np.where(ok, td['s2'], 1), 1)
        dens = np.abs(np.log2(np.maximum(iscale, 1e-30) / gscale))
    W = w.sum()
    sign = np.sign(td['det'])
    isum = np.bincount(ti, weights=sign * w, minlength=ni)
    flipped = ok & (sign * np.sign(isum[ti]) < 0)
    fcount = np.bincount(isl, minlength=ni)
    out = dict(faces=T.nf, islands=ni, islands_tiny=int((fcount < 8).sum()), seam_share=round(float(cut.sum() / T.ne), 4), fill_sum=round(float(td['uva'].sum()), 4), angle_mean=round(float((conf * w).sum() / W), 4), angle_p95=round(wpercentile(conf, w, 95), 3), area_mean=round(float((la * w).sum() / W), 4), area_p95=round(wpercentile(la, w, 95), 3), texel_mean=round(float((lg * w).sum() / W), 4), texel_p95=round(wpercentile(lg, w, 95), 3), density_spread=round(float((dens * ia3).sum() / ia3.sum()), 4), flipped_share=round(float((w * flipped).sum() / W), 5), uv_min=round(float(uv.min()), 4), uv_max=round(float(uv.max()), 4))
    extra = dict(isl=isl, ti=ti, td=td, conf=conf, la=la, T=T, cut=cut)
    if raster:
        cnt, ids, conflict = rasterize(td['Q'], ti, raster)
        cov = cnt > 0
        nc = max(int(cov.sum()), 1)
        out['fill'] = round(float(cov.mean()), 4)
        out['overlap_inter'] = round(float(conflict.sum() / nc), 5)
        out['overlap_self'] = round(float(((cnt > 1) & ~conflict).sum() / nc), 5)
        out['gap_px'] = island_gap(ids)
        extra.update(cnt=cnt, ids=ids, conflict=conflict)
    return (out, extra)


def island_report(extra):
    """Per-island quality after an unwrap: area-weighted mean angle distortion (sigma1 / sigma2), mean area
    distortion (|log2|), flipped share of the area, UV area, and `far`: the share of triangles whose area is off
    by more than 2x."""
    td = extra['td']
    ti = extra['ti']
    ni = int(ti.max()) + 1
    w = td['area3']
    a3 = np.bincount(ti, weights=w, minlength=ni)
    a3s = np.maximum(a3, 1e-30)
    ang = np.bincount(ti, weights=extra['conf'] * w, minlength=ni) / a3s
    are = np.bincount(ti, weights=extra['la'] * w, minlength=ni) / a3s
    sign = np.sign(td['det'])
    isum = np.bincount(ti, weights=sign * w, minlength=ni)
    fl = (sign * np.sign(isum[ti]) < 0) | (td['det'] == 0)
    flip = np.bincount(ti, weights=fl * w, minlength=ni) / a3s
    uva = np.bincount(ti, weights=td['uva'], minlength=ni)
    far = np.bincount(ti, weights=extra['la'] > 1, minlength=ni) / np.maximum(np.bincount(ti, minlength=ni), 1)
    return dict(area3=a3, angle=ang, area=are, flip=flip, uvarea=uva, far=far)
