"""
Efficient computation of

    s(X, Y) = (I(X) + I(Y) - I(X union Y)) / I(X union Y)

where I(Z) = int_0^R C_r(Z) dr and R is the first radius at which both
G_R(X) and G_R(Y) are connected.

The core observation is that C_r is determined by the Euclidean MST /
single-linkage tree.  If R is at least the connectivity threshold of Z,

    I(Z) = R + total_weight(MST(Z)).

For X,Y differing by at most one deletion and one insertion, write
A = X cap Y, X = A + x, Y = A + y.  Compute MST(A) once, then get MST(X),
MST(Y), and MST(A+x+y) from MST(A) plus only the star edges incident to the
new point(s).  This is exact if MST(A) is exact.

Preferred high-level API for streaming/sliding-window data:

    pointcloud_similarity_indices(points, indices_X, indices_Y, ...)
    pointcloud_similarity_index_sets(points, indices_X, indices_Y, ...)  # alias

where X = points[indices_X] and Y = points[indices_Y].  A two-array
compatibility API is also provided:

    pointcloud_similarity_one_swap(X, Y, ...)

Dependencies: numpy, scipy.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Callable, Optional, Sequence, Tuple

import numpy as np


@dataclass
class MSTResult:
    """A minimum spanning tree represented by parallel edge arrays."""

    n: int
    u: np.ndarray  # int64, shape (n_edges,)
    v: np.ndarray  # int64, shape (n_edges,)
    w: np.ndarray  # float64, shape (n_edges,)

    @property
    def total_weight(self) -> float:
        return float(np.sum(self.w, dtype=np.float64))

    @property
    def max_edge(self) -> float:
        return float(np.max(self.w)) if self.w.size else 0.0


@dataclass
class SimilarityResult:
    """Result object returned by the high-level similarity routines."""

    s: float
    R: float
    I_X: float
    I_Y: float
    I_union: float
    mst_X: MSTResult
    mst_Y: MSTResult
    mst_union: MSTResult
    mst_common: MSTResult
    n_common: int
    x_unique: Optional[np.ndarray]
    y_unique: Optional[np.ndarray]
    common_indices: Optional[np.ndarray] = None
    x_unique_index: Optional[int] = None
    y_unique_index: Optional[int] = None
    indices_X: Optional[np.ndarray] = None
    indices_Y: Optional[np.ndarray] = None


class UnionFind:
    """Union-find with path compression and union by rank."""

    __slots__ = ("parent", "rank", "count")

    def __init__(self, n: int):
        self.parent = np.arange(n, dtype=np.int64)
        self.rank = np.zeros(n, dtype=np.uint8)
        self.count = int(n)

    def find(self, x: int) -> int:
        parent = self.parent
        x = int(x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = int(parent[x])
        return x

    def union(self, a: int, b: int) -> bool:
        ra = self.find(a)
        rb = self.find(b)
        if ra == rb:
            return False
        rank = self.rank
        parent = self.parent
        if rank[ra] < rank[rb]:
            ra, rb = rb, ra
        parent[rb] = ra
        if rank[ra] == rank[rb]:
            rank[ra] += 1
        self.count -= 1
        return True

    def labels(self) -> np.ndarray:
        """Return root label for every item and compress paths."""
        parent = self.parent
        roots = np.arange(parent.size, dtype=np.int64)
        while True:
            new_roots = parent[roots]
            if np.array_equal(new_roots, roots):
                break
            roots = new_roots
        parent[:] = roots
        return roots.copy()


def _as_points(points: np.ndarray, name: str = "points") -> np.ndarray:
    pts = np.asarray(points, dtype=np.float64)
    if pts.ndim != 2:
        raise ValueError(f"{name} must have shape (n_points, dimension)")
    if not np.all(np.isfinite(pts)):
        raise ValueError(f"{name} contains NaN or inf")
    return np.ascontiguousarray(pts)


def _empty_mst(n: int) -> MSTResult:
    return MSTResult(
        int(n),
        np.empty(0, dtype=np.int64),
        np.empty(0, dtype=np.int64),
        np.empty(0, dtype=np.float64),
    )


def kruskal_mst(
    n: int,
    u: Sequence[int] | np.ndarray,
    v: Sequence[int] | np.ndarray,
    w: Sequence[float] | np.ndarray,
    *,
    assume_sorted: bool = False,
) -> MSTResult:
    """Compute an MST of a sparse candidate graph using Kruskal's algorithm."""
    n = int(n)
    if n <= 1:
        return _empty_mst(n)

    u_arr = np.asarray(u, dtype=np.int64)
    v_arr = np.asarray(v, dtype=np.int64)
    w_arr = np.asarray(w, dtype=np.float64)
    if not (u_arr.shape == v_arr.shape == w_arr.shape):
        raise ValueError("u, v, and w must have the same shape")
    if u_arr.ndim != 1:
        raise ValueError("u, v, and w must be one-dimensional")

    order = np.arange(w_arr.size) if assume_sorted else np.argsort(w_arr, kind="mergesort")
    uf = UnionFind(n)
    target_edges = n - 1
    out_u = np.empty(target_edges, dtype=np.int64)
    out_v = np.empty(target_edges, dtype=np.int64)
    out_w = np.empty(target_edges, dtype=np.float64)
    k = 0

    for e in order:
        a = int(u_arr[e])
        b = int(v_arr[e])
        if a < 0 or a >= n or b < 0 or b >= n:
            raise ValueError("edge endpoint out of range")
        if uf.union(a, b):
            out_u[k] = a
            out_v[k] = b
            out_w[k] = w_arr[e]
            k += 1
            if k == target_edges:
                break

    if k != target_edges:
        raise ValueError("candidate graph is disconnected; cannot form an MST")

    return MSTResult(n, out_u, out_v, out_w)


def _ckdtree_query(tree, query_points: np.ndarray, k: int, eps: float, workers: int):
    """Compatibility wrapper for scipy versions with/without workers=."""
    try:
        return tree.query(query_points, k=k, eps=eps, workers=workers)
    except TypeError:  # pragma: no cover - old scipy compatibility
        return tree.query(query_points, k=k, eps=eps)


def emst_boruvka_ckdtree(
    points: np.ndarray,
    *,
    initial_k: int = 8,
    leafsize: int = 40,
    eps: float = 0.0,
    workers: int = -1,
    max_query_entries: int = 2_000_000,
) -> MSTResult:
    """
    Euclidean MST via Boruvka phases and cKDTree kNN queries.

    With eps=0.0, cKDTree nearest-neighbor queries are exact.  The method avoids
    O(n^2) pairwise distances and is typically near-linear for low/moderate
    dimensional geometric data, but KD-tree methods can degrade in high
    dimension or on adversarial inputs.  Setting eps>0 trades exactness for
    approximate nearest-neighbor speed.
    """
    pts = _as_points(points)
    n = pts.shape[0]
    if n <= 1:
        return _empty_mst(n)
    if initial_k < 2:
        initial_k = 2
    if max_query_entries < initial_k:
        max_query_entries = initial_k

    try:
        from scipy.spatial import cKDTree
    except Exception as exc:  # pragma: no cover
        raise ImportError("emst_boruvka_ckdtree requires scipy") from exc

    tree = cKDTree(pts, leafsize=leafsize)
    uf = UnionFind(n)
    out_u: list[int] = []
    out_v: list[int] = []
    out_w: list[float] = []
    all_indices = np.arange(n, dtype=np.int64)

    while uf.count > 1:
        labels = uf.labels()
        nearest_j = np.full(n, -1, dtype=np.int64)
        nearest_d = np.full(n, np.inf, dtype=np.float64)

        unresolved = all_indices
        k_query = min(max(initial_k, 2), n)

        while unresolved.size:
            k_eff = min(k_query, n)
            chunk_size = max(1, int(max_query_entries // max(k_eff, 1)))
            still_unresolved_chunks = []

            for start in range(0, unresolved.size, chunk_size):
                src = unresolved[start : start + chunk_size]
                dists, idxs = _ckdtree_query(tree, pts[src], k_eff, eps, workers)
                if k_eff == 1:
                    dists = np.asarray(dists)[:, None]
                    idxs = np.asarray(idxs)[:, None]
                else:
                    dists = np.asarray(dists)
                    idxs = np.asarray(idxs)

                outside = labels[idxs] != labels[src][:, None]
                found = np.any(outside, axis=1)

                if np.any(found):
                    rows = np.flatnonzero(found)
                    cols = np.argmax(outside[rows], axis=1)
                    src_found = src[rows]
                    nearest_j[src_found] = idxs[rows, cols]
                    nearest_d[src_found] = dists[rows, cols]

                if not np.all(found):
                    still_unresolved_chunks.append(src[~found])

            if not still_unresolved_chunks:
                break

            unresolved = np.concatenate(still_unresolved_chunks)
            if k_eff >= n:
                raise RuntimeError("failed to find outgoing edges between components")
            k_query = min(2 * k_eff, n)

        valid = nearest_j >= 0
        if not np.any(valid):
            raise RuntimeError("Boruvka phase found no outgoing edges")

        srcs = all_indices[valid]
        dsts = nearest_j[valid]
        ds = nearest_d[valid]
        roots = labels[srcs]

        # Pick the lightest outgoing edge for each component.  Lexicographic
        # tie-breaking makes results deterministic.
        order = np.lexsort((dsts, srcs, ds, roots))
        roots_sorted = roots[order]
        first = np.r_[True, roots_sorted[1:] != roots_sorted[:-1]]
        chosen = order[first]

        added_this_phase = 0
        for pos in chosen:
            a = int(srcs[pos])
            b = int(dsts[pos])
            if uf.union(a, b):
                out_u.append(a)
                out_v.append(b)
                out_w.append(float(ds[pos]))
                added_this_phase += 1
                if len(out_w) == n - 1:
                    break

        if added_this_phase == 0:
            raise RuntimeError("Boruvka phase did not merge any components")

    return MSTResult(
        n,
        np.asarray(out_u, dtype=np.int64),
        np.asarray(out_v, dtype=np.int64),
        np.asarray(out_w, dtype=np.float64),
    )


def emst_delaunay(points: np.ndarray, *, qhull_options: Optional[str] = None) -> MSTResult:
    """
    Exact Euclidean MST from the Delaunay graph.

    Best for low-dimensional data, especially 2D.  In high dimensions the
    Delaunay graph can be much larger than linear or Qhull may fail on
    degeneracies.  Duplicate rows are handled by the Boruvka routine instead.
    """
    pts = _as_points(points)
    n, d = pts.shape
    if n <= 1:
        return _empty_mst(n)

    if d == 1:
        order = np.argsort(pts[:, 0], kind="mergesort")
        u = order[:-1]
        v = order[1:]
        w = np.abs(pts[v, 0] - pts[u, 0])
        return kruskal_mst(n, u, v, w, assume_sorted=True)

    # For tiny clouds, the complete graph is still small and exact.
    if n <= d + 1:
        us = []
        vs = []
        ws = []
        for i in range(n):
            for j in range(i + 1, n):
                us.append(i)
                vs.append(j)
                ws.append(float(np.linalg.norm(pts[i] - pts[j])))
        return kruskal_mst(n, us, vs, ws)

    # Qhull can be awkward with duplicate rows; cKDTree Boruvka handles them.
    if np.unique(pts, axis=0).shape[0] != n:
        return emst_boruvka_ckdtree(pts)

    try:
        from scipy.spatial import Delaunay
    except Exception as exc:  # pragma: no cover
        raise ImportError("emst_delaunay requires scipy") from exc

    tri = Delaunay(pts, qhull_options=qhull_options)
    simplices = np.asarray(tri.simplices, dtype=np.int64)
    if simplices.size == 0:
        return emst_boruvka_ckdtree(pts)

    edges = set()
    simplex_width = simplices.shape[1]
    for simplex in simplices:
        for a_pos in range(simplex_width):
            a = int(simplex[a_pos])
            for b_pos in range(a_pos + 1, simplex_width):
                b = int(simplex[b_pos])
                edges.add((a, b) if a < b else (b, a))

    if not edges:
        return emst_boruvka_ckdtree(pts)

    edge_arr = np.asarray(list(edges), dtype=np.int64)
    u = edge_arr[:, 0]
    v = edge_arr[:, 1]
    w = np.linalg.norm(pts[u] - pts[v], axis=1)
    return kruskal_mst(n, u, v, w)


def emst(
    points: np.ndarray,
    *,
    method: str = "auto",
    delaunay_max_dim: int = 3,
    **kwargs,
) -> MSTResult:
    """
    Convenience Euclidean MST builder.

    method="auto": use Delaunay for dimension <= delaunay_max_dim, otherwise
    use cKDTree Boruvka.
    method="delaunay": force Delaunay.
    method="boruvka": force cKDTree Boruvka.
    """
    pts = _as_points(points)
    method = method.lower()
    if method == "auto":
        if pts.shape[1] <= delaunay_max_dim:
            try:
                return emst_delaunay(
                    pts,
                    **{k: v for k, v in kwargs.items() if k == "qhull_options"},
                )
            except Exception:
                # Robust fallback for degenerate low-dimensional inputs.
                pass
        return emst_boruvka_ckdtree(
            pts,
            **{k: v for k, v in kwargs.items() if k in {
                "initial_k", "leafsize", "eps", "workers", "max_query_entries"
            }},
        )
    if method == "delaunay":
        return emst_delaunay(pts, **kwargs)
    if method == "boruvka":
        return emst_boruvka_ckdtree(pts, **kwargs)
    raise ValueError('method must be "auto", "delaunay", or "boruvka"')


def mst_with_inserted_points(
    common_points: np.ndarray,
    common_mst: MSTResult,
    inserted_points: Optional[np.ndarray],
) -> MSTResult:
    """
    Compute MST(A plus inserted points) from MST(A) plus inserted star edges.

    For k inserted points this uses candidate edges:
      - all edges of MST(A)
      - every edge from each inserted point to every point in A
      - every edge between inserted points

    For the target use case k <= 2, so this is O(n d + n log n) after MST(A).
    """
    A = _as_points(common_points, "common_points")
    if inserted_points is None:
        inserted = np.empty((0, A.shape[1]), dtype=np.float64)
    else:
        inserted = np.asarray(inserted_points, dtype=np.float64)
        if inserted.ndim == 1:
            inserted = inserted[None, :]
        inserted = _as_points(inserted, "inserted_points")
        if inserted.shape[1] != A.shape[1]:
            raise ValueError("inserted_points has the wrong dimension")

    nA = A.shape[0]
    k = inserted.shape[0]
    n = nA + k
    if n <= 1:
        return _empty_mst(n)
    if common_mst.n != nA:
        raise ValueError("common_mst.n does not match len(common_points)")
    if common_mst.w.size != max(0, nA - 1):
        raise ValueError("common_mst does not have n_common - 1 edges")

    u_parts = [np.asarray(common_mst.u, dtype=np.int64)]
    v_parts = [np.asarray(common_mst.v, dtype=np.int64)]
    w_parts = [np.asarray(common_mst.w, dtype=np.float64)]

    if nA > 0 and k > 0:
        base_idx = np.arange(nA, dtype=np.int64)
        for t, p in enumerate(inserted):
            new_idx = nA + t
            u_parts.append(np.full(nA, new_idx, dtype=np.int64))
            v_parts.append(base_idx)
            w_parts.append(np.linalg.norm(A - p, axis=1))

    if k > 1:
        us = []
        vs = []
        ws = []
        for a in range(k):
            for b in range(a + 1, k):
                us.append(nA + a)
                vs.append(nA + b)
                ws.append(float(np.linalg.norm(inserted[a] - inserted[b])))
        u_parts.append(np.asarray(us, dtype=np.int64))
        v_parts.append(np.asarray(vs, dtype=np.int64))
        w_parts.append(np.asarray(ws, dtype=np.float64))

    u = np.concatenate(u_parts) if u_parts else np.empty(0, dtype=np.int64)
    v = np.concatenate(v_parts) if v_parts else np.empty(0, dtype=np.int64)
    w = np.concatenate(w_parts) if w_parts else np.empty(0, dtype=np.float64)
    return kruskal_mst(n, u, v, w)


def integral_from_mst(mst: MSTResult, R: float) -> float:
    """Compute int_0^R C_r dr from MST merge radii."""
    R = float(R)
    if R < 0:
        raise ValueError("R must be nonnegative")
    if mst.n == 0:
        return 0.0
    weights = np.asarray(mst.w, dtype=np.float64)
    return float(mst.n * R - np.maximum(0.0, R - weights).sum(dtype=np.float64))


def component_counts_from_mst(mst: MSTResult, radii: Sequence[float] | np.ndarray) -> np.ndarray:
    """Evaluate C_r for many radii from MST/single-linkage merge radii."""
    r = np.asarray(radii, dtype=np.float64)
    if np.any(r < 0):
        raise ValueError("radii must be nonnegative")
    weights = np.sort(np.asarray(mst.w, dtype=np.float64))
    return mst.n - np.searchsorted(weights, r, side="right")


def one_swap_msts_from_common(
    common_points: np.ndarray,
    common_mst: MSTResult,
    x: Optional[np.ndarray],
    y: Optional[np.ndarray],
) -> Tuple[MSTResult, MSTResult, MSTResult]:
    """
    Compute MST(X), MST(Y), MST(X union Y) in one sorted Kruskal pass.

    Here X = A plus optional x and Y = A plus optional y.  The candidate edge
    set is MST(A), the star edges from inserted points to A, and the x-y edge.
    The same sorted candidate stream is fed to three union-finds, avoiding three
    separate sorts and avoiding repeated distance computation.
    """
    A = _as_points(common_points, "common_points")
    nA, dim = A.shape
    if common_mst.n != nA:
        raise ValueError("common_mst.n does not match len(common_points)")
    if common_mst.w.size != max(0, nA - 1):
        raise ValueError("common_mst does not have n_common - 1 edges")

    x_arr = None if x is None else np.asarray(x, dtype=np.float64).reshape(-1)
    y_arr = None if y is None else np.asarray(y, dtype=np.float64).reshape(-1)
    if x_arr is not None and x_arr.shape[0] != dim:
        raise ValueError("x has the wrong dimension")
    if y_arr is not None and y_arr.shape[0] != dim:
        raise ValueError("y has the wrong dimension")

    has_x = x_arr is not None
    has_y = y_arr is not None
    nX = nA + int(has_x)
    nY = nA + int(has_y)
    nU = nA + int(has_x) + int(has_y)
    xU = nA if has_x else -1
    yU = nA + int(has_x) if has_y else -1

    # Edge kind codes: 0=base A-A, 1=x-A, 2=y-A, 3=x-y.
    kind_parts = []
    a_parts = []
    b_parts = []
    w_parts = []

    if common_mst.w.size:
        m = common_mst.w.size
        kind_parts.append(np.zeros(m, dtype=np.int8))
        a_parts.append(np.asarray(common_mst.u, dtype=np.int64))
        b_parts.append(np.asarray(common_mst.v, dtype=np.int64))
        w_parts.append(np.asarray(common_mst.w, dtype=np.float64))

    if has_x and nA:
        idx = np.arange(nA, dtype=np.int64)
        kind_parts.append(np.full(nA, 1, dtype=np.int8))
        a_parts.append(idx)
        b_parts.append(np.full(nA, -1, dtype=np.int64))
        w_parts.append(np.linalg.norm(A - x_arr, axis=1))

    if has_y and nA:
        idx = np.arange(nA, dtype=np.int64)
        kind_parts.append(np.full(nA, 2, dtype=np.int8))
        a_parts.append(idx)
        b_parts.append(np.full(nA, -1, dtype=np.int64))
        w_parts.append(np.linalg.norm(A - y_arr, axis=1))

    if has_x and has_y:
        kind_parts.append(np.asarray([3], dtype=np.int8))
        a_parts.append(np.asarray([-1], dtype=np.int64))
        b_parts.append(np.asarray([-1], dtype=np.int64))
        w_parts.append(np.asarray([float(np.linalg.norm(x_arr - y_arr))], dtype=np.float64))

    if w_parts:
        kinds = np.concatenate(kind_parts)
        aa = np.concatenate(a_parts)
        bb = np.concatenate(b_parts)
        ww = np.concatenate(w_parts)
        order = np.argsort(ww, kind="mergesort")
    else:
        kinds = np.empty(0, dtype=np.int8)
        aa = np.empty(0, dtype=np.int64)
        bb = np.empty(0, dtype=np.int64)
        ww = np.empty(0, dtype=np.float64)
        order = np.empty(0, dtype=np.int64)

    ufX = UnionFind(nX)
    ufY = UnionFind(nY)
    ufU = UnionFind(nU)

    targetX = max(0, nX - 1)
    targetY = max(0, nY - 1)
    targetU = max(0, nU - 1)

    outXu: list[int] = []
    outXv: list[int] = []
    outXw: list[float] = []
    outYu: list[int] = []
    outYv: list[int] = []
    outYw: list[float] = []
    outUu: list[int] = []
    outUv: list[int] = []
    outUw: list[float] = []

    def add_edge(
        uf: UnionFind,
        ou: list[int],
        ov: list[int],
        ow: list[float],
        target: int,
        p: int,
        q: int,
        weight: float,
    ) -> None:
        if len(ow) < target and uf.union(p, q):
            ou.append(int(p))
            ov.append(int(q))
            ow.append(float(weight))

    for e in order:
        weight = float(ww[e])
        kind = int(kinds[e])
        a = int(aa[e])
        b = int(bb[e])

        if kind == 0:
            add_edge(ufX, outXu, outXv, outXw, targetX, a, b, weight)
            add_edge(ufY, outYu, outYv, outYw, targetY, a, b, weight)
            add_edge(ufU, outUu, outUv, outUw, targetU, a, b, weight)
        elif kind == 1:
            add_edge(ufX, outXu, outXv, outXw, targetX, nA, a, weight)
            add_edge(ufU, outUu, outUv, outUw, targetU, xU, a, weight)
        elif kind == 2:
            add_edge(ufY, outYu, outYv, outYw, targetY, nA, a, weight)
            add_edge(ufU, outUu, outUv, outUw, targetU, yU, a, weight)
        elif kind == 3:
            add_edge(ufU, outUu, outUv, outUw, targetU, xU, yU, weight)
        else:  # pragma: no cover
            raise RuntimeError("unknown edge kind")

        if len(outXw) == targetX and len(outYw) == targetY and len(outUw) == targetU:
            break

    if len(outXw) != targetX or len(outYw) != targetY or len(outUw) != targetU:
        raise ValueError("candidate graph did not connect X, Y, or their union")

    mst_X = MSTResult(nX, np.asarray(outXu, dtype=np.int64), np.asarray(outXv, dtype=np.int64), np.asarray(outXw, dtype=np.float64))
    mst_Y = MSTResult(nY, np.asarray(outYu, dtype=np.int64), np.asarray(outYv, dtype=np.int64), np.asarray(outYw, dtype=np.float64))
    mst_U = MSTResult(nU, np.asarray(outUu, dtype=np.int64), np.asarray(outUv, dtype=np.int64), np.asarray(outUw, dtype=np.float64))
    return mst_X, mst_Y, mst_U


def _row_key(row: np.ndarray) -> tuple:
    # Exact multiset key.  Prefer index identities or ids for floating-point data.
    return tuple(np.asarray(row).tolist())


def split_clouds_one_swap(
    X: np.ndarray,
    Y: np.ndarray,
    *,
    ids_X: Optional[Sequence[object]] = None,
    ids_Y: Optional[Sequence[object]] = None,
    check_common_coordinates: bool = True,
) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Return A, x, y with X = A plus optional x and Y = A plus optional y.

    If ids_X/ids_Y are supplied, identity is based on ids.  Otherwise rows are
    matched exactly as a multiset; this is fine for integer/exact coordinates,
    but the index-set API is preferred for floating point data stored once.
    """
    Xp = _as_points(X, "X")
    Yp = _as_points(Y, "Y")
    if Xp.shape[1] != Yp.shape[1]:
        raise ValueError("X and Y must have the same ambient dimension")

    if (ids_X is None) != (ids_Y is None):
        raise ValueError("provide both ids_X and ids_Y, or neither")

    if ids_X is not None and ids_Y is not None:
        if len(ids_X) != len(Xp) or len(ids_Y) != len(Yp):
            raise ValueError("ids_X/ids_Y must have lengths matching X/Y")
        pos_X = {}
        pos_Y = {}
        for i, key in enumerate(ids_X):
            if key in pos_X:
                raise ValueError("duplicate id in ids_X")
            pos_X[key] = i
        for i, key in enumerate(ids_Y):
            if key in pos_Y:
                raise ValueError("duplicate id in ids_Y")
            pos_Y[key] = i

        keys_X = set(pos_X)
        keys_Y = set(pos_Y)
        only_X = [key for key in ids_X if key not in keys_Y]
        only_Y = [key for key in ids_Y if key not in keys_X]
        if len(only_X) > 1 or len(only_Y) > 1:
            raise ValueError("X and Y differ by more than one deletion/insertion")

        common_keys = [key for key in ids_X if key in keys_Y]
        common_X_idx = [pos_X[key] for key in common_keys]
        common_Y_idx = [pos_Y[key] for key in common_keys]
        A = Xp[common_X_idx].copy()
        if check_common_coordinates and common_keys:
            if not np.allclose(A, Yp[common_Y_idx], rtol=0.0, atol=0.0):
                raise ValueError(
                    "common ids have different coordinates; this code assumes "
                    "delete/insert changes, not point motion"
                )
        x = Xp[pos_X[only_X[0]]].copy() if only_X else None
        y = Yp[pos_Y[only_Y[0]]].copy() if only_Y else None
        return A, x, y

    counts_Y = Counter(_row_key(row) for row in Yp)
    common_rows = []
    x_rows = []
    for row in Xp:
        key = _row_key(row)
        if counts_Y[key] > 0:
            common_rows.append(row.copy())
            counts_Y[key] -= 1
        else:
            x_rows.append(row.copy())

    y_rows = []
    for row in Yp:
        key = _row_key(row)
        if counts_Y[key] > 0:
            y_rows.append(row.copy())
            counts_Y[key] -= 1

    if len(x_rows) > 1 or len(y_rows) > 1:
        raise ValueError(
            "X and Y differ by more than one deletion/insertion under exact row matching; "
            "provide stable ids_X/ids_Y or use pointcloud_similarity_indices"
        )

    if common_rows:
        A = np.vstack(common_rows).astype(np.float64, copy=False)
    else:
        A = np.empty((0, Xp.shape[1]), dtype=np.float64)
    x = x_rows[0] if x_rows else None
    y = y_rows[0] if y_rows else None
    return A, x, y


def _normalize_indices(indices: Sequence[int] | np.ndarray, n_points: int, name: str) -> np.ndarray:
    """
    Normalize an index set to a one-dimensional int64 array.

    Integer arrays preserve their order.  Boolean masks are converted to
    increasing integer indices.  Negative indices are rejected deliberately,
    because these are set identities rather than Python slicing positions.
    """
    arr = np.asarray(indices)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional integer index array or boolean mask")
    if arr.size == 0:
        return np.empty(0, dtype=np.int64)

    if arr.dtype == np.bool_:
        if arr.size != n_points:
            raise ValueError(f"boolean mask {name} must have length len(points)")
        idx = np.flatnonzero(arr).astype(np.int64, copy=False)
    else:
        if not np.issubdtype(arr.dtype, np.integer):
            raise TypeError(f"{name} must contain integer indices, or be a boolean mask")
        idx = arr.astype(np.int64, copy=False)

    if np.any(idx < 0) or np.any(idx >= n_points):
        raise IndexError(f"{name} contains an index outside [0, len(points))")

    seen = np.zeros(n_points, dtype=np.bool_)
    if idx.size:
        seen[idx] = True
        if int(seen.sum()) != idx.size:
            raise ValueError(f"{name} contains duplicate indices")

    return np.ascontiguousarray(idx, dtype=np.int64)


def split_indices_one_swap(
    points: np.ndarray,
    indices_X: Sequence[int] | np.ndarray,
    indices_Y: Sequence[int] | np.ndarray,
) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray], np.ndarray, Optional[int], Optional[int], np.ndarray, np.ndarray]:
    """
    Return A, x, y and the corresponding global indices for two index sets.

    X = points[indices_X], Y = points[indices_Y].  The sets must differ by at
    most one deletion and at most one insertion.  Common points are ordered as
    they appear in indices_X.

    Returns
    -------
    common_points, x, y, common_indices, x_index, y_index, indices_X, indices_Y
    """
    pts = _as_points(points, "points")
    idx_X = _normalize_indices(indices_X, len(pts), "indices_X")
    idx_Y = _normalize_indices(indices_Y, len(pts), "indices_Y")

    in_X = np.zeros(len(pts), dtype=np.bool_)
    in_Y = np.zeros(len(pts), dtype=np.bool_)
    in_X[idx_X] = True
    in_Y[idx_Y] = True

    common_indices = idx_X[in_Y[idx_X]]
    only_X = idx_X[~in_Y[idx_X]]
    only_Y = idx_Y[~in_X[idx_Y]]

    if only_X.size > 1 or only_Y.size > 1:
        raise ValueError("indices_X and indices_Y differ by more than one deletion/insertion")

    common_points = pts[common_indices].copy()
    x = pts[int(only_X[0])].copy() if only_X.size else None
    y = pts[int(only_Y[0])].copy() if only_Y.size else None
    x_idx = int(only_X[0]) if only_X.size else None
    y_idx = int(only_Y[0]) if only_Y.size else None
    return common_points, x, y, common_indices.copy(), x_idx, y_idx, idx_X.copy(), idx_Y.copy()


def _similarity_from_common_split(
    A: np.ndarray,
    x: Optional[np.ndarray],
    y: Optional[np.ndarray],
    *,
    mst_builder: Optional[Callable[[np.ndarray], MSTResult]] = None,
    mst_method: str = "auto",
    common_indices: Optional[np.ndarray] = None,
    x_unique_index: Optional[int] = None,
    y_unique_index: Optional[int] = None,
    indices_X: Optional[np.ndarray] = None,
    indices_Y: Optional[np.ndarray] = None,
    **mst_kwargs,
) -> SimilarityResult:
    """Shared implementation once the common cloud and optional uniques are known."""
    A = _as_points(A, "common_points")
    if mst_builder is None:
        common_mst = emst(A, method=mst_method, **mst_kwargs)
    else:
        common_mst = mst_builder(A)
        if not isinstance(common_mst, MSTResult):
            raise TypeError("mst_builder must return an MSTResult")

    mst_X, mst_Y, mst_union = one_swap_msts_from_common(A, common_mst, x, y)

    R = max(mst_X.max_edge, mst_Y.max_edge)
    I_X = integral_from_mst(mst_X, R)
    I_Y = integral_from_mst(mst_Y, R)
    I_union = integral_from_mst(mst_union, R)
    s = float("nan") if I_union == 0.0 else (I_X + I_Y - I_union) / I_union

    return SimilarityResult(
        s=float(s),
        R=float(R),
        I_X=float(I_X),
        I_Y=float(I_Y),
        I_union=float(I_union),
        mst_X=mst_X,
        mst_Y=mst_Y,
        mst_union=mst_union,
        mst_common=common_mst,
        n_common=A.shape[0],
        x_unique=None if x is None else np.asarray(x, dtype=np.float64).copy(),
        y_unique=None if y is None else np.asarray(y, dtype=np.float64).copy(),
        common_indices=None if common_indices is None else np.asarray(common_indices, dtype=np.int64).copy(),
        x_unique_index=None if x_unique_index is None else int(x_unique_index),
        y_unique_index=None if y_unique_index is None else int(y_unique_index),
        indices_X=None if indices_X is None else np.asarray(indices_X, dtype=np.int64).copy(),
        indices_Y=None if indices_Y is None else np.asarray(indices_Y, dtype=np.int64).copy(),
    )


def pointcloud_similarity_indices(
    points: np.ndarray,
    indices_X: Sequence[int] | np.ndarray,
    indices_Y: Sequence[int] | np.ndarray,
    *,
    mst_builder: Optional[Callable[[np.ndarray], MSTResult]] = None,
    mst_method: str = "auto",
    **mst_kwargs,
) -> SimilarityResult:
    """
    Compute s(X,Y) from one shared point array and two included-index sets.

    This is the preferred API for streaming/sliding-window data.  For example,
    if points[t] is the sample at time t, consecutive windows can be compared as

        pointcloud_similarity_indices(points, np.arange(t, t+w), np.arange(t+1, t+w+1))

    The two index sets must differ by at most one removed index and at most one
    added index.  The computation is exact when the supplied/common-cloud MST
    builder is exact.
    """
    pts = _as_points(points, "points")
    A, x, y, common_idx, x_idx, y_idx, idx_X, idx_Y = split_indices_one_swap(
        pts, indices_X, indices_Y
    )
    return _similarity_from_common_split(
        A,
        x,
        y,
        mst_builder=mst_builder,
        mst_method=mst_method,
        common_indices=common_idx,
        x_unique_index=x_idx,
        y_unique_index=y_idx,
        indices_X=idx_X,
        indices_Y=idx_Y,
        **mst_kwargs,
    )


def _normalize_optional_index(index: Optional[int], n_points: int, name: str) -> Optional[int]:
    """Normalize one optional global index."""
    if index is None:
        return None
    arr = np.asarray(index)
    if arr.shape != () or not np.issubdtype(arr.dtype, np.integer):
        raise TypeError(f"{name} must be None or a single integer index")
    idx = int(arr)
    if idx < 0 or idx >= n_points:
        raise IndexError(f"{name} is outside [0, len(points))")
    return idx


def pointcloud_similarity_from_common_indices(
    points: np.ndarray,
    common_indices: Sequence[int] | np.ndarray,
    x_index: Optional[int] = None,
    y_index: Optional[int] = None,
    *,
    mst_builder: Optional[Callable[[np.ndarray], MSTResult]] = None,
    mst_method: str = "auto",
    **mst_kwargs,
) -> SimilarityResult:
    """
    Lower-level shared-array API for a known split X=A+x, Y=A+y.

    This avoids recomputing the index-set intersection.  It is useful in
    sliding-window loops, where common_indices is the overlap, x_index is the
    leaving point, and y_index is the entering point.
    """
    pts = _as_points(points, "points")
    common_idx = _normalize_indices(common_indices, len(pts), "common_indices")
    x_idx = _normalize_optional_index(x_index, len(pts), "x_index")
    y_idx = _normalize_optional_index(y_index, len(pts), "y_index")

    in_common = np.zeros(len(pts), dtype=np.bool_)
    in_common[common_idx] = True
    if x_idx is not None and in_common[x_idx]:
        raise ValueError("x_index is already in common_indices")
    if y_idx is not None and in_common[y_idx]:
        raise ValueError("y_index is already in common_indices")
    if x_idx is not None and y_idx is not None and x_idx == y_idx:
        raise ValueError("x_index and y_index must be distinct")

    A = pts[common_idx].copy()
    x = pts[x_idx].copy() if x_idx is not None else None
    y = pts[y_idx].copy() if y_idx is not None else None
    idx_X = common_idx.copy() if x_idx is None else np.r_[common_idx, np.asarray([x_idx], dtype=np.int64)]
    idx_Y = common_idx.copy() if y_idx is None else np.r_[common_idx, np.asarray([y_idx], dtype=np.int64)]

    return _similarity_from_common_split(
        A,
        x,
        y,
        mst_builder=mst_builder,
        mst_method=mst_method,
        common_indices=common_idx,
        x_unique_index=x_idx,
        y_unique_index=y_idx,
        indices_X=idx_X,
        indices_Y=idx_Y,
        **mst_kwargs,
    )

def pointcloud_similarity_one_swap(
    X: np.ndarray,
    Y: np.ndarray,
    *,
    ids_X: Optional[Sequence[object]] = None,
    ids_Y: Optional[Sequence[object]] = None,
    mst_builder: Optional[Callable[[np.ndarray], MSTResult]] = None,
    mst_method: str = "auto",
    **mst_kwargs,
) -> SimilarityResult:
    """
    Compute s(X,Y) for two separate point arrays differing by one swap.

    Prefer pointcloud_similarity_indices() when X and Y are subsets of one
    shared global point array.  This compatibility wrapper still supports two
    separate arrays, using ids_X/ids_Y when supplied and exact row matching
    otherwise.
    """
    Xp = _as_points(X, "X")
    Yp = _as_points(Y, "Y")
    A, x, y = split_clouds_one_swap(Xp, Yp, ids_X=ids_X, ids_Y=ids_Y)
    return _similarity_from_common_split(
        A,
        x,
        y,
        mst_builder=mst_builder,
        mst_method=mst_method,
        **mst_kwargs,
    )


# Index-set API aliases.
pointcloud_similarity_index_sets = pointcloud_similarity_indices
pointcloud_similarity_from_indices = pointcloud_similarity_indices
pointcloud_similarity_one_swap_indices = pointcloud_similarity_indices
similarity_index_sets = pointcloud_similarity_indices
similarity_indices = pointcloud_similarity_indices

# Backwards-compatible array API alias.
similarity_one_swap = pointcloud_similarity_one_swap
similarity_one_swap_indices = pointcloud_similarity_indices


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    A = rng.normal(size=(100, 2))
    x = np.array([[3.0, 0.0]])
    y = np.array([[-3.0, 0.0]])
    points = np.vstack([A, x, y])
    idx_X = np.r_[np.arange(len(A)), len(A)]
    idx_Y = np.r_[np.arange(len(A)), len(A) + 1]
    ans = pointcloud_similarity_indices(points, idx_X, idx_Y, mst_method="auto")
    print(f"s={ans.s:.8f}, R={ans.R:.8f}, I_union={ans.I_union:.8f}")
