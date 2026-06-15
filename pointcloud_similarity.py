"""
Efficient computation of point-cloud similarity from 0-dimensional linkage trees.

The basic score is

    s(X, Y) = (I(X) + I(Y) - I(X union Y)) / I(X union Y)

where I(Z) = int_0^R C_r(Z) dr and R is the first radius at which both
X and Y are connected.  C_r(Z) is determined by the Euclidean MST / single
linkage tree.

The preferred API for sliding-window use is the index-set API:

    pointcloud_similarity_index_sets(points, indices_X, indices_Y)
    pointcloud_similarity_from_common_indices(points, common_indices, x_index=..., y_index=...)

The optimized path assumes at most one index in X\\Y and at most one index in
Y\\X.  The original two-array API is kept as pointcloud_similarity_one_swap.

Weighted integral modes are available through measure=... without changing the
default behavior.  Supported kinds are: plain, exponential, size_weighted,
min_size, and effective.

Dependencies: numpy, scipy.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Optional, Sequence, Tuple

import numpy as np

__version__ = "0.5.1"


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
    """Result returned by the point-cloud similarity functions."""

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
    x_unique_index: Optional[int] = None
    y_unique_index: Optional[int] = None
    measure_kind: str = "plain"
    measure_info: Optional[dict[str, Any]] = None


class SimilarityIntegrals(dict):
    """Dict-like result that also supports legacy tuple unpacking.

    Mapping-style access:

        vals["s"], vals["I_X"], vals["measure_info"]

    Legacy notebook access:

        I_X, I_Y, I_union, info = vals
    """

    def __iter__(self):
        # Compatibility with the first weighted notebook draft.
        yield self["I_X"]
        yield self["I_Y"]
        yield self["I_union"]
        yield self["measure_info"]

    @property
    def s(self) -> float:
        return float(self["s"])

    @property
    def R(self) -> float:
        return float(self["R"])

    @property
    def I_X(self) -> float:
        return float(self["I_X"])

    @property
    def I_Y(self) -> float:
        return float(self["I_Y"])

    @property
    def I_union(self) -> float:
        return float(self["I_union"])

    @property
    def measure_kind(self) -> str:
        return str(self["measure_kind"])

    @property
    def measure_info(self) -> dict[str, Any]:
        return dict(self["measure_info"])

    def as_dict(self) -> dict[str, Any]:
        return dict(self)

    def as_legacy_tuple(self) -> tuple[float, float, float, dict[str, Any]]:
        return (self.I_X, self.I_Y, self.I_union, self.measure_info)


class UnionFind:
    """Union-find with path compression and union by rank."""

    __slots__ = ("parent", "rank", "count")

    def __init__(self, n: int):
        self.parent = np.arange(int(n), dtype=np.int64)
        self.rank = np.zeros(int(n), dtype=np.uint8)
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
    except TypeError:
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
    Euclidean MST using Boruvka phases and cKDTree kNN queries.

    This avoids O(n^2) pairwise distances. It is exact when eps=0.0, assuming
    exact cKDTree queries. Like all KD-tree methods, performance can degrade in
    high dimension or adversarial configurations.
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
                    dists = dists[:, None]
                    idxs = idxs[:, None]
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

    Best for low-dimensional data, especially 2D. Duplicate points are handled
    by the Boruvka routine instead.
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

    if n <= d + 1:
        us: list[int] = []
        vs: list[int] = []
        ws: list[float] = []
        for i in range(n):
            for j in range(i + 1, n):
                us.append(i)
                vs.append(j)
                ws.append(float(np.linalg.norm(pts[i] - pts[j])))
        return kruskal_mst(n, us, vs, ws)

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
                if a < b:
                    edges.add((a, b))
                else:
                    edges.add((b, a))

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
    """Convenience Euclidean MST builder."""
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
                pass
        return emst_boruvka_ckdtree(
            pts,
            **{
                k: v
                for k, v in kwargs.items()
                if k in {"initial_k", "leafsize", "eps", "workers", "max_query_entries"}
            },
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
    Compute MST(A plus inserted points) from MST(A) plus inserted candidate edges.

    Candidate edges are MST(A), all edges from each inserted point to A, and all
    edges among inserted points. This is exact for a fixed exact MST(A).
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
        us: list[int] = []
        vs: list[int] = []
        ws: list[float] = []
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


# ---------------------------------------------------------------------------
# Weighted integral modes
# ---------------------------------------------------------------------------


def _positive_scale_from_values(values: np.ndarray, quantile: float, default: float = 1.0) -> float:
    vals = np.asarray(values, dtype=np.float64)
    vals = vals[np.isfinite(vals) & (vals > 0)]
    if vals.size == 0:
        return float(default)
    scale = float(np.quantile(vals, quantile))
    if not np.isfinite(scale) or scale <= 0:
        return float(default)
    return scale


def _median_nearest_neighbor_distance(points: np.ndarray) -> float:
    pts = _as_points(points, "points_for_scale")
    n = pts.shape[0]
    if n <= 1:
        return 1.0
    try:
        from scipy.spatial import cKDTree

        tree = cKDTree(pts)
        dists, _ = _ckdtree_query(tree, pts, k=min(2, n), eps=0.0, workers=-1)
        dists = np.asarray(dists, dtype=np.float64)
        nn = dists if dists.ndim == 1 else dists[:, 1]
        return _positive_scale_from_values(nn, 0.5, default=1.0)
    except Exception:
        best = np.full(n, np.inf, dtype=np.float64)
        for i in range(n):
            d = np.linalg.norm(pts - pts[i], axis=1)
            d[i] = np.inf
            best[i] = np.min(d)
        return _positive_scale_from_values(best, 0.5, default=1.0)


def _normalise_measure(measure: Any) -> dict[str, Any]:
    if measure is None:
        return {"kind": "plain"}
    if isinstance(measure, str):
        key = measure.strip().lower().replace("-", "_")
        aliases = {
            "unweighted": "plain",
            "original": "plain",
            "count": "plain",
            "exp": "exponential",
            "exponential_radial": "exponential",
            "size": "size_weighted",
            "size_weight": "size_weighted",
            "size_weighted_count": "size_weighted",
            "size_weighted_counts": "size_weighted",
            "minimum_size": "min_size",
            "minimum_size_count": "min_size",
            "min_size_count": "min_size",
            "effective_components": "effective",
            "effective_number": "effective",
            "entropy": "effective",
        }
        return {"kind": aliases.get(key, key)}
    if isinstance(measure, Mapping):
        spec = dict(measure)
        kind = str(spec.get("kind", "plain")).strip().lower().replace("-", "_")
        spec["kind"] = _normalise_measure(kind)["kind"]
        return spec
    raise TypeError("measure must be a string, mapping, or None")


def _resolve_exponential_lambda(
    spec: Mapping[str, Any],
    *,
    mst_union: MSTResult,
    points_for_scale: Optional[np.ndarray] = None,
) -> tuple[float, dict[str, Any]]:
    for key in ("lambda", "lambda_", "lam"):
        if key in spec and spec[key] is not None:
            lam = float(spec[key])
            if not np.isfinite(lam) or lam < 0:
                raise ValueError("exponential lambda must be finite and nonnegative")
            return lam, {"kernel": "exponential", "lambda": lam, "scale": None, "half_life": False}

    scale_spec = spec.get("scale", "median_mst_union")
    if isinstance(scale_spec, str):
        scale_name = scale_spec.strip().lower().replace("-", "_")
        if scale_name in {"median_mst", "median_mst_union", "mst_median"}:
            scale = _positive_scale_from_values(mst_union.w, 0.5, default=1.0)
        elif scale_name in {"p90_mst", "p90_mst_union", "mst_p90"}:
            scale = _positive_scale_from_values(mst_union.w, 0.9, default=1.0)
        elif scale_name in {"mean_mst", "mean_mst_union", "mst_mean"}:
            vals = np.asarray(mst_union.w, dtype=np.float64)
            vals = vals[np.isfinite(vals) & (vals > 0)]
            scale = float(np.mean(vals)) if vals.size else 1.0
        elif scale_name in {"median_nn", "median_nn_union", "nn_median"}:
            if points_for_scale is None:
                scale = _positive_scale_from_values(mst_union.w, 0.5, default=1.0)
                scale_name = "median_mst_union_fallback"
            else:
                scale = _median_nearest_neighbor_distance(points_for_scale)
        else:
            try:
                scale = float(scale_spec)
                scale_name = "numeric_string"
            except ValueError as exc:
                raise ValueError(
                    "unknown exponential scale; use a positive number, "
                    "'median_mst_union', 'p90_mst_union', 'mean_mst_union', "
                    "or 'median_nn_union'"
                ) from exc
    else:
        scale = float(scale_spec)
        scale_name = "numeric"

    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("exponential scale must be finite and positive")

    half_life = bool(spec.get("half_life", True))
    lam = float(np.log(2.0) / scale) if half_life else float(1.0 / scale)
    return lam, {
        "kernel": "exponential",
        "lambda": lam,
        "scale": float(scale),
        "scale_source": scale_name,
        "half_life": half_life,
    }


def _primitive_identity(t: np.ndarray | float) -> np.ndarray | float:
    return t


def _make_exponential_primitive(lam: float):
    lam = float(lam)

    def primitive(t: np.ndarray | float) -> np.ndarray | float:
        arr = np.asarray(t, dtype=np.float64)
        if lam == 0.0:
            out = arr
        else:
            out = -np.expm1(-lam * arr) / lam
        if np.ndim(t) == 0:
            return float(out)
        return out

    return primitive


def _radial_count_integral_from_mst(
    mst: MSTResult,
    R: float,
    primitive: Callable[[Any], Any],
) -> float:
    """Integral of f(r) C_r dr, where primitive(t)=int_0^t f(r)dr."""
    R = float(R)
    if R < 0:
        raise ValueError("R must be nonnegative")
    if mst.n == 0:
        return 0.0
    weights = np.asarray(mst.w, dtype=np.float64)
    F_R = float(primitive(R))
    if weights.size == 0:
        return F_R
    return float(F_R + np.sum(primitive(np.minimum(weights, R)), dtype=np.float64))


def _component_integral_from_mst(
    mst: MSTResult,
    R: float,
    *,
    kind: str,
    alpha: float = 0.5,
    min_size: int = 2,
    primitive: Callable[[Any], Any] = _primitive_identity,
) -> float:
    """Integrate component-size functionals over the MST merge sweep."""
    R = float(R)
    if R < 0:
        raise ValueError("R must be nonnegative")
    n = int(mst.n)
    if n == 0:
        return 0.0

    parent = np.arange(n, dtype=np.int64)
    size = np.ones(n, dtype=np.int64)

    def find(a: int) -> int:
        a = int(a)
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = int(parent[a])
        return a

    kind = kind.lower().replace("-", "_")
    if kind == "size_weighted":
        alpha = float(alpha)
        if not np.isfinite(alpha):
            raise ValueError("alpha must be finite")
        current = float(n)  # n * 1**alpha

        def merge_delta(sa: int, sb: int) -> float:
            return float((sa + sb) ** alpha - sa ** alpha - sb ** alpha)

    elif kind == "min_size":
        min_size = int(min_size)
        if min_size < 1:
            raise ValueError("min_size must be at least 1")
        current = float(n if min_size <= 1 else 0)

        def merge_delta(sa: int, sb: int) -> float:
            return float(int(sa + sb >= min_size) - int(sa >= min_size) - int(sb >= min_size))

    elif kind == "effective":
        sum_s_log_s = 0.0
        current = float(n)

        def s_log_s(s: int) -> float:
            return float(s) * float(np.log(float(s))) if s > 1 else 0.0

        def merge_delta(sa: int, sb: int) -> float:
            nonlocal sum_s_log_s, current
            sum_s_log_s += s_log_s(sa + sb) - s_log_s(sa) - s_log_s(sb)
            current = float(n * np.exp(-sum_s_log_s / n))
            return 0.0

    else:
        raise ValueError("unknown component integral kind")

    weights = np.asarray(mst.w, dtype=np.float64)
    order = np.argsort(weights, kind="mergesort") if weights.size else np.empty(0, dtype=np.int64)

    total = 0.0
    prev = 0.0
    pos = 0
    m = order.size

    while pos < m:
        e0 = int(order[pos])
        w = float(weights[e0])
        if w > R:
            break
        if w > prev:
            total += current * float(primitive(w) - primitive(prev))
            prev = w

        while pos < m and float(weights[int(order[pos])]) == w:
            e = int(order[pos])
            ra = find(int(mst.u[e]))
            rb = find(int(mst.v[e]))
            if ra != rb:
                if size[ra] < size[rb]:
                    ra, rb = rb, ra
                sa = int(size[ra])
                sb = int(size[rb])
                parent[rb] = ra
                size[ra] = sa + sb
                if kind == "effective":
                    merge_delta(sa, sb)
                else:
                    current += merge_delta(sa, sb)
            pos += 1

    if R > prev:
        total += current * float(primitive(R) - primitive(prev))
    return float(total)


def _integral_from_mst_with_measure(
    mst: MSTResult,
    R: float,
    measure: Any,
    *,
    mst_union_for_scale: Optional[MSTResult] = None,
    points_for_scale: Optional[np.ndarray] = None,
) -> tuple[float, str, dict[str, Any]]:
    spec = _normalise_measure(measure)
    kind = str(spec.get("kind", "plain"))
    info: dict[str, Any] = {"kind": kind}

    if kind == "plain":
        return integral_from_mst(mst, R), kind, info

    if kind == "exponential":
        if mst_union_for_scale is None:
            mst_union_for_scale = mst
        lam, lam_info = _resolve_exponential_lambda(
            spec, mst_union=mst_union_for_scale, points_for_scale=points_for_scale
        )
        info.update(lam_info)
        primitive = _make_exponential_primitive(lam)
        return _radial_count_integral_from_mst(mst, R, primitive), kind, info

    if kind == "size_weighted":
        alpha = float(spec.get("alpha", 0.5))
        info["alpha"] = alpha
        return _component_integral_from_mst(mst, R, kind="size_weighted", alpha=alpha), kind, info

    if kind == "min_size":
        min_size = int(spec.get("min_size", spec.get("minimum_size", 2)))
        info["min_size"] = min_size
        return _component_integral_from_mst(mst, R, kind="min_size", min_size=min_size), kind, info

    if kind == "effective":
        return _component_integral_from_mst(mst, R, kind="effective"), kind, info

    raise ValueError(
        "unknown measure kind. Supported kinds are: 'plain', 'exponential', "
        "'size_weighted', 'min_size', and 'effective'."
    )


def similarity_integrals_from_msts(
    mst_X: MSTResult,
    mst_Y: MSTResult,
    mst_union: MSTResult,
    R: Optional[float] = None,
    *,
    measure: Any = "plain",
    points_for_scale: Optional[np.ndarray] = None,
    union_points: Optional[np.ndarray] = None,
) -> SimilarityIntegrals:
    """
    Evaluate the similarity score from precomputed MSTs.

    This is useful when the same linkage trees should be reused for several
    measure modes. The result is dict-like and also supports legacy tuple
    unpacking as I_X, I_Y, I_union, info.
    """
    if points_for_scale is None and union_points is not None:
        points_for_scale = union_points
    if R is None:
        R = max(mst_X.max_edge, mst_Y.max_edge)
    R = float(R)

    I_X, kind, info = _integral_from_mst_with_measure(
        mst_X, R, measure, mst_union_for_scale=mst_union, points_for_scale=points_for_scale
    )
    I_Y, _, _ = _integral_from_mst_with_measure(
        mst_Y, R, measure, mst_union_for_scale=mst_union, points_for_scale=points_for_scale
    )
    I_union, _, _ = _integral_from_mst_with_measure(
        mst_union, R, measure, mst_union_for_scale=mst_union, points_for_scale=points_for_scale
    )
    s = float("nan") if I_union == 0.0 else float((I_X + I_Y - I_union) / I_union)
    return SimilarityIntegrals(
        s=float(s),
        R=float(R),
        I_X=float(I_X),
        I_Y=float(I_Y),
        I_union=float(I_union),
        measure_kind=str(kind),
        measure_info=dict(info),
    )


def _compute_similarity_from_msts(
    mst_X: MSTResult,
    mst_Y: MSTResult,
    mst_union: MSTResult,
    *,
    mst_common: MSTResult,
    n_common: int,
    x_unique: Optional[np.ndarray],
    y_unique: Optional[np.ndarray],
    x_unique_index: Optional[int] = None,
    y_unique_index: Optional[int] = None,
    measure: Any = "plain",
    points_for_scale: Optional[np.ndarray] = None,
) -> SimilarityResult:
    vals = similarity_integrals_from_msts(
        mst_X, mst_Y, mst_union, measure=measure, points_for_scale=points_for_scale
    )
    return SimilarityResult(
        s=float(vals["s"]),
        R=float(vals["R"]),
        I_X=float(vals["I_X"]),
        I_Y=float(vals["I_Y"]),
        I_union=float(vals["I_union"]),
        mst_X=mst_X,
        mst_Y=mst_Y,
        mst_union=mst_union,
        mst_common=mst_common,
        n_common=int(n_common),
        x_unique=None if x_unique is None else np.asarray(x_unique, dtype=np.float64).copy(),
        y_unique=None if y_unique is None else np.asarray(y_unique, dtype=np.float64).copy(),
        x_unique_index=None if x_unique_index is None else int(x_unique_index),
        y_unique_index=None if y_unique_index is None else int(y_unique_index),
        measure_kind=str(vals["measure_kind"]),
        measure_info=dict(vals["measure_info"]),
    )


# ---------------------------------------------------------------------------
# Exact one-delete/one-add MST updates
# ---------------------------------------------------------------------------


def one_swap_msts_from_common(
    common_points: np.ndarray,
    common_mst: MSTResult,
    x: Optional[np.ndarray],
    y: Optional[np.ndarray],
) -> Tuple[MSTResult, MSTResult, MSTResult]:
    """
    Compute MST(X), MST(Y), MST(X union Y) in one sorted Kruskal pass.

    Here X = A plus optional x and Y = A plus optional y. The candidate edge
    set is MST(A), the star edges from inserted points to A, and the x-y edge.
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

    mst_X = MSTResult(
        nX,
        np.asarray(outXu, dtype=np.int64),
        np.asarray(outXv, dtype=np.int64),
        np.asarray(outXw, dtype=np.float64),
    )
    mst_Y = MSTResult(
        nY,
        np.asarray(outYu, dtype=np.int64),
        np.asarray(outYv, dtype=np.int64),
        np.asarray(outYw, dtype=np.float64),
    )
    mst_U = MSTResult(
        nU,
        np.asarray(outUu, dtype=np.int64),
        np.asarray(outUv, dtype=np.int64),
        np.asarray(outUw, dtype=np.float64),
    )
    return mst_X, mst_Y, mst_U


# ---------------------------------------------------------------------------
# Splitting / index-set APIs
# ---------------------------------------------------------------------------


def _row_key(row: np.ndarray) -> tuple:
    return tuple(np.asarray(row).tolist())


def split_clouds_one_swap(
    X: np.ndarray,
    Y: np.ndarray,
    *,
    ids_X: Optional[Sequence[object]] = None,
    ids_Y: Optional[Sequence[object]] = None,
    check_common_coordinates: bool = True,
) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
    """Return A, x, y with X = A plus optional x and Y = A plus optional y."""
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
            "provide stable ids_X/ids_Y if rows are floating-point copies"
        )

    A = np.vstack(common_rows).astype(np.float64, copy=False) if common_rows else np.empty((0, Xp.shape[1]), dtype=np.float64)
    x = x_rows[0] if x_rows else None
    y = y_rows[0] if y_rows else None
    return A, x, y


def _normalise_index_array(indices: Sequence[int] | np.ndarray, n_points: int, name: str) -> np.ndarray:
    arr = np.asarray(indices)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if arr.size == 0:
        return np.empty(0, dtype=np.int64)
    if not np.issubdtype(arr.dtype, np.integer):
        as_float = arr.astype(np.float64)
        if not np.all(np.isfinite(as_float)) or not np.all(as_float == np.floor(as_float)):
            raise ValueError(f"{name} must contain integer indices")
        arr = as_float.astype(np.int64)
    else:
        arr = arr.astype(np.int64, copy=False)
    if np.any(arr < 0) or np.any(arr >= n_points):
        raise IndexError(f"{name} contains an index outside [0, {n_points})")
    if np.unique(arr).size != arr.size:
        raise ValueError(f"{name} contains duplicate indices; pass point clouds as sets")
    return np.ascontiguousarray(arr, dtype=np.int64)


def _normalise_optional_single_index(index: Optional[int], n_points: int, name: str) -> Optional[int]:
    if index is None:
        return None
    if isinstance(index, np.generic):
        index = index.item()
    if not isinstance(index, (int, np.integer)):
        raise TypeError(f"{name} must be an integer index or None")
    out = int(index)
    if out < 0 or out >= n_points:
        raise IndexError(f"{name} is outside [0, {n_points})")
    return out


def pointcloud_similarity_from_common_indices(
    points: np.ndarray,
    common_indices: Sequence[int] | np.ndarray,
    *,
    x_index: Optional[int] = None,
    y_index: Optional[int] = None,
    mst_builder: Optional[Callable[[np.ndarray], MSTResult]] = None,
    mst_method: str = "auto",
    measure: Any = "plain",
    **mst_kwargs,
) -> SimilarityResult:
    """
    Compute the similarity when X=A+optional x and Y=A+optional y.

    measure="plain" preserves the original integral. Other supported modes are
    "exponential", {"kind":"size_weighted", "alpha":0.5},
    {"kind":"min_size", "min_size":m}, and "effective".
    """
    pts = _as_points(points, "points")
    n_total = pts.shape[0]
    common = _normalise_index_array(common_indices, n_total, "common_indices")
    xi = _normalise_optional_single_index(x_index, n_total, "x_index")
    yi = _normalise_optional_single_index(y_index, n_total, "y_index")

    forbidden = set(map(int, common))
    if xi is not None and xi in forbidden:
        raise ValueError("x_index is already present in common_indices")
    if yi is not None and yi in forbidden:
        raise ValueError("y_index is already present in common_indices")
    if xi is not None and yi is not None and xi == yi:
        raise ValueError("x_index and y_index must be distinct")

    A = pts[common]
    x = None if xi is None else pts[xi]
    y = None if yi is None else pts[yi]

    if mst_builder is None:
        common_mst = emst(A, method=mst_method, **mst_kwargs)
    else:
        common_mst = mst_builder(A)
        if not isinstance(common_mst, MSTResult):
            raise TypeError("mst_builder must return an MSTResult")

    mst_X, mst_Y, mst_union = one_swap_msts_from_common(A, common_mst, x, y)

    union_indices = list(map(int, common))
    if xi is not None:
        union_indices.append(xi)
    if yi is not None:
        union_indices.append(yi)
    points_for_scale = pts[np.asarray(union_indices, dtype=np.int64)] if union_indices else None

    return _compute_similarity_from_msts(
        mst_X,
        mst_Y,
        mst_union,
        mst_common=common_mst,
        n_common=A.shape[0],
        x_unique=x,
        y_unique=y,
        x_unique_index=xi,
        y_unique_index=yi,
        measure=measure,
        points_for_scale=points_for_scale,
    )


def pointcloud_similarity_index_sets(
    points: np.ndarray,
    indices_X: Sequence[int] | np.ndarray,
    indices_Y: Sequence[int] | np.ndarray,
    *,
    mst_builder: Optional[Callable[[np.ndarray], MSTResult]] = None,
    mst_method: str = "auto",
    measure: Any = "plain",
    **mst_kwargs,
) -> SimilarityResult:
    """
    Compute s(X,Y) from one shared point array and two included-index sets.

    The optimized path requires at most one index in X\\Y and at most one index
    in Y\\X.
    """
    pts = _as_points(points, "points")
    idx_X = _normalise_index_array(indices_X, pts.shape[0], "indices_X")
    idx_Y = _normalise_index_array(indices_Y, pts.shape[0], "indices_Y")

    set_X = set(map(int, idx_X))
    set_Y = set(map(int, idx_Y))
    only_X = [int(i) for i in idx_X if int(i) not in set_Y]
    only_Y = [int(i) for i in idx_Y if int(i) not in set_X]
    if len(only_X) > 1 or len(only_Y) > 1:
        raise ValueError(
            "pointcloud_similarity_index_sets uses the one-delete/one-add fast path; "
            "received more than one index in X\\Y or Y\\X"
        )

    common = np.asarray([int(i) for i in idx_X if int(i) in set_Y], dtype=np.int64)
    return pointcloud_similarity_from_common_indices(
        pts,
        common,
        x_index=only_X[0] if only_X else None,
        y_index=only_Y[0] if only_Y else None,
        mst_builder=mst_builder,
        mst_method=mst_method,
        measure=measure,
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
    measure: Any = "plain",
    **mst_kwargs,
) -> SimilarityResult:
    """
    Compute s(X,Y) for point clouds differing by at most one deletion/insertion.

    This is the original two-array API. For new code with one shared point
    array, prefer pointcloud_similarity_index_sets().
    """
    Xp = _as_points(X, "X")
    Yp = _as_points(Y, "Y")
    A, x, y = split_clouds_one_swap(Xp, Yp, ids_X=ids_X, ids_Y=ids_Y)

    if mst_builder is None:
        common_mst = emst(A, method=mst_method, **mst_kwargs)
    else:
        common_mst = mst_builder(A)
        if not isinstance(common_mst, MSTResult):
            raise TypeError("mst_builder must return an MSTResult")

    mst_X, mst_Y, mst_union = one_swap_msts_from_common(A, common_mst, x, y)

    parts = [A]
    if x is not None:
        parts.append(np.asarray(x, dtype=np.float64).reshape(1, -1))
    if y is not None:
        parts.append(np.asarray(y, dtype=np.float64).reshape(1, -1))
    points_for_scale = np.vstack(parts) if parts else None

    return _compute_similarity_from_msts(
        mst_X,
        mst_Y,
        mst_union,
        mst_common=common_mst,
        n_common=A.shape[0],
        x_unique=x,
        y_unique=y,
        measure=measure,
        points_for_scale=points_for_scale,
    )


def weighted_integral_from_mst(
    mst: MSTResult,
    R: float,
    *,
    measure: Any = "exponential",
    mst_union_for_scale: Optional[MSTResult] = None,
    points_for_scale: Optional[np.ndarray] = None,
) -> float:
    """Evaluate any supported integral mode for a single MST.

    For the original unweighted integral, call ``integral_from_mst(mst, R)``
    or use ``measure="plain"``.
    """
    value, _, _ = _integral_from_mst_with_measure(
        mst,
        R,
        measure,
        mst_union_for_scale=mst if mst_union_for_scale is None else mst_union_for_scale,
        points_for_scale=points_for_scale,
    )
    return float(value)


def component_integral_from_mst(
    mst: MSTResult,
    R: float,
    *,
    kind: str = "size_weighted",
    alpha: float = 0.5,
    min_size: int = 2,
) -> float:
    """Compatibility helper for component-weighted integrals.

    Supported ``kind`` values are ``"size_weighted"``, ``"min_size"``,
    and ``"effective"``.
    """
    return _component_integral_from_mst(
        mst,
        R,
        kind=kind,
        alpha=alpha,
        min_size=min_size,
    )


# Backwards-friendly aliases.
similarity_one_swap = pointcloud_similarity_one_swap
pointcloud_similarity_indices = pointcloud_similarity_index_sets
pointcloud_similarity_from_indices = pointcloud_similarity_index_sets
pointcloud_similarity_one_swap_indices = pointcloud_similarity_index_sets
similarity_index_sets = pointcloud_similarity_index_sets
similarity_indices = pointcloud_similarity_index_sets


__all__ = [
    "__version__",
    "MSTResult",
    "SimilarityResult",
    "SimilarityIntegrals",
    "UnionFind",
    "kruskal_mst",
    "emst",
    "emst_delaunay",
    "emst_boruvka_ckdtree",
    "mst_with_inserted_points",
    "integral_from_mst",
    "similarity_integrals_from_msts",
    "weighted_integral_from_mst",
    "component_integral_from_mst",
    "pointcloud_similarity_index_sets",
    "pointcloud_similarity_indices",
    "pointcloud_similarity_from_indices",
    "pointcloud_similarity_one_swap_indices",
    "pointcloud_similarity_from_common_indices",
    "pointcloud_similarity_one_swap",
    "similarity_index_sets",
    "similarity_indices",
    "similarity_one_swap",
]


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    A = rng.normal(size=(100, 2))
    x = np.array([[3.0, 0.0]])
    y = np.array([[-3.0, 0.0]])
    X = np.vstack([A, x])
    Y = np.vstack([A, y])
    ans = pointcloud_similarity_one_swap(X, Y, mst_method="auto")
    print(f"s={ans.s:.8f}, R={ans.R:.8f}, I_union={ans.I_union:.8f}")
