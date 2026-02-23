"""Vietoris-Rips persistence utilities based on GUDHI.

The main entry point is :func:`compute_ph`, which returns a CubicalRipser-like
PH table by default:

``[dim, birth, death, b0, b1, b2, d0, d1, d2]``

This makes outputs directly compatible with existing utilities such as
``cripser.distance.wasserstein_distance`` and
``cripser.vectorization.persistence_image``.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

try:
    import torch
except ModuleNotFoundError:
    torch = None  # type: ignore[assignment]

try:
    import gudhi
except ModuleNotFoundError:
    gudhi = None  # type: ignore[assignment]


ArrayLike = np.ndarray | Sequence[float] | Sequence[Sequence[float]]


def _require_gudhi() -> None:
    if gudhi is None:
        raise ImportError("gudhi is required for Vietoris-Rips persistence.")


def _parse_location(location: str | bool) -> bool:
    if isinstance(location, bool):
        return location
    text = str(location).strip().lower()
    if text in {"yes", "y", "true", "1"}:
        return True
    if text in {"no", "n", "false", "0"}:
        return False
    raise ValueError("`location` must be one of {'yes', 'no'} or a boolean.")


def _as_int_matrix(values: Any, n_cols: int) -> np.ndarray:
    out = np.asarray(values, dtype=np.int64)
    if out.size == 0:
        return np.empty((0, n_cols), dtype=np.int64)
    return out.reshape(-1, n_cols)


def _as_int_vector(values: Any) -> np.ndarray:
    out = np.asarray(values, dtype=np.int64)
    if out.size == 0:
        return np.empty((0,), dtype=np.int64)
    return out.reshape(-1)


def _edge_lengths_numpy(points: np.ndarray, edges: np.ndarray, *, p: float) -> np.ndarray:
    if edges.shape[0] == 0:
        return np.empty((0,), dtype=np.float64)
    diff = points[edges[:, 0]] - points[edges[:, 1]]
    return np.linalg.norm(diff, ord=float(p), axis=1).astype(np.float64, copy=False)


def _edge_lengths_torch(
    points: "torch.Tensor",
    edges: np.ndarray,
    *,
    p: float,
) -> "torch.Tensor":
    if edges.shape[0] == 0:
        return torch.empty((0,), dtype=points.dtype, device=points.device)
    edge_idx = torch.as_tensor(edges, dtype=torch.long, device=points.device)
    diff = points[edge_idx[:, 0]] - points[edge_idx[:, 1]]
    return torch.linalg.vector_norm(diff, ord=float(p), dim=1)


def _normalize_point_cloud(
    point_cloud: ArrayLike | "torch.Tensor",
) -> tuple[bool, np.ndarray, Any]:
    if torch is not None and isinstance(point_cloud, torch.Tensor):
        if point_cloud.ndim != 2:
            raise ValueError(
                f"`point_cloud` must have shape (n_points, dim), got {tuple(point_cloud.shape)}."
            )
        if point_cloud.shape[0] == 0:
            raise ValueError("`point_cloud` must contain at least one point.")
        points = point_cloud
        if not points.is_floating_point():
            points = points.to(dtype=torch.float64)
        points_np = points.detach().to(device="cpu", dtype=torch.float64).numpy()
        return True, points_np, points

    points_np = np.asarray(point_cloud, dtype=np.float64)
    if points_np.ndim != 2:
        raise ValueError(
            f"`point_cloud` must have shape (n_points, dim), got {tuple(points_np.shape)}."
        )
    if points_np.shape[0] == 0:
        raise ValueError("`point_cloud` must contain at least one point.")
    return False, points_np, points_np


def _build_numpy_rows(
    points_np: np.ndarray,
    *,
    p: float,
    maxdim: int,
    include_location: bool,
    regular_0: np.ndarray,
    regular_higher: list[np.ndarray],
    essential_0: np.ndarray,
    essential_higher: list[np.ndarray],
) -> tuple[np.ndarray, list[np.ndarray], list[dict[str, np.ndarray]]]:
    rows: list[np.ndarray] = []
    diagrams: list[np.ndarray] = []
    pairing: list[dict[str, np.ndarray]] = []

    for dim in range(maxdim + 1):
        if dim == 0:
            finite = regular_0
            essential = essential_0

            births_f = np.zeros((finite.shape[0],), dtype=np.float64)
            deaths_f = _edge_lengths_numpy(points_np, finite[:, 1:3], p=p)
            births_e = np.zeros((essential.shape[0],), dtype=np.float64)
            deaths_e = np.full((essential.shape[0],), np.inf, dtype=np.float64)

            birth_coords_f = np.full((finite.shape[0], 3), -1.0, dtype=np.float64)
            death_coords_f = np.full((finite.shape[0], 3), -1.0, dtype=np.float64)
            if finite.shape[0] > 0:
                birth_coords_f[:, 0] = finite[:, 0]
                death_coords_f[:, :2] = finite[:, 1:3]

            birth_coords_e = np.full((essential.shape[0], 3), -1.0, dtype=np.float64)
            death_coords_e = np.full((essential.shape[0], 3), -1.0, dtype=np.float64)
            if essential.shape[0] > 0:
                birth_coords_e[:, 0] = essential

            pairing.append({"finite": finite.copy(), "essential": essential.copy()})

        else:
            finite = (
                regular_higher[dim - 1]
                if dim - 1 < len(regular_higher)
                else np.empty((0, 4), dtype=np.int64)
            )
            essential = (
                essential_higher[dim - 1]
                if dim - 1 < len(essential_higher)
                else np.empty((0, 2), dtype=np.int64)
            )

            births_f = _edge_lengths_numpy(points_np, finite[:, :2], p=p)
            deaths_f = _edge_lengths_numpy(points_np, finite[:, 2:4], p=p)
            births_e = _edge_lengths_numpy(points_np, essential[:, :2], p=p)
            deaths_e = np.full((essential.shape[0],), np.inf, dtype=np.float64)

            birth_coords_f = np.full((finite.shape[0], 3), -1.0, dtype=np.float64)
            death_coords_f = np.full((finite.shape[0], 3), -1.0, dtype=np.float64)
            if finite.shape[0] > 0:
                birth_coords_f[:, :2] = finite[:, :2]
                death_coords_f[:, :2] = finite[:, 2:4]

            birth_coords_e = np.full((essential.shape[0], 3), -1.0, dtype=np.float64)
            death_coords_e = np.full((essential.shape[0], 3), -1.0, dtype=np.float64)
            if essential.shape[0] > 0:
                birth_coords_e[:, :2] = essential[:, :2]

            pairing.append({"finite": finite.copy(), "essential": essential.copy()})

        births = np.concatenate((births_f, births_e), axis=0)
        deaths = np.concatenate((deaths_f, deaths_e), axis=0)
        diagrams.append(np.stack((births, deaths), axis=1))
        if births.size == 0:
            continue

        dim_col = np.full((births.shape[0], 1), float(dim), dtype=np.float64)
        base = np.concatenate((dim_col, births[:, None], deaths[:, None]), axis=1)
        if include_location:
            birth_coords = np.concatenate((birth_coords_f, birth_coords_e), axis=0)
            death_coords = np.concatenate((death_coords_f, death_coords_e), axis=0)
            rows.append(np.concatenate((base, birth_coords, death_coords), axis=1))
        else:
            rows.append(base)

    n_cols = 9 if include_location else 3
    ph = np.concatenate(rows, axis=0) if rows else np.empty((0, n_cols), dtype=np.float64)
    return ph, diagrams, pairing


def _build_torch_rows(
    points_t: "torch.Tensor",
    *,
    p: float,
    maxdim: int,
    include_location: bool,
    regular_0: np.ndarray,
    regular_higher: list[np.ndarray],
    essential_0: np.ndarray,
    essential_higher: list[np.ndarray],
) -> tuple["torch.Tensor", list["torch.Tensor"], list[dict[str, "torch.Tensor"]]]:
    rows: list[torch.Tensor] = []
    diagrams: list[torch.Tensor] = []
    pairing: list[dict[str, torch.Tensor]] = []

    device = points_t.device
    out_dtype = points_t.dtype

    for dim in range(maxdim + 1):
        if dim == 0:
            finite = regular_0
            essential = essential_0

            births_f = torch.zeros((finite.shape[0],), dtype=out_dtype, device=device)
            deaths_f = _edge_lengths_torch(points_t, finite[:, 1:3], p=p)
            births_e = torch.zeros((essential.shape[0],), dtype=out_dtype, device=device)
            deaths_e = torch.full((essential.shape[0],), float("inf"), dtype=out_dtype, device=device)

            birth_coords_f = torch.full((finite.shape[0], 3), -1.0, dtype=out_dtype, device=device)
            death_coords_f = torch.full((finite.shape[0], 3), -1.0, dtype=out_dtype, device=device)
            if finite.shape[0] > 0:
                birth_coords_f[:, 0] = torch.as_tensor(finite[:, 0], dtype=out_dtype, device=device)
                death_coords_f[:, :2] = torch.as_tensor(
                    finite[:, 1:3], dtype=out_dtype, device=device
                )

            birth_coords_e = torch.full((essential.shape[0], 3), -1.0, dtype=out_dtype, device=device)
            death_coords_e = torch.full((essential.shape[0], 3), -1.0, dtype=out_dtype, device=device)
            if essential.shape[0] > 0:
                birth_coords_e[:, 0] = torch.as_tensor(essential, dtype=out_dtype, device=device)

            pairing.append(
                {
                    "finite": torch.as_tensor(finite, dtype=torch.long, device=device),
                    "essential": torch.as_tensor(essential, dtype=torch.long, device=device),
                }
            )

        else:
            finite = (
                regular_higher[dim - 1]
                if dim - 1 < len(regular_higher)
                else np.empty((0, 4), dtype=np.int64)
            )
            essential = (
                essential_higher[dim - 1]
                if dim - 1 < len(essential_higher)
                else np.empty((0, 2), dtype=np.int64)
            )

            births_f = _edge_lengths_torch(points_t, finite[:, :2], p=p)
            deaths_f = _edge_lengths_torch(points_t, finite[:, 2:4], p=p)
            births_e = _edge_lengths_torch(points_t, essential[:, :2], p=p)
            deaths_e = torch.full((essential.shape[0],), float("inf"), dtype=out_dtype, device=device)

            birth_coords_f = torch.full((finite.shape[0], 3), -1.0, dtype=out_dtype, device=device)
            death_coords_f = torch.full((finite.shape[0], 3), -1.0, dtype=out_dtype, device=device)
            if finite.shape[0] > 0:
                birth_coords_f[:, :2] = torch.as_tensor(finite[:, :2], dtype=out_dtype, device=device)
                death_coords_f[:, :2] = torch.as_tensor(
                    finite[:, 2:4], dtype=out_dtype, device=device
                )

            birth_coords_e = torch.full((essential.shape[0], 3), -1.0, dtype=out_dtype, device=device)
            death_coords_e = torch.full((essential.shape[0], 3), -1.0, dtype=out_dtype, device=device)
            if essential.shape[0] > 0:
                birth_coords_e[:, :2] = torch.as_tensor(
                    essential[:, :2], dtype=out_dtype, device=device
                )

            pairing.append(
                {
                    "finite": torch.as_tensor(finite, dtype=torch.long, device=device),
                    "essential": torch.as_tensor(essential, dtype=torch.long, device=device),
                }
            )

        births = torch.cat((births_f, births_e), dim=0)
        deaths = torch.cat((deaths_f, deaths_e), dim=0)
        diagrams.append(torch.stack((births, deaths), dim=1))
        if births.numel() == 0:
            continue

        dim_col = torch.full((births.shape[0], 1), float(dim), dtype=out_dtype, device=device)
        base = torch.cat((dim_col, births[:, None], deaths[:, None]), dim=1)
        if include_location:
            birth_coords = torch.cat((birth_coords_f, birth_coords_e), dim=0)
            death_coords = torch.cat((death_coords_f, death_coords_e), dim=0)
            rows.append(torch.cat((base, birth_coords, death_coords), dim=1))
        else:
            rows.append(base)

    n_cols = 9 if include_location else 3
    ph = (
        torch.cat(rows, dim=0)
        if rows
        else torch.empty((0, n_cols), dtype=out_dtype, device=device)
    )
    return ph, diagrams, pairing


def _compute_rips_ph(
    point_cloud: ArrayLike | "torch.Tensor",
    *,
    maxdim: int,
    p: float,
    max_edge_length: float | None,
    homology_coeff_field: int,
    min_persistence: float,
    include_location: bool,
) -> tuple[Any, list[Any], list[dict[str, Any]]]:
    _require_gudhi()
    if maxdim < 0:
        raise ValueError(f"`maxdim` must be non-negative, got {maxdim}.")
    if p <= 0:
        raise ValueError(f"`p` must be positive, got {p}.")
    if float(p) != 2.0:
        raise ValueError("GUDHI RipsComplex pairings are Euclidean; only `p=2` is supported.")
    if homology_coeff_field <= 1:
        raise ValueError("`homology_coeff_field` must be > 1.")
    if max_edge_length is not None and max_edge_length <= 0:
        raise ValueError("`max_edge_length` must be positive when provided.")

    is_torch, points_np, points_data = _normalize_point_cloud(point_cloud)

    rips_kwargs: dict[str, float] = {}
    if max_edge_length is not None:
        rips_kwargs["max_edge_length"] = float(max_edge_length)
    simplex_tree = gudhi.RipsComplex(points=points_np, **rips_kwargs).create_simplex_tree(
        max_dimension=maxdim + 1
    )
    simplex_tree.persistence(
        homology_coeff_field=int(homology_coeff_field),
        min_persistence=float(min_persistence),
    )

    generators = simplex_tree.flag_persistence_generators()
    regular_0 = _as_int_matrix(generators[0], 3)
    regular_higher = [_as_int_matrix(arr, 4) for arr in generators[1]]
    essential_0 = _as_int_vector(generators[2])
    essential_higher = [_as_int_matrix(arr, 2) for arr in generators[3]]

    if is_torch:
        return _build_torch_rows(
            points_data,
            p=p,
            maxdim=maxdim,
            include_location=include_location,
            regular_0=regular_0,
            regular_higher=regular_higher,
            essential_0=essential_0,
            essential_higher=essential_higher,
        )
    return _build_numpy_rows(
        points_np,
        p=p,
        maxdim=maxdim,
        include_location=include_location,
        regular_0=regular_0,
        regular_higher=regular_higher,
        essential_0=essential_0,
        essential_higher=essential_higher,
    )


def compute_ph(
    point_cloud: ArrayLike | "torch.Tensor",
    *,
    maxdim: int = 1,
    p: float = 2.0,
    max_edge_length: float | None = None,
    homology_coeff_field: int = 2,
    min_persistence: float = 0.0,
    location: str | bool = "yes",
) -> np.ndarray | "torch.Tensor":
    """Compute Vietoris-Rips PH with a CubicalRipser-compatible tabular output.

    Parameters
    - point_cloud:
      Point-cloud array/tensor with shape `(n_points, ambient_dim)`.
    - maxdim:
      Maximum homology dimension to compute.
    - p:
      Metric exponent used when converting generator edges to birth/death values.
      For GUDHI-backed Rips pairings this must be `2`.
    - max_edge_length:
      Optional edge threshold for Rips construction.
    - homology_coeff_field:
      Field characteristic used by GUDHI persistence computation.
    - min_persistence:
      Minimum persistence threshold forwarded to GUDHI.
    - location:
      `"yes"`/`True` returns 9 columns:
      `[dim, birth, death, b0, b1, b2, d0, d1, d2]`.
      `"no"`/`False` returns only `[dim, birth, death]`.

    Returns
    - NumPy array (for NumPy-like input) or torch tensor (for torch input).
      Birth/death columns remain differentiable for torch input.
    """
    include_location = _parse_location(location)
    ph, _, _ = _compute_rips_ph(
        point_cloud,
        maxdim=int(maxdim),
        p=float(p),
        max_edge_length=max_edge_length,
        homology_coeff_field=int(homology_coeff_field),
        min_persistence=float(min_persistence),
        include_location=include_location,
    )
    return ph


compute_ph_rips = compute_ph


class VietorisRipsComplex:
    """Stateful wrapper around :func:`compute_ph`.

    After calling the instance:
    - ``self.diagram`` stores per-dimension `(birth, death)` arrays/tensors.
    - ``self.pairing`` stores per-dimension generator indices:
      ``{"finite": ..., "essential": ...}``.
    """

    def __init__(
        self,
        max_dimension: int = 1,
        p: float = 2.0,
        max_edge_length: float | None = None,
        *,
        homology_coeff_field: int = 2,
        min_persistence: float = 0.0,
        location: str | bool = "yes",
    ) -> None:
        self.dim = int(max_dimension)
        self.p = float(p)
        self.max_edge_length = max_edge_length
        self.homology_coeff_field = int(homology_coeff_field)
        self.min_persistence = float(min_persistence)
        self.location = location
        self.pairing: list[dict[str, Any]] = []
        self.diagram: list[Any] = []

    def __call__(self, point_cloud: ArrayLike | "torch.Tensor") -> np.ndarray | "torch.Tensor":
        include_location = _parse_location(self.location)
        ph, diagrams, pairing = _compute_rips_ph(
            point_cloud,
            maxdim=self.dim,
            p=self.p,
            max_edge_length=self.max_edge_length,
            homology_coeff_field=self.homology_coeff_field,
            min_persistence=self.min_persistence,
            include_location=include_location,
        )
        self.diagram = diagrams
        self.pairing = pairing
        return ph


__all__ = ["compute_ph", "compute_ph_rips", "VietorisRipsComplex"]
