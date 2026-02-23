import os

import numpy as np
import pytest

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
pytest.importorskip("gudhi", reason="GUDHI is required for Rips utilities.")

from cripser.rips_utils import VietorisRipsComplex, compute_ph
from cripser.vectorization import persistence_image


def test_compute_ph_returns_cubical_compatible_table():
    points = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
        ],
        dtype=np.float64,
    )

    ph = compute_ph(points, maxdim=1, max_edge_length=2.0, location="yes")

    assert ph.ndim == 2
    assert ph.shape[1] == 9
    assert np.all(np.isin(ph[:, 0].astype(np.int64), [0, 1]))

    ph_short = compute_ph(points, maxdim=1, max_edge_length=2.0, location="no")
    assert ph_short.shape[1] == 3


def test_compute_ph_output_works_with_persistence_image():
    points = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
        ],
        dtype=np.float64,
    )

    ph = compute_ph(points, maxdim=1, max_edge_length=2.0)
    image = persistence_image(
        ph,
        homology_dims=(0, 1),
        n_birth_bins=8,
        n_life_bins=8,
        birth_range=(0.0, 2.0),
        life_range=(0.0, 2.0),
        sigma=0.2,
    )

    assert image.shape == (2, 8, 8)
    assert np.isfinite(image).all()


def test_vietoris_rips_complex_torch_compatible_with_wasserstein_distance():
    torch = pytest.importorskip("torch")
    pytest.importorskip("ot")

    from cripser.distance import wasserstein_distance

    x = torch.tensor(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
        ],
        dtype=torch.float64,
        requires_grad=True,
    )
    y = torch.tensor(
        [
            [0.1, 0.0],
            [1.1, 0.0],
            [0.0, 1.2],
            [1.0, 1.1],
        ],
        dtype=torch.float64,
        requires_grad=True,
    )

    vr = VietorisRipsComplex(max_dimension=1, max_edge_length=2.0)
    ph_x = vr(x)
    ph_y = compute_ph(y, maxdim=1, max_edge_length=2.0)

    assert ph_x.shape[1] == 9
    assert ph_y.shape[1] == 9
    assert len(vr.diagram) == 2
    assert len(vr.pairing) == 2
    assert all(set(block.keys()) == {"finite", "essential"} for block in vr.pairing)

    dist = wasserstein_distance(ph_x, ph_y, dim=0)
    assert torch.isfinite(dist)
    dist.backward()

    assert x.grad is not None
    assert y.grad is not None
    assert torch.linalg.vector_norm(x.grad).item() > 0.0
    assert torch.linalg.vector_norm(y.grad).item() > 0.0
