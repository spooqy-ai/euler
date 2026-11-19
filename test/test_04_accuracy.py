from itertools import product
from typing import Any
import pytest
import numpy as np
from euler.utils import AxisTriple, AXIS_TRIPLES
from euler.angles import angles
from euler.matrix import matrix
from euler.convert import convert
from euler.su2 import so3_to_su2, su2_to_so3

NUM_SAMPLES = 200
TIGHT_TOL = 1e-14
ACCURACY = 1e-12

SPECIAL_ANGLES = [
    0.0,
    1e-12,
    1e-9,
    1e-6,
    4e-5,
    1e-3,
    np.pi / 2 - 3e-5,
    np.pi / 2,
    np.pi / 2 + 1e-7,
    np.pi - 1e-6,
    np.pi,
]


def angle_samples(seed: int) -> list[tuple[float, float, float]]:
    """Random angle triples, half of them with entries at or near special values."""
    rng = np.random.default_rng(seed)
    samples = []
    for k in range(NUM_SAMPLES):
        triple = list(rng.uniform(-np.pi, np.pi, size=3))
        if k % 2:
            for i in range(3):
                if rng.random() < 0.6:
                    triple[i] = float(rng.choice(SPECIAL_ANGLES) * rng.choice([1, -1]))
        samples.append((triple[0], triple[1], triple[2]))
    return samples


def su2_matrix(p: AxisTriple, a: float, b: float, c: float) -> Any:
    """SU2 rotation for the given axis triple and Euler angles."""
    paulis = {
        "x": np.array([[0, 1], [1, 0]], dtype=np.complex128),
        "y": np.array([[0, -1j], [1j, 0]], dtype=np.complex128),
        "z": np.array([[1, 0], [0, -1]], dtype=np.complex128),
    }
    mat = np.eye(2, dtype=np.complex128)
    for axis, angle in zip(p, (a, b, c)):
        mat = mat @ (
            np.cos(angle / 2) * np.eye(2) - 1j * np.sin(angle / 2) * paulis[axis]
        )
    return mat


@pytest.mark.parametrize("p", AXIS_TRIPLES)
def test_angles_accuracy(p: AxisTriple) -> None:
    for a, b, c in angle_samples(0):
        mat = matrix(p, a, b, c)
        assert np.abs(matrix(p, *angles(p, mat, TIGHT_TOL)) - mat).max() < ACCURACY
        assert np.abs(matrix(p, *angles(p, mat)) - mat).max() < 1e-7


@pytest.mark.parametrize("p,q", tuple(product(AXIS_TRIPLES, repeat=2)))
def test_conversion_accuracy(p: AxisTriple, q: AxisTriple) -> None:
    for a, b, c in angle_samples(1):
        mat = matrix(p, a, b, c)
        tight = convert(p, q, a, b, c, TIGHT_TOL)
        assert np.abs(matrix(q, *tight) - mat).max() < ACCURACY
        assert np.abs(matrix(q, *convert(p, q, a, b, c)) - mat).max() < 1e-7


@pytest.mark.parametrize("q", AXIS_TRIPLES)
def test_conversion_at_gimbal_lock(q: AxisTriple) -> None:
    locks = (0.0, np.pi) if q[0] == q[2] else (np.pi / 2, -np.pi / 2)
    for lock, a in product(locks, np.linspace(-3, 3, 7)):
        _a, _b, _c = convert(q, q, a, lock + 1e-10, 0.4)
        assert _b == lock and _c == 0.0
        assert np.abs(matrix(q, _a, _b, _c) - matrix(q, a, lock, 0.4)).max() < 1e-9


@pytest.mark.parametrize("p", AXIS_TRIPLES)
def test_su2_accuracy(p: AxisTriple) -> None:
    for a, b, c in angle_samples(2):
        su2_mat = su2_matrix(p, a, b, c)
        so3_mat = matrix(p, a, b, c)
        assert np.abs(su2_to_so3(su2_mat) - so3_mat).max() < ACCURACY
        assert np.abs(su2_to_so3(np.exp(0.3j) * su2_mat) - so3_mat).max() < ACCURACY
        lifted = so3_to_su2(so3_mat)
        overlap = np.vdot(su2_mat, lifted)
        assert np.abs(su2_mat * (overlap / abs(overlap)) - lifted).max() < ACCURACY
        assert abs(overlap.imag) < ACCURACY and lifted.trace().real >= 0
