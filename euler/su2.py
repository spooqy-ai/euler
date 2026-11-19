"""
Conversion between SU2 rotations (single-qubit unitaries) and the coresponding
SO3 rotations (actions of those unitaries on the Bloch sphere).
"""
from typing import Any, TypeAlias
import numpy as np
from .utils import Float, RotMatrix

SU2Matrix: TypeAlias = np.ndarray[
    tuple[int, int], np.dtype[np.complexfloating[Any]]
]
"""Type alias for 2-by-2 SU2 matrices."""

_PAULIS = (
    np.array([[0, 1], [1, 0]], dtype=np.complex128),
    np.array([[0, -1j], [1j, 0]], dtype=np.complex128),
    np.array([[1, 0], [0, -1]], dtype=np.complex128),
)
"""The Pauli matrices X, Y and Z."""


def su2_to_so3(mat: SU2Matrix, tol: Float = 1e-8) -> RotMatrix:
    """
    Converts an SU2 rotation to the corresponding SO3 rotation.

    The matrix is allowed to be a unitary with arbitrary global phase.
    The ``tol`` argument is unused, and is retained for backwards compatibility.
    """
    mat = np.asarray(mat, dtype=np.complex128)
    mat = mat / np.sqrt(abs(np.linalg.det(mat)))
    adj = mat.conj().T
    return np.array(
        [[(p @ mat @ q @ adj).trace().real / 2 for q in _PAULIS] for p in _PAULIS],
        dtype=np.float64,
    )


def so3_to_su2(mat: RotMatrix, tol: Float = 1e-8) -> SU2Matrix:
    """
    Converts an SO3 rotation to a corresponding SU2 rotation.

    Of the two SU2 rotations which differ by a sign, the one returned has non-negative
    real part of the trace; where the trace vanishes, the first non-zero amongst its
    X, Y and Z components is positive.
    The ``tol`` argument is unused, and is retained for backwards compatibility.
    """
    m = np.asarray(mat, dtype=np.float64)
    squares = (
        1 + m[0, 0] + m[1, 1] + m[2, 2],
        1 + m[0, 0] - m[1, 1] - m[2, 2],
        1 - m[0, 0] + m[1, 1] - m[2, 2],
        1 - m[0, 0] - m[1, 1] + m[2, 2],
    )
    k = int(np.argmax(squares))
    d = 2 * np.sqrt(squares[k])
    if k == 0:
        q = (d / 4, (m[2, 1] - m[1, 2]) / d, (m[0, 2] - m[2, 0]) / d, (m[1, 0] - m[0, 1]) / d)
    elif k == 1:
        q = ((m[2, 1] - m[1, 2]) / d, d / 4, (m[0, 1] + m[1, 0]) / d, (m[0, 2] + m[2, 0]) / d)
    elif k == 2:
        q = ((m[0, 2] - m[2, 0]) / d, (m[0, 1] + m[1, 0]) / d, d / 4, (m[1, 2] + m[2, 1]) / d)
    else:
        q = ((m[1, 0] - m[0, 1]) / d, (m[0, 2] + m[2, 0]) / d, (m[1, 2] + m[2, 1]) / d, d / 4)
    w, x, y, z = (float(v) for v in q)
    if next((v for v in (w, x, y, z) if v != 0), 1.0) < 0:
        w, x, y, z = -w, -x, -y, -z
    return np.array(
        [[w - 1j * z, -y - 1j * x], [y - 1j * x, w + 1j * z]], dtype=np.complex128
    )
