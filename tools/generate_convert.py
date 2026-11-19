"""
Generates ``euler/convert.py``.

Each conversion computes the unit quaternion of the rotation from the half-angles of
its Euler angles, then reads the new Euler angles off the quaternion with arctangents.
Every generated function is self-contained, so that it can be compiled on its own.

Usage: ``python tools/generate_convert.py > euler/convert.py``
"""

from __future__ import annotations
from itertools import product

AXIS_TRIPLES = (
    "xzx", "xyx", "yxy", "yzy", "zyz", "zxz",
    "xzy", "xyz", "yxz", "yzx", "zyx", "zxy",
)  # fmt: skip

Poly = dict[tuple[str, ...], int]
"""Polynomial in the half-angle sines and cosines, as a map from monomials to coefficients."""

Quaternion = tuple[Poly, Poly, Poly, Poly]
"""Quaternion with polynomial components, scalar part first."""


def poly_mul(p: Poly, q: Poly) -> Poly:
    """Multiplies two polynomials."""
    result: Poly = {}
    for (m1, k1), (m2, k2) in product(p.items(), q.items()):
        monomial = tuple(sorted(m1 + m2))
        result[monomial] = result.get(monomial, 0) + k1 * k2
    return {m: k for m, k in result.items() if k != 0}


def poly_add(*terms: tuple[int, Poly]) -> Poly:
    """Computes a signed sum of polynomials."""
    result: Poly = {}
    for sign, p in terms:
        for m, k in p.items():
            result[m] = result.get(m, 0) + sign * k
    return {m: k for m, k in result.items() if k != 0}


def quat_mul(p: Quaternion, q: Quaternion) -> Quaternion:
    """Computes the Hamilton product of two quaternions."""
    w1, x1, y1, z1 = p
    w2, x2, y2, z2 = q
    m = poly_mul
    return (
        poly_add((1, m(w1, w2)), (-1, m(x1, x2)), (-1, m(y1, y2)), (-1, m(z1, z2))),
        poly_add((1, m(w1, x2)), (1, m(x1, w2)), (1, m(y1, z2)), (-1, m(z1, y2))),
        poly_add((1, m(w1, y2)), (1, m(y1, w2)), (1, m(z1, x2)), (-1, m(x1, z2))),
        poly_add((1, m(w1, z2)), (1, m(z1, w2)), (1, m(x1, y2)), (-1, m(y1, x2))),
    )


def axis_rotation(axis: str, angle: str) -> Quaternion:
    """Builds the quaternion of the rotation about an axis, by the named angle."""
    components: list[Poly] = [{(f"c{angle}",): 1}, {}, {}, {}]
    components[1 + "xyz".index(axis)] = {(f"s{angle}",): 1}
    return (components[0], components[1], components[2], components[3])


def expression(p: Poly) -> str:
    """Renders a polynomial as source code."""
    if not p:
        return "0.0"
    text = ""
    for monomial, k in sorted(p.items()):
        assert abs(k) == 1
        sign = "-" if k < 0 else "+"
        text += f" {sign} " + " * ".join(monomial)
    return text[3:] if text.startswith(" + ") else "-" + text[3:]


def handedness(first: str, second: str) -> int:
    """Returns 1 if the second axis follows the first in cyclic order, and -1 otherwise."""
    return 1 if ("xyz".index(second) - "xyz".index(first)) % 3 == 1 else -1


def signed(sign: int, name: str) -> str:
    """Renders a signed variable."""
    return name if sign > 0 else f"-{name}"


WRAP = """\
    if {v} > pi:
        {v} -= 2 * pi
    elif {v} <= -pi:
        {v} += 2 * pi
"""


def function(p: str, q: str) -> str:
    """Generates the source code of one conversion function."""
    quaternion = quat_mul(
        quat_mul(axis_rotation(p[0], "a"), axis_rotation(p[1], "b")),
        axis_rotation(p[2], "c"),
    )
    lines = [
        f"def convert_{p}_{q}(",
        "    a: Float, b: Float, c: Float, tol: Float = 1e-8",
        ") -> tuple[float, float, float]:",
        f'    """Convert Euler angles from {p.upper()} to {q.upper()}."""',
        "    sa, ca = sin(a / 2), cos(a / 2)",
        "    sb, cb = sin(b / 2), cos(b / 2)",
        "    sc, cc = sin(c / 2), cos(c / 2)",
    ]
    is_proper = q[0] == q[2]
    prefix = "u" if is_proper else "t"
    for name, component in zip("wxyz", quaternion):
        lines.append(f"    {prefix}{name} = {expression(component)}")
    if is_proper:
        outer, middle = q[0], q[1]
        third = next(axis for axis in "xyz" if axis not in q)
        sign = handedness(outer, middle)
        low, high = "0.0", "pi"
    else:
        outer, middle, third = q[0], q[1], q[2]
        sign = handedness(outer, middle)
        other = "xyz"["xyz".index(middle)]
        # Right-multiplication by a quarter turn about the middle axis.
        turn = quat_mul(
            ({("tw",): 1}, {("tx",): 1}, {("ty",): 1}, {("tz",): 1}),
            tuple(  # type: ignore[arg-type]
                {("1",): 1} if i == 0 or "wxyz"[i] == other else {} for i in range(4)
            ),
        )
        for name, component in zip("wxyz", turn):
            rendered = expression(
                {tuple(v for v in m if v != "1"): k for m, k in component.items()}
            )
            lines.append(f"    u{name} = ({rendered}) * SQRT_HALF")
        low, high = "-pi / 2", "pi / 2"
    negate_last = not is_proper and sign > 0
    lines += [
        f"    m_sin = hypot(u{middle}, u{third})",
        f"    m_cos = hypot(uw, u{outer})",
        "    if 2 * m_sin < tol:",
        f"        a = 2 * arctan2(u{outer}, uw)",
        f"        b = {low}",
        "        c = 0.0",
        "    elif 2 * m_cos < tol:",
        f"        a = 2 * arctan2({signed(sign, 'u' + third)}, u{middle})",
        f"        b = {high}",
        "        c = 0.0",
        "    else:",
        f"        half_sum = arctan2(u{outer}, uw)",
        f"        half_diff = arctan2({signed(sign, 'u' + third)}, u{middle})",
        "        a = half_sum + half_diff",
        "        b = 2 * arctan2(m_sin, m_cos)" + ("" if is_proper else " - pi / 2"),
        "        c = "
        + ("half_diff - half_sum" if negate_last else "half_sum - half_diff"),
    ]
    body = "\n".join(lines) + "\n" + WRAP.format(v="a") + WRAP.format(v="c")
    return body + "    return float(a), float(b), float(c)\n"


HEADER = '''"""
Functions to convert Euler angles between basis triples.

This file is generated by ``tools/generate_convert.py``: do not edit it by hand.

Each function computes the unit quaternion of the rotation from the half-angles, and
reads the new angles off it with arctangents, so that accuracy is retained arbitrarily
close to gimbal lock. The rotation is treated as being at gimbal lock when the new
middle angle is within ``tol`` radians of it, in which case the middle angle is set to
its value at gimbal lock and the third angle is set to zero.
"""

from __future__ import annotations

from numpy import pi, sin, cos, hypot, arctan2
from .utils import Float, AxisTriple

SQRT_HALF = 0.7071067811865476
"""Square root of one half."""

'''

FOOTER = '''
def convert(
    p: AxisTriple, q: AxisTriple, a: Float, b: Float, c: Float, tol: Float = 1e-8
) -> tuple[float, float, float]:
    """Convert Euler angles between given basis triples."""
    return globals()[f"convert_{p}_{q}"](a, b, c, tol=tol)  # type: ignore
'''


def main() -> None:
    """Prints the generated module."""
    print(HEADER, end="")
    for p, q in product(AXIS_TRIPLES, repeat=2):
        print("\n" + function(p, q))
    print(FOOTER, end="")


if __name__ == "__main__":
    main()
