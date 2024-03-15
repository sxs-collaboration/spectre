# Distributed under the MIT License.
# See LICENSE.tcoordst for details.

import numpy as np

a = 0.5
B0 = 1.0
M = 1.0


def TildeE(coords, spin):
    return coords * 0.0


def TildeB(coords, spin):
    x, y, z = coords
    r = np.sqrt(np.dot(coords, coords))

    tildeb = coords * 0.0

    tildeb[0] = (
        a
        * B0
        * z
        * (
            (a * x - r * y)
            * (
                1 / r**4
                + 2
                * M
                * r
                * (r**2 - a**2)
                / (r**4 + a**2 * z**2) ** 2
            )
            + a
            * M
            * r
            * x
            * (
                (r**2 - z**2) / (r**4 * (r**4 + a**2 * z**2))
                - 4 * (r**2 + z**2) / (r**4 + a**2 * z**2) ** 2
            )
        )
    )

    tildeb[1] = (
        a
        * B0
        * z
        * (
            (r * x + a * y)
            * (
                1 / r**4
                + 2
                * M
                * r
                * (r**2 - a**2)
                / (r**4 + a**2 * z**2) ** 2
            )
            + a
            * M
            * r
            * y
            * (
                (r**2 - z**2) / (r**4 * (r**4 + a**2 * z**2))
                - 4 * (r**2 + z**2) / (r**4 + a**2 * z**2) ** 2
            )
        )
    )

    tildeb[2] = B0 * (
        1
        + a**2 * z**2 / r**4
        + (M * a**2 / r**3)
        * (
            1
            - z**2
            * (a**2 + z**2)
            * (5 * r**4 + a**2 * z**2)
            / (r**4 + a**2 * z**2) ** 2
        )
    )

    return tildeb


def TildePsi(coords, spin):
    return 0.0


def TildePhi(coords, spin):
    return 0.0


def TildeQ(coords, spin):
    return 0.0
