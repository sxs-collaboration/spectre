## Distributed under the MIT License.
# See LICENSE.txt for details.

import Minmod
import MonotonisedCentral
import numpy as np
import Reconstruction


def _minmod(q):
    j = 1
    slope = Minmod.minmod(q[j] - q[j - 1], q[j + 1] - q[j])
    return [q[j] - 0.5 * slope, q[j] + 0.5 * slope]


def _mc(q):
    j = 1
    slope = MonotonisedCentral.monotonised_central(
        q[j] - q[j - 1], q[j + 1] - q[j]
    )
    return [q[j] - 0.5 * slope, q[j] + 0.5 * slope]


def _adaptive_order_5(
    q,
    keep_positive,
    low_order_recons,
    use_9th_order,
    use_7th_order,
    four_to_the_alpha_5,
    six_to_the_alpha_7,
    eight_to_the_alpha_9,
):
    j = 4 if use_9th_order else (3 if use_7th_order else 2)

    if use_9th_order:
        norm_top = (
            -1.593380762005595 * q[j + 1]
            + 0.7966903810027975 * q[j + 2]
            - 0.22762582314365648 * q[j + 3]
            + 0.02845322789295706 * q[j + 4]
            - 1.593380762005595 * q[j - 1]
            + 0.7966903810027975 * q[j - 2]
            - 0.22762582314365648 * q[j - 3]
            + 0.02845322789295706 * q[j - 4]
            + 1.991725952506994 * q[j]
        ) ** 2
        norm_full = (
            q[j + 1]
            * (
                25.393963433621668 * q[j + 1]
                - 31.738453392103736 * q[j + 2]
                + 14.315575523531798 * q[j + 3]
                - 5.422933317103013 * q[j + 4]
                + 45.309550145164756 * q[j - 1]
                - 25.682667845756164 * q[j - 2]
                + 10.394184200706238 * q[j - 3]
                - 3.5773996341558414 * q[j - 4]
                - 56.63693768145594 * q[j]
            )
            + q[j + 2]
            * (
                10.664627625179254 * q[j + 2]
                - 9.781510753231265 * q[j + 3]
                + 3.783820939683476 * q[j + 4]
                - 25.682667845756164 * q[j - 1]
                + 13.59830711617153 * q[j - 2]
                - 5.064486634342602 * q[j - 3]
                + 1.5850428636128617 * q[j - 4]
                + 33.99576779042882 * q[j]
            )
            + q[j + 3]
            * (
                2.5801312593878514 * q[j + 3]
                - 1.812843724346584 * q[j + 4]
                + 10.394184200706238 * q[j - 1]
                - 5.064486634342602 * q[j - 2]
                + 1.6716163773782988 * q[j - 3]
                - 0.4380794296257583 * q[j - 4]
                - 14.626643302060115 * q[j]
            )
            + q[j + 4]
            * (
                0.5249097623867759 * q[j + 4]
                - 3.5773996341558414 * q[j - 1]
                + 1.5850428636128617 * q[j - 2]
                - 0.4380794296257583 * q[j - 3]
                + 0.07624062080823268 * q[j - 4]
                + 5.336843456576288 * q[j]
            )
            + q[j - 1]
            * (
                25.393963433621668 * q[j - 1]
                - 31.738453392103736 * q[j - 2]
                + 14.315575523531798 * q[j - 3]
                - 5.422933317103013 * q[j - 4]
                - 56.63693768145594 * q[j]
            )
            + q[j - 2]
            * (
                10.664627625179254 * q[j - 2]
                - 9.781510753231265 * q[j - 3]
                + 3.783820939683476 * q[j - 4]
                + 33.99576779042882 * q[j]
            )
            + q[j - 3]
            * (
                2.5801312593878514 * q[j - 3]
                - 1.812843724346584 * q[j - 4]
                - 14.626643302060115 * q[j]
            )
            + q[j - 4]
            * (0.5249097623867759 * q[j - 4] + 5.336843456576288 * q[j])
            + 33.758463458609164 * q[j] ** 2
        )
        if eight_to_the_alpha_9**2 * norm_top <= norm_full:
            result = [
                -0.179443359375 * q[j + 1]
                + 0.0538330078125 * q[j + 2]
                - 0.010986328125 * q[j + 3]
                + 0.001068115234375 * q[j + 4]
                + 0.538330078125 * q[j - 1]
                - 0.0897216796875 * q[j - 2]
                + 0.015380859375 * q[j - 3]
                - 0.001373291015625 * q[j - 4]
                + 0.67291259765625 * q[j],
                0.538330078125 * q[j + 1]
                - 0.0897216796875 * q[j + 2]
                + 0.015380859375 * q[j + 3]
                - 0.001373291015625 * q[j + 4]
                - 0.179443359375 * q[j - 1]
                + 0.0538330078125 * q[j - 2]
                - 0.010986328125 * q[j - 3]
                + 0.001068115234375 * q[j - 4]
                + 0.67291259765625 * q[j],
            ]
            if (not keep_positive) or (result[0] > 0.0 and result[1] > 0.0):
                return result

    if use_7th_order:
        norm_top = (
            2
            * (
                16807 * q[j - 3] / 95040
                - 16807 * q[j - 2] / 15840
                + 16807 * q[j - 1] / 6336
                - 16807 * q[j] / 4752
                + 16807 * q[j + 1] / 6336
                - 16807 * q[j + 2] / 15840
                + 16807 * q[j + 3] / 95040
            )
            ** 2
            / 13
        )
        norm_full = (
            q[j + 1]
            * (
                3.93094886671763 * q[j + 1]
                - 4.4887583031366605 * q[j + 2]
                + 2.126671427664419 * q[j + 3]
                + 6.081742742499426 * q[j - 1]
                - 3.1180508323787337 * q[j - 2]
                + 1.2660604719155235 * q[j - 3]
                - 8.108990323332568 * q[j]
            )
            + q[j + 2]
            * (
                1.7504056695205172 * q[j + 2]
                - 1.402086588589091 * q[j + 3]
                - 3.1180508323787337 * q[j - 1]
                + 1.384291080027286 * q[j - 2]
                - 0.46498946172145633 * q[j - 3]
                + 4.614303600090953 * q[j]
            )
            + q[j + 3]
            * (
                0.5786954880513824 * q[j + 3]
                + 1.2660604719155235 * q[j - 1]
                - 0.46498946172145633 * q[j - 2]
                + 0.10352871936656591 * q[j - 3]
                - 2.0705743873313183 * q[j]
            )
            + q[j - 1]
            * (
                3.93094886671763 * q[j - 1]
                - 4.4887583031366605 * q[j - 2]
                + 2.126671427664419 * q[j - 3]
                - 8.108990323332568 * q[j]
            )
            + q[j - 2]
            * (
                1.7504056695205172 * q[j - 2]
                - 1.402086588589091 * q[j - 3]
                + 4.614303600090953 * q[j]
            )
            + q[j - 3]
            * (0.5786954880513824 * q[j - 3] - 2.0705743873313183 * q[j])
            + 5.203166203165525 * q[j] ** 2
        )
        if six_to_the_alpha_7**2 * norm_top <= norm_full:
            result = [
                -175 * q[j + 1] / 1024
                + 21 * q[j + 2] / 512
                - 5 * q[j + 3] / 1024
                + 525 * q[j - 1] / 1024
                - 35 * q[j - 2] / 512
                + 7 * q[j - 3] / 1024
                + 175 * q[j] / 256,
                525 * q[j + 1] / 1024
                - 35 * q[j + 2] / 512
                + 7 * q[j + 3] / 1024
                - 175 * q[j - 1] / 1024
                + 21 * q[j - 2] / 512
                - 5 * q[j - 3] / 1024
                + 175 * q[j] / 256,
            ]
            if (not keep_positive) or (result[0] > 0.0 and result[1] > 0.0):
                return result

    norm_top = (
        0.2222222222222222
        * (
            -1.4880952380952381 * q[j + 1]
            + 0.37202380952380953 * q[j + 2]
            - 1.4880952380952381 * q[j - 1]
            + 0.37202380952380953 * q[j - 2]
            + 2.232142857142857 * q[j]
        )
        ** 2
    )
    norm_full = (
        q[j + 1]
        * (
            1.179711612654321 * q[j + 1]
            - 0.963946414792769 * q[j + 2]
            + 1.0904086750440918 * q[j - 1]
            - 0.5030502507716049 * q[j - 2]
            - 1.6356130125661377 * q[j]
        )
        + q[j + 2]
        * (
            0.6699388830329586 * q[j + 2]
            - 0.5030502507716049 * q[j - 1]
            + 0.154568572944224 * q[j - 2]
            + 0.927411437665344 * q[j]
        )
        + q[j - 1]
        * (
            1.179711612654321 * q[j - 1]
            - 0.963946414792769 * q[j - 2]
            - 1.6356130125661377 * q[j]
        )
        + q[j - 2] * (0.6699388830329586 * q[j - 2] + 0.927411437665344 * q[j])
        + 1.4061182415674602 * q[j] ** 2
    )
    if (four_to_the_alpha_5) ** 2 * norm_top <= norm_full:
        result = [
            -0.15625 * q[j + 1]
            + 0.0234375 * q[j + 2]
            + 0.46875 * q[j - 1]
            - 0.0390625 * q[j - 2]
            + 0.703125 * q[j],
            0.46875 * q[j + 1]
            - 0.0390625 * q[j + 2]
            - 0.15625 * q[j - 1]
            + 0.0234375 * q[j - 2]
            + 0.703125 * q[j],
        ]
        if (not keep_positive) or (result[0] > 0.0 and result[1] > 0.0):
            return result
    low_order_result = low_order_recons([q[j - 1], q[j], q[j + 1]])
    if (not keep_positive) or (
        low_order_result[0] > 0.0 and low_order_result[1] > 0.0
    ):
        return low_order_result
    return [q[j], q[j]]


def compute_face_values_t(
    recons_upper_of_cell,
    recons_lower_of_cell,
    v,
    i,
    j,
    k,
    dim_to_recons,
    keep_positive,
    low_order_recons,
    use_9th_order,
    use_7th_order,
    four_to_the_alpha_5,
    six_to_the_alpha_7,
    eight_to_the_alpha_9,
):
    stencil_half_width = 4 if use_9th_order else (3 if use_7th_order else 2)
    v_stencil = []

    if dim_to_recons == 0:
        for l in range(-stencil_half_width, stencil_half_width + 1):
            v_stencil.append(v[i + l, j, k])

        lower, upper = _adaptive_order_5(
            np.asarray(v_stencil),
            keep_positive,
            low_order_recons,
            use_9th_order,
            use_7th_order,
            four_to_the_alpha_5,
            six_to_the_alpha_7,
            eight_to_the_alpha_9,
        )
        recons_lower_of_cell.append(lower)
        recons_upper_of_cell.append(upper)
    if dim_to_recons == 1:
        for l in range(-stencil_half_width, stencil_half_width + 1):
            v_stencil.append(v[i, j + l, k])

        lower, upper = _adaptive_order_5(
            np.asarray(v_stencil),
            keep_positive,
            low_order_recons,
            use_9th_order,
            use_7th_order,
            four_to_the_alpha_5,
            six_to_the_alpha_7,
            eight_to_the_alpha_9,
        )
        recons_lower_of_cell.append(lower)
        recons_upper_of_cell.append(upper)
    if dim_to_recons == 2:
        for l in range(-stencil_half_width, stencil_half_width + 1):
            v_stencil.append(v[i, j, k + l])

        lower, upper = _adaptive_order_5(
            np.asarray(v_stencil),
            keep_positive,
            low_order_recons,
            use_9th_order,
            use_7th_order,
            four_to_the_alpha_5,
            six_to_the_alpha_7,
            eight_to_the_alpha_9,
        )
        recons_lower_of_cell.append(lower)
        recons_upper_of_cell.append(upper)


def test_adaptive_order_with_mc(
    u,
    extents,
    dim,
    use_9th_order,
    use_7th_order,
    four_to_the_alpha_5,
    six_to_the_alpha_7,
    eight_to_the_alpha_9,
):
    ghost_zone = 4 if use_9th_order else (3 if use_7th_order else 2)

    def compute_face_values(
        recons_upper_of_cell, recons_lower_of_cell, v, i, j, k, dim_to_recons
    ):
        return compute_face_values_t(
            recons_upper_of_cell,
            recons_lower_of_cell,
            v,
            i,
            j,
            k,
            dim_to_recons,
            False,
            _mc,
            use_9th_order,
            use_7th_order,
            four_to_the_alpha_5,
            six_to_the_alpha_7,
            eight_to_the_alpha_9,
        )

    return Reconstruction.reconstruct(
        u,
        extents,
        dim,
        [ghost_zone, ghost_zone, ghost_zone],
        compute_face_values,
    )


def test_adaptive_order_with_minmod(
    u,
    extents,
    dim,
    use_9th_order,
    use_7th_order,
    four_to_the_alpha_5,
    six_to_the_alpha_7,
    eight_to_the_alpha_9,
):
    ghost_zone = 4 if use_9th_order else (3 if use_7th_order else 2)

    def compute_face_values(
        recons_upper_of_cell, recons_lower_of_cell, v, i, j, k, dim_to_recons
    ):
        return compute_face_values_t(
            recons_upper_of_cell,
            recons_lower_of_cell,
            v,
            i,
            j,
            k,
            dim_to_recons,
            False,
            _minmod,
            use_9th_order,
            use_7th_order,
            four_to_the_alpha_5,
            six_to_the_alpha_7,
            eight_to_the_alpha_9,
        )

    return Reconstruction.reconstruct(
        u,
        extents,
        dim,
        [ghost_zone, ghost_zone, ghost_zone],
        compute_face_values,
    )


def test_positivity_preserving_adaptive_order_with_mc(
    u,
    extents,
    dim,
    use_9th_order,
    use_7th_order,
    four_to_the_alpha_5,
    six_to_the_alpha_7,
    eight_to_the_alpha_9,
):
    ghost_zone = 4 if use_9th_order else (3 if use_7th_order else 2)

    def compute_face_values(
        recons_upper_of_cell, recons_lower_of_cell, v, i, j, k, dim_to_recons
    ):
        return compute_face_values_t(
            recons_upper_of_cell,
            recons_lower_of_cell,
            v,
            i,
            j,
            k,
            dim_to_recons,
            True,
            _mc,
            use_9th_order,
            use_7th_order,
            four_to_the_alpha_5,
            six_to_the_alpha_7,
            eight_to_the_alpha_9,
        )

    return Reconstruction.reconstruct(
        u,
        extents,
        dim,
        [ghost_zone, ghost_zone, ghost_zone],
        compute_face_values,
    )


def test_positivity_preserving_adaptive_order_with_minmod(
    u,
    extents,
    dim,
    use_9th_order,
    use_7th_order,
    four_to_the_alpha_5,
    six_to_the_alpha_7,
    eight_to_the_alpha_9,
):
    ghost_zone = 4 if use_9th_order else (3 if use_7th_order else 2)

    def compute_face_values(
        recons_upper_of_cell, recons_lower_of_cell, v, i, j, k, dim_to_recons
    ):
        return compute_face_values_t(
            recons_upper_of_cell,
            recons_lower_of_cell,
            v,
            i,
            j,
            k,
            dim_to_recons,
            True,
            _minmod,
            use_9th_order,
            use_7th_order,
            four_to_the_alpha_5,
            six_to_the_alpha_7,
            eight_to_the_alpha_9,
        )

    return Reconstruction.reconstruct(
        u,
        extents,
        dim,
        [ghost_zone, ghost_zone, ghost_zone],
        compute_face_values,
    )
