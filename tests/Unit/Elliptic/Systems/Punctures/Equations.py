# Distributed under the MIT License.
# See LICENSE.txt for details.


def sources(alpha, beta, field):
    return beta * (alpha * (1.0 + field) + 1.0) ** (-7)


def linearized_sources(alpha, beta, field, field_correction):
    return (
        -7.0
        * alpha
        * beta
        * (alpha * (1.0 + field) + 1.0) ** (-8)
        * field_correction
    )
