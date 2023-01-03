# Distributed under the MIT License.
# See LICENSE.txt for details.


def sources(alpha, beta, field):
    return beta * (alpha * (1. + field) + 1.)**(-7)


def linearized_sources(alpha, beta, field, field_correction):
    return (-7. * alpha * beta * (alpha * (1. + field) + 1.)**(-8) *
            field_correction)
