# Distributed under the MIT License.
# See LICENSE.txt for details.

import numpy as np


def call_operator(coords, power):
    return np.power(coords, power)[0]


def first_deriv(coords, power):
    if power == 0.0:
        return np.array([0.0])
    else:
        return power * np.power(coords, power - 1.0)


def second_deriv(coords, power):
    if power == 0.0 or power == 1.0:
        return np.array([[0.0]])
    else:
        return np.array([(power - 1.0) * power * np.power(coords, power - 2.0)])


def third_deriv(coords, power):
    if power == 0.0 or power == 1.0 or power == 2.0:
        return np.array([[[0.0]]])
    else:
        return np.array(
            [
                [
                    (power - 2.0)
                    * (power - 1.0)
                    * power
                    * np.power(coords, power - 3.0)
                ]
            ]
        )
