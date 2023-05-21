# Distributed under the MIT License.
# See LICENSE.txt for details.
import numpy as np


def ricci_scalar(ricci_tensor, inverse_metric):
    ricci_up_down = np.einsum("cb,ac", ricci_tensor, inverse_metric)
    return np.einsum("aa", ricci_up_down)
