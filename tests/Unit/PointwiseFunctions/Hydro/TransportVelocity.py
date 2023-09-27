# Distributed under the MIT License.
# See LICENSE.txt for details.

import numpy as np

# Functions testing TransportVelocity


def transport_velocity(spatial_velocity, lapse, shift):
    return spatial_velocity * lapse - shift


# End functions for testing TransportVelocity
