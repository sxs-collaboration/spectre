# Distributed under the MIT License.
# See LICENSE.txt for details.

import numpy as np

from Evolution.Systems.GeneralizedHarmonic.TestFunctions import (
    char_speed_upsi, char_speed_uzero, char_speed_uplus, char_speed_uminus)
from Evolution.Systems.GrMhd.ValenciaDivClean.TestFunctions import\
    characteristic_speeds as char_speeds_mhd


# Functions for testing Characteristics.cpp
def characteristic_speeds(lapse, shift, spatial_velocity,
                          spatial_velocity_sqrd, sound_speed_sqrd,
                          alfven_speed_sqrd, normal_oneform,
                          gh_constraint_gamma1):
    return list(
        np.append(
            char_speeds_mhd(lapse, shift, spatial_velocity,
                            spatial_velocity_sqrd, sound_speed_sqrd,
                            alfven_speed_sqrd, normal_oneform),
            np.array([
                char_speed_upsi(gh_constraint_gamma1, lapse, shift,
                                normal_oneform),
                char_speed_uzero(gh_constraint_gamma1, lapse, shift,
                                 normal_oneform),
                char_speed_uplus(gh_constraint_gamma1, lapse, shift,
                                 normal_oneform),
                char_speed_uminus(gh_constraint_gamma1, lapse, shift,
                                  normal_oneform)
            ])))


# End functions for testing Characteristics.cpp
