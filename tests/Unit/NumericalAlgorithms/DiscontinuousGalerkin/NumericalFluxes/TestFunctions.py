# Distributed under the MIT License.
# See LICENSE.txt for details.

import numpy as np


def characteristic_speeds(var_1, var_2, var_3):
    return [np.cos(i) * var_1 - (1.0 - np.sin(i)) * np.dot(var_2, var_3)
            for i in range((var_2.size + 1) * (var_2.size + 1))]


# Functions for testing Hll.hpp
def hll_flux(ndotf_int, minus_ndotf_ext, var_int, var_ext,
             min_signal_speed, max_signal_speed):
    return ((max_signal_speed * ndotf_int + min_signal_speed * minus_ndotf_ext
             - max_signal_speed * min_signal_speed * (var_int - var_ext)) /
            (max_signal_speed - min_signal_speed))


def apply_var_1_hll_flux(ndotf_1_int, ndotf_2_int, ndotf_3_int, ndotf_4_int,
                         var_1_int, var_2_int, var_3_int, var_4_int,
                         minus_ndotf_1_ext, minus_ndotf_2_ext,
                         minus_ndotf_3_ext, minus_ndotf_4_ext,
                         var_1_ext, var_2_ext, var_3_ext, var_4_ext):
    char_speeds = (characteristic_speeds(var_1_int, var_2_int, var_3_int) +
                   characteristic_speeds(var_1_ext, var_2_ext, var_3_ext) +
                   [0.0])
    return hll_flux(ndotf_1_int, minus_ndotf_1_ext, var_1_int, var_1_ext,
                    min(char_speeds), max(char_speeds))


def apply_var_2_hll_flux(ndotf_1_int, ndotf_2_int, ndotf_3_int, ndotf_4_int,
                         var_1_int, var_2_int, var_3_int, var_4_int,
                         minus_ndotf_1_ext, minus_ndotf_2_ext,
                         minus_ndotf_3_ext, minus_ndotf_4_ext,
                         var_1_ext, var_2_ext, var_3_ext, var_4_ext):
    char_speeds = (characteristic_speeds(var_1_int, var_2_int, var_3_int) +
                   characteristic_speeds(var_1_ext, var_2_ext, var_3_ext) +
                   [0.0])
    return hll_flux(ndotf_2_int, minus_ndotf_2_ext, var_2_int, var_2_ext,
                    min(char_speeds), max(char_speeds))


def apply_var_3_hll_flux(ndotf_1_int, ndotf_2_int, ndotf_3_int, ndotf_4_int,
                         var_1_int, var_2_int, var_3_int, var_4_int,
                         minus_ndotf_1_ext, minus_ndotf_2_ext,
                         minus_ndotf_3_ext, minus_ndotf_4_ext,
                         var_1_ext, var_2_ext, var_3_ext, var_4_ext):
    char_speeds = (characteristic_speeds(var_1_int, var_2_int, var_3_int) +
                   characteristic_speeds(var_1_ext, var_2_ext, var_3_ext) +
                   [0.0])
    return hll_flux(ndotf_3_int, minus_ndotf_3_ext, var_3_int, var_3_ext,
                    min(char_speeds), max(char_speeds))


def apply_var_4_hll_flux(ndotf_1_int, ndotf_2_int, ndotf_3_int, ndotf_4_int,
                         var_1_int, var_2_int, var_3_int, var_4_int,
                         minus_ndotf_1_ext, minus_ndotf_2_ext,
                         minus_ndotf_3_ext, minus_ndotf_4_ext,
                         var_1_ext, var_2_ext, var_3_ext, var_4_ext):
    char_speeds = (characteristic_speeds(var_1_int, var_2_int, var_3_int) +
                   characteristic_speeds(var_1_ext, var_2_ext, var_3_ext) +
                   [0.0])
    return hll_flux(ndotf_4_int, minus_ndotf_4_ext, var_4_int, var_4_ext,
                    min(char_speeds), max(char_speeds))


# End functions for testing Hll.hpp


# Functions for testing LocalLaxFriedrichs.hpp
def max_abs_speed(var_1, var_2, var_3):
    return max([abs(x) for x in characteristic_speeds(var_1, var_2, var_3)])


def llf_flux(ndotf_int, minus_ndotf_ext, var_int, var_ext, max_speed):
    return 0.5 * (ndotf_int - minus_ndotf_ext + max_speed * (var_int - var_ext))


def apply_var_1_llf_flux(ndotf_1_int, ndotf_2_int, ndotf_3_int, ndotf_4_int,
                         var_1_int, var_2_int, var_3_int, var_4_int,
                         minus_ndotf_1_ext, minus_ndotf_2_ext,
                         minus_ndotf_3_ext, minus_ndotf_4_ext,
                         var_1_ext, var_2_ext, var_3_ext, var_4_ext):
    return llf_flux(ndotf_1_int, minus_ndotf_1_ext, var_1_int, var_1_ext,
                    max(max_abs_speed(var_1_int, var_2_int, var_3_int),
                        max_abs_speed(var_1_ext, var_2_ext, var_3_ext)))


def apply_var_2_llf_flux(ndotf_1_int, ndotf_2_int, ndotf_3_int, ndotf_4_int,
                         var_1_int, var_2_int, var_3_int, var_4_int,
                         minus_ndotf_1_ext, minus_ndotf_2_ext,
                         minus_ndotf_3_ext, minus_ndotf_4_ext,
                         var_1_ext, var_2_ext, var_3_ext, var_4_ext):
    return llf_flux(ndotf_2_int, minus_ndotf_2_ext, var_2_int, var_2_ext,
                    max(max_abs_speed(var_1_int, var_2_int, var_3_int),
                        max_abs_speed(var_1_ext, var_2_ext, var_3_ext)))


def apply_var_3_llf_flux(ndotf_1_int, ndotf_2_int, ndotf_3_int, ndotf_4_int,
                         var_1_int, var_2_int, var_3_int, var_4_int,
                         minus_ndotf_1_ext, minus_ndotf_2_ext,
                         minus_ndotf_3_ext, minus_ndotf_4_ext,
                         var_1_ext, var_2_ext, var_3_ext, var_4_ext):
    return llf_flux(ndotf_3_int, minus_ndotf_3_ext, var_3_int, var_3_ext,
                    max(max_abs_speed(var_1_int, var_2_int, var_3_int),
                        max_abs_speed(var_1_ext, var_2_ext, var_3_ext)))


def apply_var_4_llf_flux(ndotf_1_int, ndotf_2_int, ndotf_3_int, ndotf_4_int,
                         var_1_int, var_2_int, var_3_int, var_4_int,
                         minus_ndotf_1_ext, minus_ndotf_2_ext,
                         minus_ndotf_3_ext, minus_ndotf_4_ext,
                         var_1_ext, var_2_ext, var_3_ext, var_4_ext):
    return llf_flux(ndotf_4_int, minus_ndotf_4_ext, var_4_int, var_4_ext,
                    max(max_abs_speed(var_1_int, var_2_int, var_3_int),
                        max_abs_speed(var_1_ext, var_2_ext, var_3_ext)))


# End functions for testing LocalLaxFriedrichs.hpp
