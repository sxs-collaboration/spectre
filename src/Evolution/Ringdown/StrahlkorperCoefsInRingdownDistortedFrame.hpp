// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <cstddef>
#include <string>
#include <vector>

#include "DataStructures/DataVector.hpp"

/*!
 * \brief Functionality for evolving a ringdown following a compact-binary
 * merger.
 */
namespace evolution::Ringdown {
/*!
 * \brief Transform `Strahlkorper` coefs to ringdown distorted frame.
 *
 * \details Reads Strahlkorper coefficients (assumed to be in the inertial
 * frame) from a file, then transforms them into the
 * ringdown distorted frame defined by the expansion and rotation maps
 * specified by `exp_func_and_2_derivs`, `exp_outer_bdry_func_and_2_derivs`,
 * and `rot_func_and_2_derivs`, which correspond to the ringdown frame's
 * expansion and rotation maps at the time given by `match_time`, and by
 * `settling_timescale`, the timescale for the maps to settle to constant
 * values. Only Strahlkorpers within `requested_number_of_times_from_end` times
 * from the final time are returned. This function is used to transition
 * from inspiral to ringdown; in this case, the inertial-frame Strahlkorper
 * is the common apparent horizon from a binary-black-hole inspiral;
 * the ringdown-distorted-frame coefficients are used to initialize
 * the shape map for the ringdown domain.
 */
std::vector<DataVector> strahlkorper_coefs_in_ringdown_distorted_frame(
    const std::string& path_to_horizons_h5,
    const std::string& surface_subfile_name,
    size_t requested_number_of_times_from_end, double match_time,
    double settling_timescale,
    const std::array<double, 3>& exp_func_and_2_derivs,
    const std::array<double, 3>& exp_outer_bdry_func_and_2_derivs,
    const std::vector<std::array<double, 4>>& rot_func_and_2_derivs);
}  // namespace evolution::Ringdown
