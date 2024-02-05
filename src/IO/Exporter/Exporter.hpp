// Distributed under the MIT License.
// See LICENSE.txt for details.

/// \file
/// Declares functions for interpolating data in volume files to target points.
/// This file is intended to be included in external programs, so it
/// intentionally has no dependencies on any other headers.

#pragma once

#include <array>
#include <cstddef>
#include <optional>
#include <string>
#include <variant>
#include <vector>

namespace spectre::Exporter {

/*!
 * \brief Interpolate data in volume files to target points
 *
 * \tparam Dim Dimension of the domain
 * \param volume_files_or_glob The list of H5 files, or a glob pattern
 * \param subfile_name The name of the subfile in the H5 files containing the
 * volume data
 * \param observation_step The index of the observation in the volume files
 * to interpolate. A value of 0 would be the first observation, and a value of
 * -1 would be the last observation.
 * \param tensor_components The tensor components to interpolate, e.g.
 * "Lapse", "Shift_x", "Shift_y", "Shift_z", "SpatialMetric_xx", etc.
 * Look into the H5 file to see what components are available.
 * \param target_points The points to interpolate to, in inertial coordinates.
 * \param num_threads The number of threads to use if OpenMP is linked in. If
 * not specified, OpenMP will determine the number of threads automatically.
 * It's also possible to set the number of threads using the environment
 * variable OMP_NUM_THREADS. It's an error to specify num_threads if OpenMP is
 * not linked in. Set num_threads to 1 to disable OpenMP.
 * \return std::vector<std::vector<double>> The interpolated data. The first
 * dimension corresponds to the selected tensor components, and the second
 * dimension corresponds to the target points.
 */
template <size_t Dim>
std::vector<std::vector<double>> interpolate_to_points(
    const std::variant<std::vector<std::string>, std::string>&
        volume_files_or_glob,
    std::string subfile_name, int observation_step,
    const std::vector<std::string>& tensor_components,
    const std::array<std::vector<double>, Dim>& target_points,
    std::optional<size_t> num_threads = std::nullopt);

}  // namespace spectre::Exporter
