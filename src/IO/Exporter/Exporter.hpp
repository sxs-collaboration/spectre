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

/// Identifies an observation by its ID in the volume data file.
struct ObservationId {
  explicit ObservationId(size_t local_value) : value(local_value) {}
  size_t value;
};

/// Identifies an observation by its index in the ordered list of observations.
/// Negative indices are counted from the end of the list.
struct ObservationStep {
  explicit ObservationStep(int local_value) : value(local_value) {}
  int value;
};

/*!
 * \brief Interpolate data in volume files to target points
 *
 * \tparam Dim Dimension of the domain
 * \param volume_files_or_glob The list of H5 files, or a glob pattern
 * \param subfile_name The name of the subfile in the H5 files containing the
 * volume data
 * \param observation Either the observation ID as a `size_t`, or the index of
 * the observation in the volume files to interpolate as an `int` (a value of 0
 * would be the first observation, and a value of -1 would be the last
 * observation).
 * \param tensor_components The tensor components to interpolate, e.g.
 * "Lapse", "Shift_x", "Shift_y", "Shift_z", "SpatialMetric_xx", etc.
 * Look into the H5 file to see what components are available.
 * \param target_points The points to interpolate to, in inertial coordinates.
 * \param extrapolate_into_excisions Enables extrapolation into excision regions
 * of the domain (default is `false`). This can be useful to fill the excision
 * region with (constraint-violating but smooth) data so it can be imported into
 * moving puncture codes. Specifically, we implement the strategy used in
 * \cite Etienne2008re adjusted for distorted excisions: we choose uniformly
 * spaced radial anchor points spaced as $\Delta r = 0.3 r_\mathrm{AH}$ in the
 * grid frame (where the excision is spherical), then map the anchor points to
 * the distorted frame (where we have the target point) and do a 7th order
 * polynomial extrapolation into the excision region.
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
    const std::string& subfile_name,
    const std::variant<ObservationId, ObservationStep>& observation,
    const std::vector<std::string>& tensor_components,
    const std::array<std::vector<double>, Dim>& target_points,
    bool extrapolate_into_excisions = false,
    std::optional<size_t> num_threads = std::nullopt);

}  // namespace spectre::Exporter
