// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <string>
#include <vector>

#include "NumericalAlgorithms/SphericalHarmonics/Strahlkorper.hpp"

namespace ylm {
/// \ingroup SurfacesGroup
/// \brief Returns a list of `ylm::Strahlkorper`s constructed from reading in
/// spherical harmonic data for a surface at a requested list of times
///
/// \details The `ylm::Strahlkorper`s are constructed by reading in data from
/// an H5 subfile that is expected to be in the format described by
/// `intrp::callbacks::ObserveSurfaceData`. It is assumed that
/// \f$l_{max} = m_{max}\f$.
///
/// \param file_name name of the H5 file containing the surface's spherical
/// harmonic data
/// \param surface_subfile_name name of the subfile (with no leading slash nor
/// the `.dat` extension) within `file_name` that contains the surface's
/// spherical harmonic data to read in
/// \param requested_number_of_times_from_end the number of times to read in
/// starting backwards from the final time found in `surface_subfile_name`
template <typename Frame>
std::vector<ylm::Strahlkorper<Frame>> read_surface_ylm(
    const std::string& file_name, const std::string& surface_subfile_name,
    size_t requested_number_of_times_from_end);

/*!
 * \brief Similar to `ylm::read_surface_ylm`, this reads in spherical harmonic
 * data for a surface and constructs a `ylm::Strahlkorper`. However, this
 * function only does it at a specific time and returns a single
 * `ylm::Strahlkorper`.
 *
 * \note If two times are found within \p epsilon of the \p time, then an error
 * will occur. Similarly, if no \p time is found within the \p epsilon, then an
 * error will occur as well.
 *
 * \param file_name name of the H5 file containing the surface's spherical
 * harmonic data
 * \param surface_subfile_name name of the subfile (with no leading slash nor
 * the `.dat` extension) within `file_name` that contains the surface's
 * spherical harmonic data to read in
 * \param time Time to read the coefficients at.
 * \param relative_epsilon How much error is allowed when looking for a specific
 * time. This is useful so users don't have to know the specific time to machine
 * precision.
 * \param check_frame Whether to check the frame in the subfile or not.
 */
template <typename Frame>
ylm::Strahlkorper<Frame> read_surface_ylm_single_time(
    const std::string& file_name, const std::string& surface_subfile_name,
    double time, double relative_epsilon, bool check_frame = true);
}  // namespace ylm
