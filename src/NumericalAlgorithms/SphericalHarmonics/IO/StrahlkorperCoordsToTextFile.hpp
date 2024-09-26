// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <cstddef>
#include <string>

#include "NumericalAlgorithms/SphericalHarmonics/AngularOrdering.hpp"
#include "NumericalAlgorithms/SphericalHarmonics/Strahlkorper.hpp"

namespace ylm {
/// @{
/*!
 * \brief Writes the collocation points of a `ylm::Strahlkorper` to an output
 * text file.
 *
 * \details The ordering of the points can be either the typical
 * `ylm::Spherepack` ordering or CCE ordering that works with libsharp. Also, an
 * error will occur if the output file already exists, but the output file can
 * be overwritten with the \p overwrite_file option.
 *
 * The second overload will construct a spherical `ylm::Strahlkorper` with the
 * given \p radius, \p l_max, and \p center.
 *
 * The output file format will be as follows with no comment or header lines,
 * a space as the delimiter, and decimals written in scientific notation:
 *
 * ```
 * x0 y0 z0
 * x1 y1 z1
 * x2 y2 z2
 * ...
 * ```
 */
template <typename Frame>
void write_strahlkorper_coords_to_text_file(
    const Strahlkorper<Frame>& strahlkorper,
    const std::string& output_file_name, AngularOrdering ordering,
    bool overwrite_file = false);

void write_strahlkorper_coords_to_text_file(double radius, size_t l_max,
                                            const std::array<double, 3>& center,
                                            const std::string& output_file_name,
                                            AngularOrdering ordering,
                                            bool overwrite_file = false);
/// @}
}  // namespace ylm
