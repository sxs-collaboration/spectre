// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <string>
#include <vector>

#include "NumericalAlgorithms/SphericalHarmonics/Strahlkorper.hpp"
#include "Utilities/Gsl.hpp"

namespace ylm {
/*!
 * \brief Fills the legend and row of spherical harmonic data to write to disk
 *
 * The number of coefficients to write is based on `max_l`, the maximum value
 * that the input `strahlkorper` could possibly have. When
 * `strahlkorper.l_max() < max_l`, coefficients with \f$l\f$ higher than
 * `strahlkorper.l_max()` will simply be zero. Assuming the same `max_l` is
 * always used for a given surface, we will always write the same number of
 * columns for each row, as `max_l` sets the number of columns to write
 */
template <typename Frame>
void fill_ylm_legend_and_data(gsl::not_null<std::vector<std::string>*> legend,
                              gsl::not_null<std::vector<double>*> data,
                              const ylm::Strahlkorper<Frame>& strahlkorper,
                              double time, size_t max_l);
}  // namespace ylm
