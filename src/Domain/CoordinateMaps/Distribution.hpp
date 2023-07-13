// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <ostream>

/// \cond
namespace Options {
class Option;
template <typename T>
struct create_from_yaml;
}  // namespace Options
/// \endcond

namespace domain::CoordinateMaps {

/*!
 * \brief Distribution of grid points in one dimension
 *
 * Used to select a distribution of grid points in the input file.
 *
 * \see domain::CoordinateMaps::Wedge
 */
enum class Distribution {
  Linear,
  Equiangular,
  Logarithmic,
  Inverse,
  Projective
};

std::ostream& operator<<(std::ostream& os, Distribution distribution);

}  // namespace domain::CoordinateMaps

template <>
struct Options::create_from_yaml<domain::CoordinateMaps::Distribution> {
  template <typename Metavariables>
  static domain::CoordinateMaps::Distribution create(
      const Options::Option& options) {
    return create<void>(options);
  }
};
template <>
domain::CoordinateMaps::Distribution
Options::create_from_yaml<domain::CoordinateMaps::Distribution>::create<void>(
    const Options::Option& options);
