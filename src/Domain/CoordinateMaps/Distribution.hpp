// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <optional>
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

/*!
 * \brief A `Distribution` and the corresponding singularity position
 *
 * The `singularity_position` is only meaningful for `Distribution::Logarithmic`
 * and `Distribution::Inverse`.
 *
 * This class can be option-created like so:
 * - Just the name of the distribution for `Linear`, `Equiangular`, and
 *  `Projective`.
 * - The name of the distribution and the singularity position for
 *   `Logarithmic` and `Inverse`:
 *
 * ```yaml
 * Logarithmic:
 *   SingularityPosition: 0.0
 * ```
 */
struct DistributionAndSingularityPosition {
  Distribution distribution = Distribution::Linear;
  std::optional<double> singularity_position = std::nullopt;
};

bool operator==(const DistributionAndSingularityPosition& lhs,
                const DistributionAndSingularityPosition& rhs);

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

template <>
struct Options::create_from_yaml<
    domain::CoordinateMaps::DistributionAndSingularityPosition> {
  template <typename Metavariables>
  static domain::CoordinateMaps::DistributionAndSingularityPosition create(
      const Options::Option& options) {
    return create<void>(options);
  }
};
template <>
domain::CoordinateMaps::DistributionAndSingularityPosition
Options::create_from_yaml<
    domain::CoordinateMaps::DistributionAndSingularityPosition>::
    create<void>(const Options::Option& options);
