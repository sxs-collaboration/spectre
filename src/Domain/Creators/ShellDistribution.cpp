// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Domain/Creators/ShellDistribution.hpp"

#include <algorithm>
#include <cstddef>
#include <string>
#include <variant>
#include <vector>

#include "Domain/CoordinateMaps/Distribution.hpp"
#include "Options/ParseError.hpp"
#include "Utilities/StdHelpers.hpp"

namespace domain::creators {
namespace {
struct DistributionVisitor {
  size_t num_shells;

  std::vector<domain::CoordinateMaps::Distribution> operator()(
      const domain::CoordinateMaps::Distribution distribution) const {
    // NOLINTNEXTLINE(modernize-return-braced-init-list)
    return std::vector<domain::CoordinateMaps::Distribution>(num_shells,
                                                             distribution);
  }

  std::vector<domain::CoordinateMaps::Distribution> operator()(
      const std::vector<domain::CoordinateMaps::Distribution>& distributions)
      const {
    return distributions;
  }
};
}  // namespace

void set_shell_distribution(
    const gsl::not_null<size_t*> number_of_shells,
    const gsl::not_null<std::vector<domain::CoordinateMaps::Distribution>*>
        radial_distribution,
    const std::vector<double>& radial_partitioning,
    const std::variant<domain::CoordinateMaps::Distribution,
                       std::vector<domain::CoordinateMaps::Distribution>>&
        input_radial_distribution,
    const double innermost_shell_radius, const double outermost_shell_radius,
    const std::string& innermost_radius_name,
    const std::string& outermost_radius_name, const Options::Context& context) {
  if (not std::is_sorted(radial_partitioning.begin(),
                         radial_partitioning.end())) {
    PARSE_ERROR(context,
                "Specify radial partitioning in ascending order. Specified "
                "radial partitioning is: "
                    << radial_partitioning);
  }

  if (not radial_partitioning.empty()) {
    if (radial_partitioning.front() <= innermost_shell_radius) {
      PARSE_ERROR(context, "First radial partition must be larger than the "
                               << innermost_radius_name << " radius, but is: "
                               << innermost_shell_radius);
    }
    if (radial_partitioning.back() >= outermost_shell_radius) {
      PARSE_ERROR(context, "Last radial partition must be smaller than the "
                               << outermost_radius_name << " radius, but is: "
                               << outermost_shell_radius);
    }
    const auto duplicate = std::adjacent_find(radial_partitioning.begin(),
                                              radial_partitioning.end());
    if (duplicate != radial_partitioning.end()) {
      PARSE_ERROR(context, "Radial partitioning contains duplicate element: "
                               << *duplicate);
    }
  }

  (*number_of_shells) = 1 + radial_partitioning.size();
  (*radial_distribution) = std::visit(DistributionVisitor{*number_of_shells},
                                      input_radial_distribution);
  if (radial_distribution->size() != *number_of_shells) {
    PARSE_ERROR(context,
                "Specify a 'RadialDistribution' for every spherical shell. You "
                "specified "
                    << radial_distribution->size()
                    << " items, but the domain has " << *number_of_shells
                    << " shells.");
  }
}
}  // namespace domain::creators
