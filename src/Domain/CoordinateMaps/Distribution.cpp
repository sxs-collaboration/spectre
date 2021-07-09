// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Domain/CoordinateMaps/Distribution.hpp"

#include <ostream>
#include <string>

#include "Options/Options.hpp"
#include "Options/ParseOptions.hpp"
#include "Utilities/ErrorHandling/Error.hpp"

namespace domain::CoordinateMaps {

std::ostream& operator<<(std::ostream& os,
                         const Distribution distribution) noexcept {
  switch (distribution) {
    case Distribution::Linear:
      return os << "Linear";
    case Distribution::Equiangular:
      return os << "Equiangular";
    case Distribution::Logarithmic:
      return os << "Logarithmic";
    case Distribution::Inverse:
      return os << "Inverse";
    default:
      ERROR("Unknown domain::CoordinateMaps::Distribution type");
  }
}

}  // namespace domain::CoordinateMaps

template <>
domain::CoordinateMaps::Distribution
Options::create_from_yaml<domain::CoordinateMaps::Distribution>::create<void>(
    const Options::Option& options) {
  const auto distribution = options.parse_as<std::string>();
  if (distribution == "Linear") {
    return domain::CoordinateMaps::Distribution::Linear;
  } else if (distribution == "Equiangular") {
    return domain::CoordinateMaps::Distribution::Equiangular;
  } else if (distribution == "Logarithmic") {
    return domain::CoordinateMaps::Distribution::Logarithmic;
  } else if (distribution == "Inverse") {
    return domain::CoordinateMaps::Distribution::Inverse;
  }
  PARSE_ERROR(options.context(),
              "Distribution must be 'Linear', 'Logarithmic' or 'Inverse'");
}
