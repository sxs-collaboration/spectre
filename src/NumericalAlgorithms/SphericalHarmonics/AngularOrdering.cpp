// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "NumericalAlgorithms/SphericalHarmonics/AngularOrdering.hpp"

#include <string>

#include "Options/Options.hpp"
#include "Options/ParseOptions.hpp"
#include "Utilities/ErrorHandling/Error.hpp"

namespace ylm {
std::ostream& operator<<(std::ostream& os, const AngularOrdering ordering) {
  switch (ordering) {
    case AngularOrdering::Strahlkorper:
      return os << "Strahlkorper";
    case AngularOrdering::Cce:
      return os << "Cce";
    default:
      ERROR("Unknown AngularOrdering type");
  }
}
}  // namespace ylm

template <>
ylm::AngularOrdering
Options::create_from_yaml<ylm::AngularOrdering>::create<void>(
    const Options::Option& options) {
  const auto ordering = options.parse_as<std::string>();
  if (ordering == "Strahlkorper") {
    return ylm::AngularOrdering::Strahlkorper;
  } else if (ordering == "Cce") {
    return ylm::AngularOrdering::Cce;
  }
  PARSE_ERROR(options.context(),
              "AngularOrdering must be 'Strahlkorper' or 'Cce'");
}
