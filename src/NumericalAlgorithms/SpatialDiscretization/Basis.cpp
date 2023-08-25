// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "NumericalAlgorithms/SpatialDiscretization/Basis.hpp"

#include <ostream>
#include <vector>

#include "Options/Options.hpp"
#include "Options/ParseOptions.hpp"
#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/GetOutput.hpp"
#include "Utilities/StdHelpers.hpp"

namespace {
std::vector<SpatialDiscretization::Basis> known_bases() {
  return std::vector{SpatialDiscretization::Basis::Legendre,
                     SpatialDiscretization::Basis::Chebyshev,
                     SpatialDiscretization::Basis::FiniteDifference,
                     SpatialDiscretization::Basis::SphericalHarmonic};
}
}  // namespace

namespace SpatialDiscretization {
std::ostream& operator<<(std::ostream& os, const Basis& basis) {
  switch (basis) {
    case Basis::Legendre:
      return os << "Legendre";
    case Basis::Chebyshev:
      return os << "Chebyshev";
    case Basis::FiniteDifference:
      return os << "FiniteDifference";
    case Basis::SphericalHarmonic:
      return os << "SphericalHarmonic";
    default:
      ERROR("Invalid basis");
  }
}
}  // namespace SpatialDiscretization

template <>
SpatialDiscretization::Basis
Options::create_from_yaml<SpatialDiscretization::Basis>::create<void>(
    const Options::Option& options) {
  const auto type_read = options.parse_as<std::string>();
  for (const auto basis : known_bases()) {
    if (type_read == get_output(basis)) {
      return basis;
    }
  }
  using ::operator<<;
  PARSE_ERROR(options.context(),
              "Failed to convert \""
                  << type_read
                  << "\" to SpatialDiscretization::Basis.\nMust be one of "
                  << known_bases() << ".");
}
