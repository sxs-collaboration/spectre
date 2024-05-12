// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "NumericalAlgorithms/Spectral/Basis.hpp"

#include <array>
#include <ostream>
#include <string>

#include "Options/Options.hpp"
#include "Options/ParseOptions.hpp"
#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/GetOutput.hpp"
#include "Utilities/StdHelpers.hpp"

namespace Spectral {

std::array<Basis, 5> all_bases() {
  return std::array{Basis::Uninitialized, Basis::Chebyshev, Basis::Legendre,
                    Basis::FiniteDifference, Basis::SphericalHarmonic};
}

Basis to_basis(const std::string& basis) {
  for (const auto the_basis : all_bases()) {
    if (basis == get_output(the_basis)) {
      return the_basis;
    }
  }
  using ::operator<<;
  ERROR("Failed to convert \""
        << basis << "\" to Spectral::Basis.\nMust be one of " << all_bases()
        << ".");
}

std::ostream& operator<<(std::ostream& os, const Basis& basis) {
  switch (basis) {
    case Basis::Uninitialized:
      return os << "Uninitialized";
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
}  // namespace Spectral

template <>
Spectral::Basis Options::create_from_yaml<Spectral::Basis>::create<void>(
    const Options::Option& options) {
  const auto type_read = options.parse_as<std::string>();
  for (const auto basis : Spectral::all_bases()) {
    if (type_read == get_output(basis)) {
      return basis;
    }
  }
  using ::operator<<;
  PARSE_ERROR(options.context(),
              "Failed to convert \""
                  << type_read << "\" to Spectral::Basis.\nMust be one of "
                  << Spectral::all_bases() << ".");
}
