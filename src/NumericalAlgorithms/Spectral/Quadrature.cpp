// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "NumericalAlgorithms/Spectral/Quadrature.hpp"

#include <array>
#include <ostream>
#include <string>

#include "Options/Options.hpp"
#include "Options/ParseOptions.hpp"
#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/GetOutput.hpp"
#include "Utilities/StdHelpers.hpp"

namespace Spectral {
std::array<Quadrature, 6> all_quadratures() {
  return std::array{Quadrature::Uninitialized, Quadrature::Gauss,
                    Quadrature::GaussLobatto,  Quadrature::CellCentered,
                    Quadrature::FaceCentered,  Quadrature::Equiangular};
}

Quadrature to_quadrature(const std::string& quadrature) {
  for (const auto the_quadrature : all_quadratures()) {
    if (quadrature == get_output(the_quadrature)) {
      return the_quadrature;
    }
  }
  using ::operator<<;
  ERROR("Failed to convert \"" << quadrature
                               << "\" to Spectral::Quadrature.\nMust be one of "
                               << all_quadratures() << ".");
}

std::ostream& operator<<(std::ostream& os, const Quadrature& quadrature) {
  switch (quadrature) {
    case Quadrature::Uninitialized:
      return os << "Uninitialized";
    case Quadrature::Gauss:
      return os << "Gauss";
    case Quadrature::GaussLobatto:
      return os << "GaussLobatto";
    case Quadrature::CellCentered:
      return os << "CellCentered";
    case Quadrature::FaceCentered:
      return os << "FaceCentered";
    case Quadrature::Equiangular:
      return os << "Equiangular";
    default:
      ERROR("Invalid quadrature");
  }
}
}  // namespace Spectral

template <>
Spectral::Quadrature
Options::create_from_yaml<Spectral::Quadrature>::create<void>(
    const Options::Option& options) {
  const auto type_read = options.parse_as<std::string>();
  for (const auto quadrature : Spectral::all_quadratures()) {
    if (type_read == get_output(quadrature)) {
      return quadrature;
    }
  }
  using ::operator<<;
  PARSE_ERROR(options.context(),
              "Failed to convert \""
                  << type_read << "\" to Spectral::Quadrature.\nMust be one of "
                  << Spectral::all_quadratures() << ".");
}
