// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "NumericalAlgorithms/SpatialDiscretization/Quadrature.hpp"

#include <ostream>
#include <vector>

#include "Options/Options.hpp"
#include "Options/ParseOptions.hpp"
#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/GetOutput.hpp"
#include "Utilities/StdHelpers.hpp"

namespace {
std::vector<SpatialDiscretization::Quadrature> known_quadratures() {
  return std::vector{SpatialDiscretization::Quadrature::Gauss,
                     SpatialDiscretization::Quadrature::GaussLobatto,
                     SpatialDiscretization::Quadrature::CellCentered,
                     SpatialDiscretization::Quadrature::FaceCentered,
                     SpatialDiscretization::Quadrature::Equiangular};
}
}  // namespace

namespace SpatialDiscretization {
std::ostream& operator<<(std::ostream& os, const Quadrature& quadrature) {
  switch (quadrature) {
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
}  // namespace SpatialDiscretization

template <>
SpatialDiscretization::Quadrature
Options::create_from_yaml<SpatialDiscretization::Quadrature>::create<void>(
    const Options::Option& options) {
  const auto type_read = options.parse_as<std::string>();
  for (const auto quadrature : known_quadratures()) {
    if (type_read == get_output(quadrature)) {
      return quadrature;
    }
  }
  using ::operator<<;
  PARSE_ERROR(options.context(),
              "Failed to convert \""
                  << type_read
                  << "\" to SpatialDiscretization::Quadrature.\nMust be one of "
                  << known_quadratures() << ".");
}
