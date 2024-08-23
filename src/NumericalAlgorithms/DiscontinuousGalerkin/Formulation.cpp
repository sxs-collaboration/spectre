// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "NumericalAlgorithms/DiscontinuousGalerkin/Formulation.hpp"

#include <ostream>
#include <string>

#include "Options/ParseOptions.hpp"
#include "Utilities/ErrorHandling/Error.hpp"

namespace dg {
std::ostream& operator<<(std::ostream& os, const Formulation t) {
  switch (t) {
    case Formulation::StrongInertial:
      return os << "StrongInertial";
    case Formulation::WeakInertial:
      return os << "WeakInertial";
    case Formulation::StrongLogical:
      return os << "StrongLogical";
    default:
      ERROR("Unknown DG formulation.");
  }
}
}  // namespace dg

template <>
dg::Formulation Options::create_from_yaml<dg::Formulation>::create<void>(
    const Options::Option& options) {
  const auto type_read = options.parse_as<std::string>();
  if ("StrongInertial" == type_read) {
    return dg::Formulation::StrongInertial;
  } else if ("WeakInertial" == type_read) {
    return dg::Formulation::WeakInertial;
  } else if ("StrongLogical" == type_read) {
    return dg::Formulation::StrongLogical;
  }
  PARSE_ERROR(options.context(),
              "Failed to convert \""
                  << type_read
                  << "\" to dg::Formulation. Must be one of "
                     "StrongInertial, WeakInertial, or StrongLogical.");
}
