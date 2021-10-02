// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/NewtonianEuler/Limiters/VariablesToLimit.hpp"

#include <ostream>
#include <string>

#include "Options/Options.hpp"
#include "Options/ParseOptions.hpp"
#include "Utilities/ErrorHandling/Error.hpp"

std::ostream& NewtonianEuler::Limiters::operator<<(
    std::ostream& os,
    const NewtonianEuler::Limiters::VariablesToLimit vars_to_limit) {
  switch (vars_to_limit) {
    case NewtonianEuler::Limiters::VariablesToLimit::Conserved:
      return os << "Conserved";
    case NewtonianEuler::Limiters::VariablesToLimit::Characteristic:
      return os << "Characteristic";
    case NewtonianEuler::Limiters::VariablesToLimit::NumericalCharacteristic:
      return os << "NumericalCharacteristic";
    default:  // LCOV_EXCL_LINE
      // LCOV_EXCL_START
      ERROR("Missing a case for operator<<(VariablesToLimit)");
      // LCOV_EXCL_STOP
  }
}

template <>
NewtonianEuler::Limiters::VariablesToLimit
Options::create_from_yaml<NewtonianEuler::Limiters::VariablesToLimit>::create<
    void>(const Options::Option& options) {
  const auto vars_to_limit_read_type = options.parse_as<std::string>();
  if (vars_to_limit_read_type == "Conserved") {
    return NewtonianEuler::Limiters::VariablesToLimit::Conserved;
  } else if (vars_to_limit_read_type == "Characteristic") {
    return NewtonianEuler::Limiters::VariablesToLimit::Characteristic;
  } else if (vars_to_limit_read_type == "NumericalCharacteristic") {
    return NewtonianEuler::Limiters::VariablesToLimit::NumericalCharacteristic;
  }
  PARSE_ERROR(options.context(),
              "Failed to convert \""
                  << vars_to_limit_read_type
                  << "\" to VariablesToLimit. Expected one of: "
                     "{Conserved, Characteristic, NumericalCharacteristic}.");
}
