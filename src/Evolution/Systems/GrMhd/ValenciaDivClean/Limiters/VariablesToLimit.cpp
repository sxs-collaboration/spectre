// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/GrMhd/ValenciaDivClean/Limiters/VariablesToLimit.hpp"

#include <ostream>
#include <string>

#include "Options/Options.hpp"
#include "Options/ParseOptions.hpp"
#include "Utilities/ErrorHandling/Error.hpp"

std::ostream& grmhd::ValenciaDivClean::Limiters::operator<<(
    std::ostream& os, const grmhd::ValenciaDivClean::Limiters::VariablesToLimit
                          vars_to_limit) noexcept {
  switch (vars_to_limit) {
    case grmhd::ValenciaDivClean::Limiters::VariablesToLimit::Conserved:
      return os << "Conserved";
    case grmhd::ValenciaDivClean::Limiters::VariablesToLimit::
        NumericalCharacteristic:
      return os << "NumericalCharacteristic";
    default:  // LCOV_EXCL_LINE
      // LCOV_EXCL_START
      ERROR("Missing a case for operator<<(VariablesToLimit)");
      // LCOV_EXCL_STOP
  }
}

template <>
grmhd::ValenciaDivClean::Limiters::VariablesToLimit
Options::create_from_yaml<grmhd::ValenciaDivClean::Limiters::VariablesToLimit>::
    create<void>(const Options::Option& options) {
  const auto vars_to_limit_read_type = options.parse_as<std::string>();
  if (vars_to_limit_read_type == "Conserved") {
    return grmhd::ValenciaDivClean::Limiters::VariablesToLimit::Conserved;
  } else if (vars_to_limit_read_type == "NumericalCharacteristic") {
    return grmhd::ValenciaDivClean::Limiters::VariablesToLimit::
        NumericalCharacteristic;
  }
  PARSE_ERROR(options.context(),
              "Failed to convert \""
                  << vars_to_limit_read_type
                  << "\" to VariablesToLimit. Expected one of: "
                     "{Conserved, NumericalCharacteristic}.");
}
