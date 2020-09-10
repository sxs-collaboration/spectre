// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "NumericalAlgorithms/Convergence/Reason.hpp"

#include <ostream>
#include <string>

#include "ErrorHandling/Error.hpp"
#include "Options/Options.hpp"
#include "Options/ParseOptions.hpp"
#include "Utilities/GetOutput.hpp"

namespace Convergence {

std::ostream& operator<<(std::ostream& os, const Reason& reason) noexcept {
  switch (reason) {
    case Reason::MaxIterations:
      return os << "MaxIterations";
    case Reason::AbsoluteResidual:
      return os << "AbsoluteResidual";
    case Reason::RelativeResidual:
      return os << "RelativeResidual";
    default:
      ERROR("Unknown convergence reason");
  }
}

}  // namespace Convergence

template <>
Convergence::Reason
Options::create_from_yaml<Convergence::Reason>::create<void>(
    const Options::Option& options) {
  const std::string type_read = options.parse_as<std::string>();
  if (type_read == get_output(Convergence::Reason::MaxIterations)) {
    return Convergence::Reason::MaxIterations;
  } else if (type_read == get_output(Convergence::Reason::AbsoluteResidual)) {
    return Convergence::Reason::AbsoluteResidual;
  } else if (type_read == get_output(Convergence::Reason::RelativeResidual)) {
    return Convergence::Reason::RelativeResidual;
  }
  PARSE_ERROR(options.context(),
              "Failed to convert \""
                  << type_read << "\" to Convergence::Reason. Must be one of '"
                  << get_output(Convergence::Reason::MaxIterations) << "', '"
                  << get_output(Convergence::Reason::AbsoluteResidual)
                  << "' or '"
                  << get_output(Convergence::Reason::RelativeResidual) << "'.");
}
