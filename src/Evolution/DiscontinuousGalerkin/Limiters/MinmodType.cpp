// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/DiscontinuousGalerkin/Limiters/MinmodType.hpp"

#include <ostream>
#include <string>

#include "Options/Options.hpp"
#include "Options/ParseOptions.hpp"
#include "Utilities/ErrorHandling/Error.hpp"

std::ostream& Limiters::operator<<(std::ostream& os,
                                   const Limiters::MinmodType minmod_type) {
  switch (minmod_type) {
    case Limiters::MinmodType::LambdaPi1:
      return os << "LambdaPi1";
    case Limiters::MinmodType::LambdaPiN:
      return os << "LambdaPiN";
    case Limiters::MinmodType::Muscl:
      return os << "Muscl";
    default:  // LCOV_EXCL_LINE
      // LCOV_EXCL_START
      ERROR("Missing a case for operator<<(MinmodType)");
      // LCOV_EXCL_STOP
  }
}

template <>
Limiters::MinmodType
Options::create_from_yaml<Limiters::MinmodType>::create<void>(
    const Options::Option& options) {
  const auto minmod_type_read = options.parse_as<std::string>();
  if (minmod_type_read == "LambdaPi1") {
    return Limiters::MinmodType::LambdaPi1;
  } else if (minmod_type_read == "LambdaPiN") {
    return Limiters::MinmodType::LambdaPiN;
  } else if (minmod_type_read == "Muscl") {
    return Limiters::MinmodType::Muscl;
  }
  PARSE_ERROR(options.context(), "Failed to convert \""
                                     << minmod_type_read
                                     << "\" to MinmodType. Expected one of: "
                                        "{LambdaPi1, LambdaPiN, Muscl}.");
}
