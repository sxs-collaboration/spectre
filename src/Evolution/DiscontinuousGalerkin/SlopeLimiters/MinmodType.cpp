// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/DiscontinuousGalerkin/SlopeLimiters/MinmodType.hpp"

#include <ostream>
#include <string>

#include "ErrorHandling/Error.hpp"
#include "Options/Options.hpp"
#include "Options/ParseOptions.hpp"

std::ostream& SlopeLimiters::operator<<(
    std::ostream& os, const SlopeLimiters::MinmodType& minmod_type) {
  switch (minmod_type) {
    case SlopeLimiters::MinmodType::LambdaPi1:
      return os << "LambdaPi1";
    case SlopeLimiters::MinmodType::LambdaPiN:
      return os << "LambdaPiN";
    case SlopeLimiters::MinmodType::Muscl:
      return os << "Muscl";
    default:  // LCOV_EXCL_LINE
      // LCOV_EXCL_START
      ERROR("Missing a case for operator<<(MinmodType)");
      // LCOV_EXCL_STOP
  }
}

template <>
SlopeLimiters::MinmodType
create_from_yaml<SlopeLimiters::MinmodType>::create<void>(
    const Option& options) {
  const std::string minmod_type_read = options.parse_as<std::string>();
  if (minmod_type_read == "LambdaPi1") {
    return SlopeLimiters::MinmodType::LambdaPi1;
  } else if (minmod_type_read == "LambdaPiN") {
    return SlopeLimiters::MinmodType::LambdaPiN;
  } else if (minmod_type_read == "Muscl") {
    return SlopeLimiters::MinmodType::Muscl;
  }
  PARSE_ERROR(options.context(), "Failed to convert \""
                                     << minmod_type_read
                                     << "\" to MinmodType. Expected one of: "
                                        "{LambdaPi1, LambdaPiN, Muscl}.");
}
