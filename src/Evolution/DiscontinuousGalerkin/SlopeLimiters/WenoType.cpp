// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/DiscontinuousGalerkin/SlopeLimiters/WenoType.hpp"

#include <ostream>
#include <string>

#include "ErrorHandling/Error.hpp"
#include "Options/Options.hpp"
#include "Options/ParseOptions.hpp"

std::ostream& SlopeLimiters::operator<<(
    std::ostream& os, const SlopeLimiters::WenoType weno_type) noexcept {
  switch (weno_type) {
    case SlopeLimiters::WenoType::Hweno:
      return os << "Hweno";
    case SlopeLimiters::WenoType::SimpleWeno:
      return os << "SimpleWeno";
    default:  // LCOV_EXCL_LINE
      // LCOV_EXCL_START
      ERROR("Missing a case for operator<<(WenoType)");
      // LCOV_EXCL_STOP
  }
}

template <>
SlopeLimiters::WenoType create_from_yaml<SlopeLimiters::WenoType>::create<void>(
    const Option& options) {
  const std::string weno_type_read = options.parse_as<std::string>();
  if (weno_type_read == "Hweno") {
    return SlopeLimiters::WenoType::Hweno;
  } else if (weno_type_read == "SimpleWeno") {
    return SlopeLimiters::WenoType::SimpleWeno;
  }
  PARSE_ERROR(options.context(), "Failed to convert \""
                                     << weno_type_read
                                     << "\" to WenoType. Expected one of: "
                                        "{Hweno, SimpleWeno}.");
}
