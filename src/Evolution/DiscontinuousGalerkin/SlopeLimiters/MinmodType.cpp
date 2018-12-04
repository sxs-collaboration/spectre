// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/DiscontinuousGalerkin/SlopeLimiters/MinmodType.hpp"

#include <ostream>
#include <string>

#include "Options/Options.hpp"
#include "Options/ParseOptions.hpp"

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
