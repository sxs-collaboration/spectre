// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "NumericalAlgorithms/FiniteDifference/FallbackReconstructorType.hpp"

#include <ostream>
#include <string>

#include "Options/Options.hpp"
#include "Options/ParseOptions.hpp"
#include "Utilities/ErrorHandling/Error.hpp"

std::ostream& fd::reconstruction::operator<<(
    std::ostream& os,
    const fd::reconstruction::FallbackReconstructorType recons_type) {
  switch (recons_type) {
    case fd::reconstruction::FallbackReconstructorType::Minmod:
      return os << "Minmod";
    case fd::reconstruction::FallbackReconstructorType::MonotonisedCentral:
      return os << "MonotonisedCentral";
    case fd::reconstruction::FallbackReconstructorType::None:
      return os << "None";
    default:  // LCOV_EXCL_LINE
      // LCOV_EXCL_START
      ERROR("Missing a case for operator<<(FallbackReconstructorType)");
      // LCOV_EXCL_STOP
  }
}

template <>
fd::reconstruction::FallbackReconstructorType
Options::create_from_yaml<fd::reconstruction::FallbackReconstructorType>::
    create<void>(const Options::Option& options) {
  const auto recons_type_read = options.parse_as<std::string>();
  if (recons_type_read == "Minmod") {
    return fd::reconstruction::FallbackReconstructorType::Minmod;
  } else if (recons_type_read == "MonotonisedCentral") {
    return fd::reconstruction::FallbackReconstructorType::MonotonisedCentral;
  } else if (recons_type_read == "None") {
    return fd::reconstruction::FallbackReconstructorType::None;
  }
  PARSE_ERROR(options.context(),
              "Failed to convert \""
                  << recons_type_read
                  << "\" to FallbackReconstructorType. Expected one of: "
                     "{Minmod, MonotonisedCentral, None}.");
}
