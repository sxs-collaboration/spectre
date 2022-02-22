// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "IO/Importers/ObservationSelector.hpp"

#include <ostream>
#include <string>

#include "Options/ParseOptions.hpp"
#include "Utilities/ErrorHandling/Error.hpp"

namespace importers {
std::ostream& operator<<(std::ostream& os, const ObservationSelector value) {
  switch (value) {
    case ObservationSelector::First:
      return os << "First";
    case ObservationSelector::Last:
      return os << "Last";
      // LCOV_EXCL_START
    default:
      ERROR("Unknown importers::ObservationSelector");
      // LCOV_EXCL_STOP
  }
}
}  // namespace importers

template <>
importers::ObservationSelector
Options::create_from_yaml<importers::ObservationSelector>::create<void>(
    const Options::Option& options) {
  const auto value = options.parse_as<std::string>();
  if (value == "First") {
    return importers::ObservationSelector::First;
  } else if (value == "Last") {
    return importers::ObservationSelector::Last;
  }
  PARSE_ERROR(options.context(), "Failed to convert '"
                                     << value
                                     << "' to importers::ObservationSelector. "
                                        "Must be one of First, Last.");
}
