// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Time/LtsMode.hpp"

#include <ostream>
#include <string>

#include "Options/Options.hpp"
#include "Options/ParseError.hpp"
#include "Options/ParseOptions.hpp"
#include "Utilities/ErrorHandling/Error.hpp"

std::ostream& operator<<(std::ostream& os, LtsMode value) {
  switch (value) {
    case LtsMode::Off:
      return os << "Off";
    case LtsMode::Conservative:
      return os << "Conservative";
    default:
      ERROR("Unknown LtsMode: " << static_cast<int>(value));
  }
}

template <>
LtsMode Options::create_from_yaml<LtsMode>::create<void>(
    const Options::Option& options) {
  const auto value = options.parse_as<std::string>();
  if (value == "Off") {
    return LtsMode::Off;
  } else if (value == "Conservative") {
    return LtsMode::Conservative;
  }
  PARSE_ERROR(options.context(),
              "Invalid LtsMode '"
                  << value
                  << "'.  Valid choices are 'Off' and 'Conservative'.");
}
