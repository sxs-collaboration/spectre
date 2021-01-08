// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Informer/Verbosity.hpp"

#include <ostream>
#include <string>

#include "Options/Options.hpp"
#include "Options/ParseOptions.hpp"
#include "Utilities/ErrorHandling/Error.hpp"

template <>
Verbosity Options::create_from_yaml<Verbosity>::create<void>(
    const Options::Option& options) {
  const auto type_read = options.parse_as<std::string>();
  if ("Silent" == type_read) {
    return Verbosity::Silent;
  } else if ("Quiet" == type_read) {
    return Verbosity::Quiet;
  } else if ("Verbose" == type_read) {
    return Verbosity::Verbose;
  } else if ("Debug" == type_read) {
    return Verbosity::Debug;
  }
  PARSE_ERROR(options.context(),
              "Failed to convert \"" << type_read
                                     << "\" to Verbosity. Must be one "
                                        "of Silent, Quiet, Verbose, or Debug.");
}

std::ostream& operator<<(std::ostream& os,
                         const Verbosity& verbosity) noexcept {
  switch (verbosity) {
    case Verbosity::Silent:
      return os << "Silent";
    case Verbosity::Quiet:
      return os << "Quiet";
    case Verbosity::Verbose:
      return os << "Verbose";
    case Verbosity::Debug:
      return os << "Debug";
    default:  // LCOV_EXCL_LINE
      // LCOV_EXCL_START
      ERROR("Need to add another case, don't understand value of 'verbosity'");
      // LCOV_EXCL_STOP
  }
}
