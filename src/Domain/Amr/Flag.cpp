// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Domain/Amr/Flag.hpp"
#include "Options/Options.hpp"
#include "Options/ParseOptions.hpp"
#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/GetOutput.hpp"
#include "Utilities/StdHelpers.hpp"

#include <ostream>
#include <vector>

namespace {
std::vector<amr::Flag> known_amr_flags() {
  return std::vector{amr::Flag::Undefined,          amr::Flag::Join,
                     amr::Flag::DecreaseResolution, amr::Flag::DoNothing,
                     amr::Flag::IncreaseResolution, amr::Flag::Split};
}
}  // namespace

namespace amr {

std::ostream& operator<<(std::ostream& os, const Flag& flag) {
  switch (flag) {
    case Flag::Undefined:
      os << "Undefined";
      break;
    case Flag::Join:
      os << "Join";
      break;
    case Flag::DecreaseResolution:
      os << "DecreaseResolution";
      break;
    case Flag::DoNothing:
      os << "DoNothing";
      break;
    case Flag::IncreaseResolution:
      os << "IncreaseResolution";
      break;
    case Flag::Split:
      os << "Split";
      break;
    default:
      ERROR("An undefined flag was passed to the stream operator.");
  }
  return os;
}
}  // namespace amr

template <>
amr::Flag Options::create_from_yaml<amr::Flag>::create<void>(
    const Options::Option& options) {
  const auto type_read = options.parse_as<std::string>();
  for (const auto flag : known_amr_flags()) {
    if (type_read == get_output(flag)) {
      return flag;
    }
  }
  using ::operator<<;
  PARSE_ERROR(options.context(), "Failed to convert \""
                                     << type_read
                                     << "\" to amr::Flag.\nMust be one of "
                                     << known_amr_flags() << ".");
}
