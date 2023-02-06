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
std::vector<amr::domain::Flag> known_amr_flags() {
  return std::vector{
      amr::domain::Flag::Undefined,          amr::domain::Flag::Join,
      amr::domain::Flag::DecreaseResolution, amr::domain::Flag::DoNothing,
      amr::domain::Flag::IncreaseResolution, amr::domain::Flag::Split};
}
}  // namespace

namespace amr::domain {

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
}  // namespace amr::domain

template <>
amr::domain::Flag Options::create_from_yaml<amr::domain::Flag>::create<void>(
    const Options::Option& options) {
  const auto type_read = options.parse_as<std::string>();
  for (const auto flag : known_amr_flags()) {
    if (type_read == get_output(flag)) {
      return flag;
    }
  }
  using ::operator<<;
  PARSE_ERROR(options.context(),
              "Failed to convert \""
                  << type_read << "\" to amr::domain::Flag.\nMust be one of "
                  << known_amr_flags() << ".");
}
