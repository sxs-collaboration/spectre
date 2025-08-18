// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "ParallelAlgorithms/Amr/Criteria/Type.hpp"

#include <ostream>
#include <vector>

#include "Options/Options.hpp"
#include "Options/ParseOptions.hpp"
#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/GetOutput.hpp"
#include "Utilities/StdHelpers.hpp"

namespace {
std::vector<amr::Criteria::Type> known_amr_criteria_types() {
  return std::vector{amr::Criteria::Type::h, amr::Criteria::Type::p};
}
}  // namespace

namespace amr::Criteria {

std::ostream& operator<<(std::ostream& os, const Type& type) {
  switch (type) {
    case Type::h:
      os << "h";
      break;
    case Type::p:
      os << "p";
      break;
    default:  // LCOV_EXCL_LINE
      // LCOV_EXCL_START
      ERROR("An unknown AMR criteria type was passed to the stream operator.");
      // LCOV_EXCL_STOP
  }
  return os;
}
}  // namespace amr::Criteria

template <>
amr::Criteria::Type
Options::create_from_yaml<amr::Criteria::Type>::create<void>(
    const Options::Option& options) {
  const auto type_read = options.parse_as<std::string>();
  for (const auto type : known_amr_criteria_types()) {
    if (type_read == get_output(type)) {
      return type;
    }
  }
  using ::operator<<;
  PARSE_ERROR(options.context(),
              "Failed to convert \""
                  << type_read << "\" to amr::Criteria::Type.\nMust be one of "
                  << known_amr_criteria_types() << ".");
}
