// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "ParallelAlgorithms/Amr/Policies/Isotropy.hpp"

#include <ostream>
#include <vector>

#include "Options/Options.hpp"
#include "Options/ParseOptions.hpp"
#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/GetOutput.hpp"
#include "Utilities/StdHelpers.hpp"

namespace {
std::vector<amr::Isotropy> known_amr_isotropies() {
  return std::vector{amr::Isotropy::Anisotropic, amr::Isotropy::Isotropic};
}
}  // namespace

namespace amr {

std::ostream& operator<<(std::ostream& os, const Isotropy& isotropy) {
  switch (isotropy) {
    case Isotropy::Anisotropic:
      os << "Anisotropic";
      break;
    case Isotropy::Isotropic:
      os << "Isotropic";
      break;
    default:
      ERROR("An undefined isotropy was passed to the stream operator.");
  }
  return os;
}
}  // namespace amr

template <>
amr::Isotropy Options::create_from_yaml<amr::Isotropy>::create<void>(
    const Options::Option& options) {
  const auto type_read = options.parse_as<std::string>();
  for (const auto isotropy : known_amr_isotropies()) {
    if (type_read == get_output(isotropy)) {
      return isotropy;
    }
  }
  using ::operator<<;
  PARSE_ERROR(options.context(), "Failed to convert \""
                                     << type_read
                                     << "\" to amr::Isotropy.\nMust be one of "
                                     << known_amr_isotropies() << ".");
}
