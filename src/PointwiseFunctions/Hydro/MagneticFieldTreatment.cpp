// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "PointwiseFunctions/Hydro/MagneticFieldTreatment.hpp"

#include <ostream>
#include <string>

#include "Options/Options.hpp"
#include "Options/ParseOptions.hpp"
#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/GetOutput.hpp"

namespace hydro {
std::ostream& operator<<(std::ostream& os, const MagneticFieldTreatment t) {
  switch (t) {
    case MagneticFieldTreatment::AssumeZero:
      return os << "AssumeZero";
    case MagneticFieldTreatment::CheckIfZero:
      return os << "CheckIfZero";
    case MagneticFieldTreatment::AssumeNonZero:
      return os << "AssumeNonZero";
    default:
      ERROR("Unknown value for MagneticFieldTreatment " << static_cast<int>(t));
  };
}
}  // namespace hydro

template <>
hydro::MagneticFieldTreatment
Options::create_from_yaml<hydro::MagneticFieldTreatment>::create<void>(
    const Options::Option& options) {
  const auto recons_method = options.parse_as<std::string>();
  if (recons_method == get_output(type::AssumeZero)) {
    return type::AssumeZero;
  } else if (recons_method == get_output(type::CheckIfZero)) {
    return type::CheckIfZero;
  } else if (recons_method == get_output(type::AssumeNonZero)) {
    return type::AssumeNonZero;
  } else {
    PARSE_ERROR(options.context(),
                "MagneticFieldTreatment must be '"
                    << get_output(type::AssumeZero) << "', '"
                    << get_output(type::CheckIfZero) << "', or '"
                    << get_output(type::AssumeNonZero) << "'");
  }
}
