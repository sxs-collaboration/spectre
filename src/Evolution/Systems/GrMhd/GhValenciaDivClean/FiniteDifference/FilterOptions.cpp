// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/GrMhd/GhValenciaDivClean/FiniteDifference/FilterOptions.hpp"

#include <pup.h>

#include "Options/ParseError.hpp"
#include "Utilities/Serialization/PupStlCpp17.hpp"

namespace grmhd::GhValenciaDivClean::fd {
FilterOptions::FilterOptions(
    const std::optional<double> in_spacetime_dissipation,
    const Options::Context& context)
    : spacetime_dissipation(in_spacetime_dissipation) {
  if (spacetime_dissipation.has_value() and
      (spacetime_dissipation.value() <= 0.0 or
       spacetime_dissipation.value() >= 1.0)) {
    PARSE_ERROR(context,
                "Spacetime dissipation must be between 0 and 1, but got "
                    << spacetime_dissipation.value());
  }
}

void FilterOptions::pup(PUP::er& p) { p | spacetime_dissipation; }
}  // namespace grmhd::GhValenciaDivClean::fd
