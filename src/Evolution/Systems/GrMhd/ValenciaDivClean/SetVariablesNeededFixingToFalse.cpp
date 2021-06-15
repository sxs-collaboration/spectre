// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/GrMhd/ValenciaDivClean/SetVariablesNeededFixingToFalse.hpp"

#include "Utilities/Gsl.hpp"

namespace grmhd::ValenciaDivClean {
void SetVariablesNeededFixingToFalse::apply(
    const gsl::not_null<bool*> variables_needed_fixing) noexcept {
  *variables_needed_fixing = false;
}
}  // namespace grmhd::ValenciaDivClean
