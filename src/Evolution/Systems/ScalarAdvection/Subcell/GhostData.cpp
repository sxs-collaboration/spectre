// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/ScalarAdvection/Subcell/GhostData.hpp"

#include <cstddef>

#include "DataStructures/Variables.hpp"
#include "Evolution/Systems/ScalarAdvection/Tags.hpp"
#include "Utilities/TMPL.hpp"

namespace ScalarAdvection::subcell {
Variables<tmpl::list<ScalarAdvection::Tags::U>> GhostVariables::apply(
    const Variables<tmpl::list<ScalarAdvection::Tags::U>>& vars) {
  return vars;
}
}  // namespace ScalarAdvection::subcell
