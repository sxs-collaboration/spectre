// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/Burgers/Subcell/GhostData.hpp"

#include "DataStructures/Variables.hpp"
#include "Evolution/Systems/Burgers/Tags.hpp"
#include "Utilities/TMPL.hpp"

namespace Burgers::subcell {
Variables<tmpl::list<Burgers::Tags::U>> GhostVariables::apply(
    const Variables<tmpl::list<Burgers::Tags::U>>& vars) {
  return vars;
}
}  // namespace Burgers::subcell
