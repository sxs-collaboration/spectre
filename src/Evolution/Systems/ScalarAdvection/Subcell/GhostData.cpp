// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/ScalarAdvection/Subcell/GhostData.hpp"

#include <cstddef>

#include "DataStructures/Variables.hpp"
#include "Evolution/Systems/ScalarAdvection/Tags.hpp"
#include "Utilities/TMPL.hpp"

namespace ScalarAdvection::subcell {
DataVector GhostVariables::apply(
    const Variables<tmpl::list<ScalarAdvection::Tags::U>>& vars,
    const size_t rdmp_size) {
  DataVector buffer{vars.number_of_grid_points() + rdmp_size};
  DataVector var_view{buffer.data(), vars.number_of_grid_points()};
  var_view = get(get<ScalarAdvection::Tags::U>(vars));
  return buffer;
}
}  // namespace ScalarAdvection::subcell
