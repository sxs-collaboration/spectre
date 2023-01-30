// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/Burgers/Subcell/GhostData.hpp"

#include "DataStructures/Variables.hpp"
#include "Evolution/Systems/Burgers/Tags.hpp"
#include "Utilities/TMPL.hpp"

namespace Burgers::subcell {
DataVector GhostVariables::apply(
    const Variables<tmpl::list<Burgers::Tags::U>>& vars,
    const size_t rdmp_size) {
  DataVector buffer{vars.number_of_grid_points() + rdmp_size};
  DataVector var_view{buffer.data(), vars.number_of_grid_points()};
  var_view = get(get<Burgers::Tags::U>(vars));
  return buffer;
}
}  // namespace Burgers::subcell
