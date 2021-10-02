// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/GrMhd/ValenciaDivClean/Subcell/SwapGrTags.hpp"

#include <algorithm>
#include <cstddef>

#include "DataStructures/Variables.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "Utilities/Gsl.hpp"

namespace grmhd::ValenciaDivClean::subcell {
void SwapGrTags::apply(
    const gsl::not_null<
        Variables<typename System::spacetime_variables_tag::tags_list>*>
        active_gr_vars,
    const gsl::not_null<typename evolution::dg::subcell::Tags::Inactive<
        typename System::spacetime_variables_tag>::type*>
        inactive_gr_vars,
    const Mesh<3>& dg_mesh, const Mesh<3>& subcell_mesh,
    const evolution::dg::subcell::ActiveGrid active_grid) {
  if (active_grid == evolution::dg::subcell::ActiveGrid::Dg) {
    // We might request a switch to the DG grid even if we are already on the DG
    // grid, and in this case we do nothing. This can occur when applying
    // SwapGrTags to a collection of elements that may have different TCI
    // results.
    if (active_gr_vars->number_of_grid_points() !=
        dg_mesh.number_of_grid_points()) {
      ASSERT(
          active_gr_vars->number_of_grid_points() ==
              subcell_mesh.number_of_grid_points(),
          "When swapping the GR variables from subcell to DG, the active "
          "GR variables should be holding the subcell variables and be of size "
              << subcell_mesh.number_of_grid_points()
              << " but they are of size "
              << active_gr_vars->number_of_grid_points());
      using std::swap;
      swap(*active_gr_vars, *inactive_gr_vars);
    }
  } else {
    if (active_gr_vars->number_of_grid_points() !=
        subcell_mesh.number_of_grid_points()) {
      ASSERT(active_gr_vars->number_of_grid_points() ==
                 dg_mesh.number_of_grid_points(),
             "When swapping the GR variables from DG to subcell, the active "
             "GR variables should be holding the DG variables and be of size "
                 << dg_mesh.number_of_grid_points() << " but they are of size "
                 << active_gr_vars->number_of_grid_points());
      using std::swap;
      swap(*active_gr_vars, *inactive_gr_vars);
    }
  }
}
}  // namespace grmhd::ValenciaDivClean::subcell
