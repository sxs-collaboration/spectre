// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/Burgers/Subcell/GhostData.hpp"

#include "DataStructures/Variables.hpp"
#include "Evolution/DgSubcell/Projection.hpp"
#include "Evolution/Systems/Burgers/Tags.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "Utilities/TMPL.hpp"

namespace Burgers::subcell {
Variables<tmpl::list<Burgers::Tags::U>> GhostDataOnSubcells::apply(
    const Variables<tmpl::list<Burgers::Tags::U>>& vars) {
  return vars;
}

Variables<tmpl::list<Burgers::Tags::U>> GhostDataToSlice::apply(
    const Variables<tmpl::list<Burgers::Tags::U>>& vars, const Mesh<1>& dg_mesh,
    const Mesh<1>& subcell_mesh) {
  return evolution::dg::subcell::fd::project(vars, dg_mesh,
                                             subcell_mesh.extents());
}
}  // namespace Burgers::subcell
