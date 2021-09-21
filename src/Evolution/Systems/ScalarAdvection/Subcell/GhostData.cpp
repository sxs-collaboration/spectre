// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/ScalarAdvection/Subcell/GhostData.hpp"

#include <cstddef>

#include "DataStructures/Variables.hpp"
#include "Domain/Tags.hpp"
#include "Evolution/DgSubcell/Projection.hpp"
#include "Evolution/DgSubcell/Tags/Mesh.hpp"
#include "Evolution/Systems/ScalarAdvection/Tags.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/TMPL.hpp"

namespace ScalarAdvection::subcell {
Variables<tmpl::list<ScalarAdvection::Tags::U>> GhostDataOnSubcells::apply(
    const Variables<tmpl::list<ScalarAdvection::Tags::U>>& vars) noexcept {
  return vars;
}

template <size_t Dim>
Variables<tmpl::list<ScalarAdvection::Tags::U>> GhostDataToSlice<Dim>::apply(
    const Variables<tmpl::list<ScalarAdvection::Tags::U>>& vars,
    const Mesh<Dim>& dg_mesh, const Mesh<Dim>& subcell_mesh) noexcept {
  return evolution::dg::subcell::fd::project(vars, dg_mesh,
                                             subcell_mesh.extents());
}

#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)
#define INSTANTIATION(r, data) template class GhostDataToSlice<DIM(data)>;
GENERATE_INSTANTIATIONS(INSTANTIATION, (1, 2))
#undef INSTANTIATION
#undef DIM
}  // namespace ScalarAdvection::subcell
