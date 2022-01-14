// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/DgSubcell/Tags/ObserverMesh.hpp"

#include <cstddef>

#include "Evolution/DgSubcell/ActiveGrid.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

namespace evolution::dg::subcell::Tags {
template <size_t Dim>
void ObserverMeshCompute<Dim>::function(
    const gsl::not_null<return_type*> active_mesh, const ::Mesh<Dim>& dg_mesh,
    const ::Mesh<Dim>& subcell_mesh, const subcell::ActiveGrid active_grid) {
  if (active_grid == subcell::ActiveGrid::Dg) {
    *active_mesh = dg_mesh;
  } else {
    ASSERT(active_grid == subcell::ActiveGrid::Subcell,
           "The active grid must be either DG or subcell");
    *active_mesh = subcell_mesh;
  }
}

#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATION(r, data) template class ObserverMeshCompute<DIM(data)>;

GENERATE_INSTANTIATIONS(INSTANTIATION, (1, 2, 3))

#undef GET_DIM
#undef INSTANTIATION

}  // namespace evolution::dg::subcell::Tags
