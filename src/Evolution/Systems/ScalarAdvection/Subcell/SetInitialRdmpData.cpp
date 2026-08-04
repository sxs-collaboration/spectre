// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/ScalarAdvection/Subcell/SetInitialRdmpData.hpp"

#include <cstddef>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Evolution/DgSubcell/Mesh.hpp"
#include "Evolution/DgSubcell/PerssonTci.hpp"
#include "Evolution/DgSubcell/Projection.hpp"
#include "Evolution/DgSubcell/TwoMeshRdmpTci.hpp"
#include "Utilities/GenerateInstantiations.hpp"

namespace ScalarAdvection::subcell {
template <size_t Dim>
void SetInitialRdmpData<Dim>::apply(
    const gsl::not_null<evolution::dg::subcell::RdmpTciData*> rdmp_tci_data,
    const Scalar<DataVector>& u,
    const evolution::dg::subcell::ActiveGrid active_grid,
    const Mesh<Dim>& dg_mesh, const Mesh<Dim>& subcell_mesh) {
  // Also skip projection for non-hypercube elements (which can never use
  // subcell) to avoid projecting onto the invalid subcell mesh
  if (active_grid == evolution::dg::subcell::ActiveGrid::Subcell or
      not evolution::dg::subcell::fd::dg_mesh_supports_subcell(dg_mesh)) {
    *rdmp_tci_data = {{max(get(u))}, {min(get(u))}};
  } else {
    using std::max;
    using std::min;
    const auto subcell_u = evolution::dg::subcell::fd::project(
        get(u), dg_mesh, subcell_mesh.extents());

    *rdmp_tci_data = {{max(max(get(u)), max(subcell_u))},
                      {min(min(get(u)), min(subcell_u))}};
  }
}

#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATION(r, data) template struct SetInitialRdmpData<DIM(data)>;

GENERATE_INSTANTIATIONS(INSTANTIATION, (1, 2))

#undef INSTANTIATION

#undef DIM
}  // namespace ScalarAdvection::subcell
