// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/Burgers/Subcell/SetInitialRdmpData.hpp"

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Evolution/DgSubcell/ActiveGrid.hpp"
#include "Evolution/DgSubcell/PerssonTci.hpp"
#include "Evolution/DgSubcell/Projection.hpp"
#include "Evolution/DgSubcell/TwoMeshRdmpTci.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "Utilities/Gsl.hpp"

namespace Burgers::subcell {
void SetInitialRdmpData::apply(
    const gsl::not_null<evolution::dg::subcell::RdmpTciData*> rdmp_tci_data,
    const Scalar<DataVector>& u,
    const evolution::dg::subcell::ActiveGrid active_grid,
    const Mesh<1>& dg_mesh, const Mesh<1>& subcell_mesh) {
  if (active_grid == evolution::dg::subcell::ActiveGrid::Subcell) {
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
}  // namespace Burgers::subcell
