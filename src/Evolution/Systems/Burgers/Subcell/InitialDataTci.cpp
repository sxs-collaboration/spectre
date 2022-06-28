// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/Burgers/Subcell/InitialDataTci.hpp"

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Evolution/DgSubcell/ActiveGrid.hpp"
#include "Evolution/DgSubcell/PerssonTci.hpp"
#include "Evolution/DgSubcell/Projection.hpp"
#include "Evolution/DgSubcell/TwoMeshRdmpTci.hpp"
#include "Utilities/Gsl.hpp"

namespace Burgers::subcell {
std::tuple<bool, evolution::dg::subcell::RdmpTciData> DgInitialDataTci::apply(
    const Variables<tmpl::list<Burgers::Tags::U>>& dg_vars, double rdmp_delta0,
    double rdmp_epsilon, double persson_exponent, const Mesh<1>& dg_mesh,
    const Mesh<1>& subcell_mesh) {
  const auto subcell_vars = evolution::dg::subcell::fd::project(
      dg_vars, dg_mesh, subcell_mesh.extents());

  const auto& dg_u = get<Burgers::Tags::U>(dg_vars);
  const auto& subcell_u = get<Burgers::Tags::U>(subcell_vars);
  using std::max;
  using std::min;
  evolution::dg::subcell::RdmpTciData rdmp_tci_data{
      {max(max(get(dg_u)), max(get(subcell_u)))},
      {min(min(get(dg_u)), min(get(subcell_u)))}};

  return {evolution::dg::subcell::two_mesh_rdmp_tci(
              dg_vars, subcell_vars, rdmp_delta0, rdmp_epsilon) or
              evolution::dg::subcell::persson_tci(
                  get<Burgers::Tags::U>(dg_vars), dg_mesh, persson_exponent),
          std::move(rdmp_tci_data)};
}

void SetInitialRdmpData::apply(
    const gsl::not_null<evolution::dg::subcell::RdmpTciData*> rdmp_tci_data,
    const Scalar<DataVector>& subcell_u,
    const evolution::dg::subcell::ActiveGrid active_grid) {
  if (active_grid == evolution::dg::subcell::ActiveGrid::Subcell) {
    *rdmp_tci_data = {{max(get(subcell_u))}, {min(get(subcell_u))}};
  }
}
}  // namespace Burgers::subcell
