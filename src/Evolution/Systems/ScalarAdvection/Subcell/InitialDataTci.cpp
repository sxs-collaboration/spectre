// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/ScalarAdvection/Subcell/InitialDataTci.hpp"

#include <cstddef>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Evolution/DgSubcell/PerssonTci.hpp"
#include "Evolution/DgSubcell/Projection.hpp"
#include "Evolution/DgSubcell/TwoMeshRdmpTci.hpp"
#include "Utilities/GenerateInstantiations.hpp"

namespace ScalarAdvection::subcell {
template <size_t Dim>
std::tuple<bool, evolution::dg::subcell::RdmpTciData>
DgInitialDataTci<Dim>::apply(
    const Variables<tmpl::list<ScalarAdvection::Tags::U>>& dg_vars,
    double rdmp_delta0, double rdmp_epsilon, double persson_exponent,
    const Mesh<Dim>& dg_mesh, const Mesh<Dim>& subcell_mesh,
    const TciOptions& tci_options) {
  const auto subcell_vars = evolution::dg::subcell::fd::project(
      dg_vars, dg_mesh, subcell_mesh.extents());

  const auto& dg_u = get<ScalarAdvection::Tags::U>(dg_vars);
  const auto& subcell_u = get<ScalarAdvection::Tags::U>(subcell_vars);
  using std::max;
  using std::min;
  evolution::dg::subcell::RdmpTciData rdmp_tci_data{
      {max(max(get(dg_u)), max(get(subcell_u)))},
      {min(min(get(dg_u)), min(get(subcell_u)))}};

  const double max_abs_u =
      max(abs(get(get<ScalarAdvection::Tags::U>(dg_vars))));

  return {evolution::dg::subcell::two_mesh_rdmp_tci(
              dg_vars, subcell_vars, rdmp_delta0, rdmp_epsilon) or
              ((max_abs_u > tci_options.u_cutoff) and
               evolution::dg::subcell::persson_tci(
                   get<ScalarAdvection::Tags::U>(dg_vars), dg_mesh,
                   persson_exponent)),
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

#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATION(r, data) template struct DgInitialDataTci<DIM(data)>;

GENERATE_INSTANTIATIONS(INSTANTIATION, (1, 2))

#undef INSTANTIATION

#undef DIM
}  // namespace ScalarAdvection::subcell
