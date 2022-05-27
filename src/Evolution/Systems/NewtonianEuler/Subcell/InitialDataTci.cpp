// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/NewtonianEuler/Subcell/InitialDataTci.hpp"

#include <cstddef>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Evolution/DgSubcell/ActiveGrid.hpp"
#include "Evolution/DgSubcell/PerssonTci.hpp"
#include "Evolution/DgSubcell/Projection.hpp"
#include "Evolution/DgSubcell/TwoMeshRdmpTci.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "Utilities/GenerateInstantiations.hpp"

namespace NewtonianEuler::subcell {
template <size_t Dim>
std::tuple<bool, evolution::dg::subcell::RdmpTciData>
DgInitialDataTci<Dim>::apply(
    const Variables<
        tmpl::list<MassDensityCons, MomentumDensity, EnergyDensity>>& dg_vars,
    const double rdmp_delta0, const double rdmp_epsilon,
    const double persson_exponent, const Mesh<Dim>& dg_mesh,
    const Mesh<Dim>& subcell_mesh) {
  const auto subcell_vars = evolution::dg::subcell::fd::project(
      dg_vars, dg_mesh, subcell_mesh.extents());

  const Scalar<DataVector>& subcell_mass_density =
      get<MassDensityCons>(subcell_vars);
  const Scalar<DataVector>& subcell_energy_density =
      get<EnergyDensity>(subcell_vars);
  const Scalar<DataVector>& dg_mass_density = get<MassDensityCons>(dg_vars);
  const Scalar<DataVector>& dg_energy_density = get<EnergyDensity>(dg_vars);

  using std::max;
  using std::min;
  evolution::dg::subcell::RdmpTciData rdmp_tci_data{
      {max(max(get(dg_mass_density)), max(get(subcell_mass_density))),
       max(max(get(dg_energy_density)), max(get(subcell_energy_density)))},
      {min(min(get(dg_mass_density)), min(get(subcell_mass_density))),
       min(min(get(dg_energy_density)), min(get(subcell_energy_density)))}};

  return {evolution::dg::subcell::two_mesh_rdmp_tci(
              dg_vars, subcell_vars, rdmp_delta0, rdmp_epsilon) or
              evolution::dg::subcell::persson_tci(dg_mass_density, dg_mesh,
                                                  persson_exponent) or
              evolution::dg::subcell::persson_tci(dg_energy_density, dg_mesh,
                                                  persson_exponent),
          std::move(rdmp_tci_data)};
}

template <size_t Dim>
void SetInitialRdmpData<Dim>::apply(
    const gsl::not_null<evolution::dg::subcell::RdmpTciData*> rdmp_tci_data,
    const Variables<tmpl::list<MassDensityCons, MomentumDensity,
                               EnergyDensity>>& subcell_vars,
    const evolution::dg::subcell::ActiveGrid active_grid) {
  if (active_grid == evolution::dg::subcell::ActiveGrid::Subcell) {
    const Scalar<DataVector>& subcell_mass_density =
        get<MassDensityCons>(subcell_vars);
    const Scalar<DataVector>& subcell_energy_density =
        get<EnergyDensity>(subcell_vars);
    *rdmp_tci_data = {
        {max(get(subcell_mass_density)), max(get(subcell_energy_density))},
        {min(get(subcell_mass_density)), min(get(subcell_energy_density))}};
  }
}

#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)
#define INSTANTIATION(r, data)                 \
  template struct DgInitialDataTci<DIM(data)>; \
  template struct SetInitialRdmpData<DIM(data)>;
GENERATE_INSTANTIATIONS(INSTANTIATION, (1, 2, 3))
#undef INSTANTIATION
#undef DIM
}  // namespace NewtonianEuler::subcell
