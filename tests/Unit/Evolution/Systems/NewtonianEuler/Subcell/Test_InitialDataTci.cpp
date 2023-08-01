// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cstddef>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Evolution/DgSubcell/ActiveGrid.hpp"
#include "Evolution/DgSubcell/Mesh.hpp"
#include "Evolution/DgSubcell/Projection.hpp"
#include "Evolution/Systems/NewtonianEuler/Subcell/InitialDataTci.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/Spectral.hpp"

namespace {
template <size_t Dim>
void test() {
  using MassDensityCons = NewtonianEuler::Tags::MassDensityCons;
  using EnergyDensity = NewtonianEuler::Tags::EnergyDensity;
  using MomentumDensity = NewtonianEuler::Tags::MomentumDensity<Dim>;
  using ConsVars =
      Variables<tmpl::list<MassDensityCons, MomentumDensity, EnergyDensity>>;
  const Mesh<Dim> dg_mesh{5, Spectral::Basis::Legendre,
                          Spectral::Quadrature::GaussLobatto};
  const Mesh<Dim> subcell_mesh = evolution::dg::subcell::fd::mesh(dg_mesh);
  ConsVars dg_vars{dg_mesh.number_of_grid_points(), 1.0};

  // While the code is supposed to be used on the subcells, that doesn't
  // actually matter.
  using std::max;
  using std::min;
  const auto& dg_mass_density =
      get<NewtonianEuler::Tags::MassDensityCons>(dg_vars);
  const auto& dg_energy_density =
      get<NewtonianEuler::Tags::EnergyDensity>(dg_vars);
  const auto subcell_mass_density = evolution::dg::subcell::fd::project(
      get(dg_mass_density), dg_mesh, subcell_mesh.extents());
  const auto subcell_energy_density = evolution::dg::subcell::fd::project(
      get(dg_energy_density), dg_mesh, subcell_mesh.extents());
  evolution::dg::subcell::RdmpTciData rdmp_data{};
  NewtonianEuler::subcell::SetInitialRdmpData<Dim>::apply(
      make_not_null(&rdmp_data), dg_vars,
      evolution::dg::subcell::ActiveGrid::Dg, dg_mesh, subcell_mesh);
  const evolution::dg::subcell::RdmpTciData expected_dg_rdmp_data{
      {max(max(get(dg_mass_density)), max(subcell_mass_density)),
       max(max(get(dg_energy_density)), max(subcell_energy_density))},
      {min(min(get(dg_mass_density)), min(subcell_mass_density)),
       min(min(get(dg_energy_density)), min(subcell_energy_density))}};
  CHECK(rdmp_data == expected_dg_rdmp_data);

  NewtonianEuler::subcell::SetInitialRdmpData<Dim>::apply(
      make_not_null(&rdmp_data), dg_vars,
      evolution::dg::subcell::ActiveGrid::Subcell, dg_mesh, subcell_mesh);
  const evolution::dg::subcell::RdmpTciData expected_subcell_rdmp_data{
      {max(get(dg_mass_density)), max(get(dg_energy_density))},
      {min(get(dg_mass_density)), min(get(dg_energy_density))}};
  CHECK(rdmp_data == expected_subcell_rdmp_data);
}
}  // namespace

SPECTRE_TEST_CASE(
    "Unit.Evolution.Systems.NewtonianEuler.Subcell.InitialDataTci",
    "[Unit][Evolution]") {
  test<1>();
  test<2>();
  test<3>();
}
