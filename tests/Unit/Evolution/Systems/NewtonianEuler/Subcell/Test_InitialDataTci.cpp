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

  const double rdmp_delta0 = 1.0e-4;
  const double rdmp_epsilon = 1.0e-3;
  const double persson_exponent = 4.0;

  const auto compute_expected_rdmp_tci_data = [&dg_vars, &dg_mesh,
                                               &subcell_mesh]() {
    const auto subcell_vars = evolution::dg::subcell::fd::project(
        dg_vars, dg_mesh, subcell_mesh.extents());
    using std::max;
    using std::min;
    evolution::dg::subcell::RdmpTciData rdmp_tci_data{
        {max(max(get(get<MassDensityCons>(dg_vars))),
             max(get(get<MassDensityCons>(subcell_vars)))),
         max(max(get(get<EnergyDensity>(dg_vars))),
             max(get(get<EnergyDensity>(subcell_vars))))},
        {min(min(get(get<MassDensityCons>(dg_vars))),
             min(get(get<MassDensityCons>(subcell_vars)))),
         min(min(get(get<EnergyDensity>(dg_vars))),
             min(get(get<EnergyDensity>(subcell_vars))))}};
    return rdmp_tci_data;
  };

  {
    INFO("TCI is happy");
    const auto result = NewtonianEuler::subcell::DgInitialDataTci<Dim>::apply(
        dg_vars, rdmp_delta0, rdmp_epsilon, persson_exponent, dg_mesh,
        subcell_mesh);
    CHECK_FALSE(std::get<0>(result));
    CHECK(std::get<1>(result) == compute_expected_rdmp_tci_data());
  }

  {
    INFO("Two mesh RDMP fails");
    // set subcell_vars to be smooth but quite different from dg_vars
    // Test that the 2-mesh RDMP fails be setting an absurdly small epsilon
    // and delta_0 tolerance.
    get(get<MassDensityCons>(dg_vars))[dg_mesh.number_of_grid_points() / 2] *=
        1.0 + std::numeric_limits<double>::epsilon() * 2.0;
    const auto result = NewtonianEuler::subcell::DgInitialDataTci<Dim>::apply(
        dg_vars, 1.0e-100, 1.0e-18, persson_exponent, dg_mesh, subcell_mesh);
    CHECK(std::get<0>(result));
    CHECK(std::get<1>(result) == compute_expected_rdmp_tci_data());
    get(get<MassDensityCons>(dg_vars))[dg_mesh.number_of_grid_points() / 2] /=
        1.0 + std::numeric_limits<double>::epsilon() * 2.0;
  }

  {
    INFO("Persson TCI mass density fails");
    get(get<MassDensityCons>(dg_vars))[dg_mesh.number_of_grid_points() / 2] +=
        2.0e10;
    // set rdmp_delta0 to be very large to ensure that it's the Persson TCI
    // which triggers alarm here
    const auto result = NewtonianEuler::subcell::DgInitialDataTci<Dim>::apply(
        dg_vars, 1.0e100, rdmp_epsilon, persson_exponent, dg_mesh,
        subcell_mesh);
    CHECK(std::get<0>(result));
    CHECK(std::get<1>(result) == compute_expected_rdmp_tci_data());
  }

  {
    INFO("Persson TCI energy density fails");
    get(get<EnergyDensity>(dg_vars))[dg_mesh.number_of_grid_points() / 2] +=
        2.0e10;
    // set rdmp_delta0 to be very large to ensure that it's the Persson TCI
    // which triggers alarm here
    const auto result = NewtonianEuler::subcell::DgInitialDataTci<Dim>::apply(
        dg_vars, 1.0e100, rdmp_epsilon, persson_exponent, dg_mesh,
        subcell_mesh);
    CHECK(std::get<0>(result));
    CHECK(std::get<1>(result) == compute_expected_rdmp_tci_data());
  }

  {
    INFO("Test SetInitialRdmpData");
    // While the code is supposed to be used on the subcells, that doesn't
    // actually matter.
    evolution::dg::subcell::RdmpTciData rdmp_data{};
    NewtonianEuler::subcell::SetInitialRdmpData<Dim>::apply(
        make_not_null(&rdmp_data), dg_vars,
        evolution::dg::subcell::ActiveGrid::Dg);
    CHECK(rdmp_data == evolution::dg::subcell::RdmpTciData{});
    NewtonianEuler::subcell::SetInitialRdmpData<Dim>::apply(
        make_not_null(&rdmp_data), dg_vars,
        evolution::dg::subcell::ActiveGrid::Subcell);
    const auto& dg_mass_density =
        get<NewtonianEuler::Tags::MassDensityCons>(dg_vars);
    const auto& dg_energy_density =
        get<NewtonianEuler::Tags::EnergyDensity>(dg_vars);
    const evolution::dg::subcell::RdmpTciData expected_rdmp_data{
        {max(get(dg_mass_density)), max(get(dg_energy_density))},
        {min(get(dg_mass_density)), min(get(dg_energy_density))}};

    CHECK(rdmp_data == expected_rdmp_data);
  }
}
}  // namespace

SPECTRE_TEST_CASE(
    "Unit.Evolution.Systems.NewtonianEuler.Subcell.InitialDataTci",
    "[Unit][Evolution]") {
  test<1>();
  test<2>();
  test<3>();
}
