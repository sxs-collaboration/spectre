// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cstddef>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Evolution/DgSubcell/Mesh.hpp"
#include "Evolution/DgSubcell/Tags/Inactive.hpp"
#include "Evolution/Systems/NewtonianEuler/Subcell/InitialDataTci.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/Spectral.hpp"

namespace {
template <typename Tag>
using Inactive = evolution::dg::subcell::Tags::Inactive<Tag>;

template <size_t Dim>
void test() {
  using MassDensityCons = NewtonianEuler::Tags::MassDensityCons;
  using EnergyDensity = NewtonianEuler::Tags::EnergyDensity;
  using MomentumDensity = NewtonianEuler::Tags::MomentumDensity<Dim>;
  using ConsVars =
      Variables<tmpl::list<MassDensityCons, MomentumDensity, EnergyDensity>>;
  using InactiveConsVars =
      Variables<tmpl::list<Inactive<MassDensityCons>, Inactive<MomentumDensity>,
                           Inactive<EnergyDensity>>>;

  const Mesh<Dim> dg_mesh{5, Spectral::Basis::Legendre,
                          Spectral::Quadrature::GaussLobatto};
  const Mesh<Dim> subcell_mesh = evolution::dg::subcell::fd::mesh(dg_mesh);
  ConsVars dg_vars{dg_mesh.number_of_grid_points(), 1.0};
  Scalar<DataVector> dg_pressure{dg_mesh.number_of_grid_points(), 1.0};
  const double delta0 = 1.0e-4;
  const double epsilon = 1.0e-3;
  const double exponent = 4.0;

  {
    INFO("TCI is happy");
    const InactiveConsVars subcell_vars{subcell_mesh.number_of_grid_points(),
                                        1.0};
    CHECK_FALSE(NewtonianEuler::subcell::DgInitialDataTci<Dim>::apply(
        dg_vars, subcell_vars, delta0, epsilon, exponent, dg_mesh,
        dg_pressure));
  }

  {
    INFO("Two mesh RDMP fails");
    const InactiveConsVars subcell_vars{subcell_mesh.number_of_grid_points(),
                                        2.0};
    CHECK(NewtonianEuler::subcell::DgInitialDataTci<Dim>::apply(
        dg_vars, subcell_vars, delta0, epsilon, exponent, dg_mesh,
        dg_pressure));
  }

  {
    INFO("Persson TCI mass density fails");
    const InactiveConsVars subcell_vars{subcell_mesh.number_of_grid_points(),
                                        1.0};
    get(get<MassDensityCons>(dg_vars))[dg_mesh.number_of_grid_points() / 2] +=
        2.0e10;
    CHECK(NewtonianEuler::subcell::DgInitialDataTci<Dim>::apply(
        dg_vars, subcell_vars, 1.0e100, epsilon, exponent, dg_mesh,
        dg_pressure));
    get(get<MassDensityCons>(dg_vars))[dg_mesh.number_of_grid_points() / 2] =
        1.0;
  }

  {
    INFO("Persson TCI pressure fails");
    const InactiveConsVars subcell_vars{subcell_mesh.number_of_grid_points(),
                                        1.0};
    get(dg_pressure)[dg_mesh.number_of_grid_points() / 2] *= 2.0;
    CHECK(NewtonianEuler::subcell::DgInitialDataTci<Dim>::apply(
        dg_vars, subcell_vars, 1.0e100, epsilon, exponent, dg_mesh,
        dg_pressure));
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
