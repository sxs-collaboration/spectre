// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cstddef>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Evolution/DgSubcell/Mesh.hpp"
#include "Evolution/DgSubcell/Tags/Inactive.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/Subcell/InitialDataTci.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/System.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/Spectral.hpp"

namespace {
template <typename Tag>
using Inactive = evolution::dg::subcell::Tags::Inactive<Tag>;
}  // namespace

SPECTRE_TEST_CASE(
    "Unit.Evolution.Systems.ValenciaDivClean.Subcell.InitialDataTci",
    "[Unit][Evolution]") {
  using ConsVars =
      typename grmhd::ValenciaDivClean::System::variables_tag::type;
  using InactiveConsVars = typename evolution::dg::subcell::Tags::Inactive<
      typename grmhd::ValenciaDivClean::System::variables_tag>::type;

  const Mesh<3> dg_mesh{5, Spectral::Basis::Legendre,
                        Spectral::Quadrature::GaussLobatto};
  const Mesh<3> subcell_mesh = evolution::dg::subcell::fd::mesh(dg_mesh);
  ConsVars dg_vars{dg_mesh.number_of_grid_points(), 1.0};
  const double delta0 = 1.0e-4;
  const double epsilon = 1.0e-3;
  const double exponent = 4.0;

  {
    INFO("TCI is happy");
    const InactiveConsVars subcell_vars{subcell_mesh.number_of_grid_points(),
                                        1.0};
    CHECK_FALSE(grmhd::ValenciaDivClean::subcell::DgInitialDataTci::apply(
        dg_vars, subcell_vars, delta0, epsilon, exponent, dg_mesh));
  }

  {
    INFO("Two mesh RDMP fails");
    const InactiveConsVars subcell_vars{subcell_mesh.number_of_grid_points(),
                                        2.0};
    CHECK(grmhd::ValenciaDivClean::subcell::DgInitialDataTci::apply(
        dg_vars, subcell_vars, delta0, epsilon, exponent, dg_mesh));
  }

  {
    INFO("Persson TCI TildeD fails");
    const InactiveConsVars subcell_vars{subcell_mesh.number_of_grid_points(),
                                        1.0};
    get(get<grmhd::ValenciaDivClean::Tags::TildeD>(
        dg_vars))[dg_mesh.number_of_grid_points() / 2] += 2.0e10;
    CHECK(grmhd::ValenciaDivClean::subcell::DgInitialDataTci::apply(
        dg_vars, subcell_vars, 1.0e100, epsilon, exponent, dg_mesh));
    get(get<grmhd::ValenciaDivClean::Tags::TildeD>(
        dg_vars))[dg_mesh.number_of_grid_points() / 2] = 1.0;
  }

  {
    INFO("Persson TCI pressure fails");
    const InactiveConsVars subcell_vars{subcell_mesh.number_of_grid_points(),
                                        1.0};
    get(get<grmhd::ValenciaDivClean::Tags::TildeTau>(
        dg_vars))[dg_mesh.number_of_grid_points() / 2] += 2.0e10;
    CHECK(grmhd::ValenciaDivClean::subcell::DgInitialDataTci::apply(
        dg_vars, subcell_vars, 1.0e100, epsilon, exponent, dg_mesh));
    get(get<grmhd::ValenciaDivClean::Tags::TildeTau>(
        dg_vars))[dg_mesh.number_of_grid_points() / 2] = 1.0;
  }
}
