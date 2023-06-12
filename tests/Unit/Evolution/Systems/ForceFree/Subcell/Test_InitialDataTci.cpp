// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <algorithm>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/EagerMath/Magnitude.hpp"
#include "Evolution/DgSubcell/Mesh.hpp"
#include "Evolution/DgSubcell/Projection.hpp"
#include "Evolution/DgSubcell/RdmpTciData.hpp"
#include "Evolution/Systems/ForceFree/Subcell/InitialDataTci.hpp"
#include "Evolution/Systems/ForceFree/System.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/Spectral.hpp"

namespace ForceFree::subcell {
namespace {

SPECTRE_TEST_CASE("Unit.Evolution.Systems.ForceFree.Subcell.InitialDataTci",
                  "[Unit][Evolution]") {
  const Mesh<3> dg_mesh{5, Spectral::Basis::Legendre,
                        Spectral::Quadrature::GaussLobatto};
  const Mesh<3> subcell_mesh = evolution::dg::subcell::fd::mesh(dg_mesh);

  using EvolvedVars = typename System::variables_tag::type;
  EvolvedVars dg_vars{dg_mesh.number_of_grid_points(), 1.0};

  const double delta0 = 1.0e-4;
  const double epsilon = 1.0e-3;
  const double exponent = 4.0;

  const auto compute_expected_rdmp_tci_data = [&dg_vars, &dg_mesh,
                                               &subcell_mesh]() {
    const auto dg_tilde_e_magnitude = magnitude(get<Tags::TildeE>(dg_vars));
    const auto dg_tilde_b_magnitude = magnitude(get<Tags::TildeB>(dg_vars));
    const auto& dg_tilde_q = get<Tags::TildeQ>(dg_vars);

    const auto subcell_vars = evolution::dg::subcell::fd::project(
        dg_vars, dg_mesh, subcell_mesh.extents());
    const auto subcell_tilde_e_magnitude =
        magnitude(get<Tags::TildeE>(subcell_vars));
    const auto subcell_tilde_b_magnitude =
        magnitude(get<Tags::TildeB>(subcell_vars));
    const auto& subcell_tilde_q = get<Tags::TildeQ>(subcell_vars);

    using std::max;
    using std::min;
    evolution::dg::subcell::RdmpTciData rdmp_tci_data{};
    rdmp_tci_data.max_variables_values =
        DataVector{max(max(get(dg_tilde_e_magnitude)),
                       max(get(subcell_tilde_e_magnitude))),
                   max(max(get(dg_tilde_b_magnitude)),
                       max(get(subcell_tilde_b_magnitude))),
                   max(max(get(dg_tilde_q)), max(get(subcell_tilde_q)))};
    rdmp_tci_data.min_variables_values =
        DataVector{min(min(get(dg_tilde_e_magnitude)),
                       min(get(subcell_tilde_e_magnitude))),
                   min(min(get(dg_tilde_b_magnitude)),
                       min(get(subcell_tilde_b_magnitude))),
                   min(min(get(dg_tilde_q)), min(get(subcell_tilde_q)))};
    return rdmp_tci_data;
  };

  {
    INFO("TCI is happy");
    const auto result = DgInitialDataTci::apply(
        dg_vars, delta0, epsilon, exponent, dg_mesh, subcell_mesh);
    CHECK(std::get<0>(result) == 0);
    CHECK(std::get<1>(result) == compute_expected_rdmp_tci_data());
  }

  {
    INFO("Persson TCI fails for mag(TildeE)");
    get<Tags::TildeE>(dg_vars).get(0)[dg_mesh.number_of_grid_points() / 2] +=
        2.0;
    const auto result = subcell::DgInitialDataTci::apply(
        dg_vars, 1.0e100, epsilon, exponent, dg_mesh, subcell_mesh);
    CHECK(std::get<0>(result) == -1);
    CHECK(std::get<1>(result) == compute_expected_rdmp_tci_data());
    get<Tags::TildeE>(dg_vars).get(0)[dg_mesh.number_of_grid_points() / 2] =
        1.0;
  }

  {
    INFO("Persson TCI fails for mag(TildeB)");
    get<Tags::TildeB>(dg_vars).get(0)[dg_mesh.number_of_grid_points() / 2] +=
        2.0;
    const auto result = subcell::DgInitialDataTci::apply(
        dg_vars, 1.0e100, epsilon, exponent, dg_mesh, subcell_mesh);
    CHECK(std::get<0>(result) == -2);
    CHECK(std::get<1>(result) == compute_expected_rdmp_tci_data());
    get<Tags::TildeB>(dg_vars).get(0)[dg_mesh.number_of_grid_points() / 2] =
        1.0;
  }

  {
    INFO("Persson TCI fails for TildeQ");
    get(get<Tags::TildeQ>(dg_vars))[dg_mesh.number_of_grid_points() / 2] += 2.0;
    const auto result = subcell::DgInitialDataTci::apply(
        dg_vars, 1.0e100, epsilon, exponent, dg_mesh, subcell_mesh);
    CHECK(std::get<0>(result) == -3);
    CHECK(std::get<1>(result) == compute_expected_rdmp_tci_data());
    get(get<Tags::TildeQ>(dg_vars))[dg_mesh.number_of_grid_points() / 2] = 1.0;
  }

  {
    INFO("Two-mesh RDMP fails");
    // Test that the 2-mesh RDMP fails by setting an absurdly small epsilon
    // and delta_0 tolerance where Persson TCI is not triggered
    get(get<Tags::TildeQ>(dg_vars))[dg_mesh.number_of_grid_points() / 2] *= 2.0;
    const auto result = DgInitialDataTci::apply(dg_vars, 1.0e-100, 1.0e-100,
                                                0.0, dg_mesh, subcell_mesh);
    CHECK(std::get<0>(result) == -4);
    CHECK(std::get<1>(result) == compute_expected_rdmp_tci_data());
  }
}

}  // namespace
}  // namespace ForceFree::subcell
