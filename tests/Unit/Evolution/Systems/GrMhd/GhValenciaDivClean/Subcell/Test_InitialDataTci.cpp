// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cstddef>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/EagerMath/Magnitude.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Evolution/DgSubcell/Mesh.hpp"
#include "Evolution/DgSubcell/Projection.hpp"
#include "Evolution/Systems/GrMhd/GhValenciaDivClean/Subcell/InitialDataTci.hpp"
#include "Evolution/Systems/GrMhd/GhValenciaDivClean/System.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/Tags.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/Spectral.hpp"

SPECTRE_TEST_CASE(
    "Unit.Evolution.Systems.GhValenciaDivClean.Subcell.InitialDataTci",
    "[Unit][Evolution]") {
  using ConsVars =
      typename grmhd::GhValenciaDivClean::System::variables_tag::type;

  const Mesh<3> dg_mesh{5, Spectral::Basis::Legendre,
                        Spectral::Quadrature::GaussLobatto};
  const Mesh<3> subcell_mesh = evolution::dg::subcell::fd::mesh(dg_mesh);
  ConsVars dg_vars{dg_mesh.number_of_grid_points(), 1.0};
  const double delta0 = 1.0e-4;
  const double epsilon = 1.0e-3;
  const double exponent = 4.0;
  const grmhd::ValenciaDivClean::subcell::TciOptions tci_options{
      1.0e-20, 0.001, 1.0e-40, 1.1e-12, 1.0e-12, std::optional<double>{1.0e-2}};

  const auto compute_expected_rdmp_tci_data = [&dg_vars, &dg_mesh,
                                               &subcell_mesh]() {
    evolution::dg::subcell::RdmpTciData rdmp_tci_data{};
    using std::max;
    using std::min;
    const auto& dg_tilde_d =
        get<grmhd::ValenciaDivClean::Tags::TildeD>(dg_vars);
    const auto& dg_tilde_ye =
        get<grmhd::ValenciaDivClean::Tags::TildeYe>(dg_vars);
    const auto& dg_tilde_tau =
        get<grmhd::ValenciaDivClean::Tags::TildeTau>(dg_vars);
    const auto dg_tilde_b_magnitude =
        magnitude(get<grmhd::ValenciaDivClean::Tags::TildeB<>>(dg_vars));
    const auto subcell_vars = evolution::dg::subcell::fd::project(
        dg_vars, dg_mesh, subcell_mesh.extents());
    const auto& subcell_tilde_d =
        get<grmhd::ValenciaDivClean::Tags::TildeD>(subcell_vars);
    const auto& subcell_tilde_ye =
        get<grmhd::ValenciaDivClean::Tags::TildeYe>(subcell_vars);
    const auto& subcell_tilde_tau =
        get<grmhd::ValenciaDivClean::Tags::TildeTau>(subcell_vars);
    const auto subcell_tilde_b_magnitude =
        magnitude(get<grmhd::ValenciaDivClean::Tags::TildeB<>>(subcell_vars));
    rdmp_tci_data.max_variables_values =
        DataVector{max(max(get(dg_tilde_d)), max(get(subcell_tilde_d))),
                   max(max(get(dg_tilde_ye)), max(get(subcell_tilde_ye))),
                   max(max(get(dg_tilde_tau)), max(get(subcell_tilde_tau))),
                   max(max(get(dg_tilde_b_magnitude)),
                       max(get(subcell_tilde_b_magnitude)))};
    rdmp_tci_data.min_variables_values =
        DataVector{min(min(get(dg_tilde_d)), min(get(subcell_tilde_d))),
                   min(min(get(dg_tilde_ye)), min(get(subcell_tilde_ye))),
                   min(min(get(dg_tilde_tau)), min(get(subcell_tilde_tau))),
                   min(min(get(dg_tilde_b_magnitude)),
                       min(get(subcell_tilde_b_magnitude)))};
    return rdmp_tci_data;
  };

  {
    INFO("TCI is happy");
    const auto result =
        grmhd::GhValenciaDivClean::subcell::DgInitialDataTci::apply(
            dg_vars, delta0, epsilon, exponent, dg_mesh, subcell_mesh,
            tci_options);
    CHECK_FALSE(std::get<0>(result));

    CHECK(std::get<1>(result) == compute_expected_rdmp_tci_data());
  }

  {
    INFO("Two mesh RDMP fails");
    // Test that the 2-mesh RDMP fails be setting an absurdly small epsilon
    // and delta_0 tolerance.
    get(get<grmhd::ValenciaDivClean::Tags::TildeD>(
        dg_vars))[dg_mesh.number_of_grid_points() / 2] *=
        1.0 + std::numeric_limits<double>::epsilon() * 2.0;
    const auto result =
        grmhd::GhValenciaDivClean::subcell::DgInitialDataTci::apply(
            dg_vars, 1.0e-100, 1.0e-18, exponent, dg_mesh, subcell_mesh,
            tci_options);
    CHECK(std::get<0>(result));
    CHECK(std::get<1>(result) == compute_expected_rdmp_tci_data());
    get(get<grmhd::ValenciaDivClean::Tags::TildeD>(
        dg_vars))[dg_mesh.number_of_grid_points() / 2] /=
        1.0 + std::numeric_limits<double>::epsilon() * 2.0;

    // Verify TCI passes after restoring value
    CHECK_FALSE(
        std::get<0>(grmhd::GhValenciaDivClean::subcell::DgInitialDataTci::apply(
            dg_vars, delta0, epsilon, exponent, dg_mesh, subcell_mesh,
            tci_options)));
  }

  {
    INFO("Persson TCI TildeD fails");
    get(get<grmhd::ValenciaDivClean::Tags::TildeD>(
        dg_vars))[dg_mesh.number_of_grid_points() / 2] += 2.0e10;
    const auto result =
        grmhd::GhValenciaDivClean::subcell::DgInitialDataTci::apply(
            dg_vars, 1.0e100, epsilon, exponent, dg_mesh, subcell_mesh,
            tci_options);
    CHECK(std::get<0>(result));
    CHECK(std::get<1>(result) == compute_expected_rdmp_tci_data());
    get(get<grmhd::ValenciaDivClean::Tags::TildeD>(
        dg_vars))[dg_mesh.number_of_grid_points() / 2] = 1.0;
  }

  {
    INFO("Persson TCI TildeB fails");
    for (size_t i = 0; i < 3; ++i) {
      get<grmhd::ValenciaDivClean::Tags::TildeB<>>(dg_vars).get(
          i)[dg_mesh.number_of_grid_points() / 2] += 2.0e10;
    }
    const auto result =
        grmhd::GhValenciaDivClean::subcell::DgInitialDataTci::apply(
            dg_vars, 1.0e100, epsilon, exponent, dg_mesh, subcell_mesh,
            tci_options);
    CHECK(std::get<0>(result));
    CHECK(std::get<1>(result) == compute_expected_rdmp_tci_data());
    for (size_t i = 0; i < 3; ++i) {
      get<grmhd::ValenciaDivClean::Tags::TildeB<>>(dg_vars).get(
          i)[dg_mesh.number_of_grid_points() / 2] = 1.0;
    }
  }

  {
    INFO("Persson TCI TildeTau fails");
    get(get<grmhd::ValenciaDivClean::Tags::TildeTau>(
        dg_vars))[dg_mesh.number_of_grid_points() / 2] += 2.0e10;
    const auto result =
        grmhd::GhValenciaDivClean::subcell::DgInitialDataTci::apply(
            dg_vars, 1.0e100, epsilon, exponent, dg_mesh, subcell_mesh,
            tci_options);
    CHECK(std::get<0>(result));
    CHECK(std::get<1>(result) == compute_expected_rdmp_tci_data());
    get(get<grmhd::ValenciaDivClean::Tags::TildeTau>(
        dg_vars))[dg_mesh.number_of_grid_points() / 2] = 1.0;
  }

  {
    INFO("Negative TildeD");
    get(get<grmhd::ValenciaDivClean::Tags::TildeD>(
        dg_vars))[dg_mesh.number_of_grid_points() / 2] = -1.0e-20;
    auto result = grmhd::GhValenciaDivClean::subcell::DgInitialDataTci::apply(
        dg_vars, 1.0e100, epsilon, 1.0, dg_mesh, subcell_mesh, tci_options);
    CHECK(std::get<0>(result));
    CHECK(std::get<1>(result) == compute_expected_rdmp_tci_data());
    get(get<grmhd::ValenciaDivClean::Tags::TildeD>(
        dg_vars))[dg_mesh.number_of_grid_points() / 2] = 1.0;

    // Verify that the restored state is admissible by the TCI.
    result = grmhd::GhValenciaDivClean::subcell::DgInitialDataTci::apply(
        dg_vars, 1.0e100, epsilon, exponent, dg_mesh, subcell_mesh,
        tci_options);
    CHECK_FALSE(std::get<0>(result));
    CHECK(std::get<1>(result) == compute_expected_rdmp_tci_data());
  }

  {
    INFO("Negative TildeTau");
    get(get<grmhd::ValenciaDivClean::Tags::TildeTau>(
        dg_vars))[dg_mesh.number_of_grid_points() / 2] = -1.0e-20;
    auto result = grmhd::GhValenciaDivClean::subcell::DgInitialDataTci::apply(
        dg_vars, 1.0e100, epsilon, exponent, dg_mesh, subcell_mesh,
        tci_options);
    CHECK(std::get<0>(result));
    CHECK(std::get<1>(result) == compute_expected_rdmp_tci_data());
    get(get<grmhd::ValenciaDivClean::Tags::TildeTau>(
        dg_vars))[dg_mesh.number_of_grid_points() / 2] = 1.0;

    // Verify that the restored state is admissible by the TCI.
    result = grmhd::GhValenciaDivClean::subcell::DgInitialDataTci::apply(
        dg_vars, 1.0e100, epsilon, epsilon, dg_mesh, subcell_mesh, tci_options);
    CHECK_FALSE(std::get<0>(result));
    CHECK(std::get<1>(result) == compute_expected_rdmp_tci_data());
  }
}
