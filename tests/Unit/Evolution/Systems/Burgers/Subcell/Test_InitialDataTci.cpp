// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cstddef>

#include "DataStructures/Variables.hpp"
#include "Evolution/DgSubcell/ActiveGrid.hpp"
#include "Evolution/DgSubcell/Mesh.hpp"
#include "Evolution/DgSubcell/Projection.hpp"
#include "Evolution/Systems/Burgers/Subcell/InitialDataTci.hpp"
#include "Evolution/Systems/Burgers/Tags.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/Spectral.hpp"
#include "Utilities/TMPL.hpp"

SPECTRE_TEST_CASE("Unit.Evolution.Systems.Burgers.Subcell.InitialDataTci",
                  "[Unit][Evolution]") {
  using Vars = Variables<tmpl::list<Burgers::Tags::U>>;

  const Mesh<1> dg_mesh{5, Spectral::Basis::Legendre,
                        Spectral::Quadrature::GaussLobatto};
  const Mesh<1> subcell_mesh = evolution::dg::subcell::fd::mesh(dg_mesh);
  const size_t number_of_dg_grid_points{dg_mesh.number_of_grid_points()};

  Vars dg_vars{number_of_dg_grid_points, 1.0};

  // TCI parameters
  const double persson_exponent{4.0};
  const double rdmp_delta0{1.0e-4};
  const double rdmp_epsilon{1.0e-3};

  const auto compute_expected_rdmp_tci_data = [&dg_vars, &dg_mesh,
                                               &subcell_mesh]() {
    const auto subcell_vars = evolution::dg::subcell::fd::project(
        dg_vars, dg_mesh, subcell_mesh.extents());
    using std::max;
    using std::min;
    evolution::dg::subcell::RdmpTciData rdmp_tci_data{
        {max(max(get(get<Burgers::Tags::U>(dg_vars))),
             max(get(get<Burgers::Tags::U>(subcell_vars))))},
        {min(min(get(get<Burgers::Tags::U>(dg_vars))),
             min(get(get<Burgers::Tags::U>(subcell_vars))))}};
    return rdmp_tci_data;
  };

  {
    INFO("TCI is happy");
    const auto result = Burgers::subcell::DgInitialDataTci::apply(
        dg_vars, rdmp_delta0, rdmp_epsilon, persson_exponent, dg_mesh,
        subcell_mesh);
    CHECK_FALSE(std::get<0>(result));
    CHECK(std::get<1>(result) == compute_expected_rdmp_tci_data());
  }

  {
    INFO("Two mesh RDMP fails");
    // Test that the 2-mesh RDMP fails be setting an absurdly small epsilon
    // and delta_0 tolerance.
    get(get<Burgers::Tags::U>(dg_vars))[dg_mesh.number_of_grid_points() / 2] *=
        1.0 + std::numeric_limits<double>::epsilon() * 2.0;
    const auto result = Burgers::subcell::DgInitialDataTci::apply(
        dg_vars, 1.0e-100, 1.0e-18, persson_exponent, dg_mesh, subcell_mesh);
    CHECK(std::get<0>(result));
    CHECK(std::get<1>(result) == compute_expected_rdmp_tci_data());
    get(get<Burgers::Tags::U>(dg_vars))[dg_mesh.number_of_grid_points() / 2] /=
        1.0 + std::numeric_limits<double>::epsilon() * 2.0;
  }

  {
    INFO("Persson TCI fails");
    // set dg_vars to have a sharp peak
    get(get<Burgers::Tags::U>(dg_vars))[number_of_dg_grid_points / 2] += 1.0;
    // set rdmp_delta0 to be very large to ensure that it's the Persson TCI
    // which triggers alarm here
    const auto result = Burgers::subcell::DgInitialDataTci::apply(
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
    Burgers::subcell::SetInitialRdmpData::apply(
        make_not_null(&rdmp_data), get<Burgers::Tags::U>(dg_vars),
        evolution::dg::subcell::ActiveGrid::Dg);
    CHECK(rdmp_data == evolution::dg::subcell::RdmpTciData{});
    Burgers::subcell::SetInitialRdmpData::apply(
        make_not_null(&rdmp_data), get<Burgers::Tags::U>(dg_vars),
        evolution::dg::subcell::ActiveGrid::Subcell);
    const evolution::dg::subcell::RdmpTciData expected_rdmp_data{
        {max(get(get<Burgers::Tags::U>(dg_vars)))},
        {min(get(get<Burgers::Tags::U>(dg_vars)))}};
    CHECK(rdmp_data == expected_rdmp_data);
  }
}
