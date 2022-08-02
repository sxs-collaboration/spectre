// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cstddef>

#include "DataStructures/Variables.hpp"
#include "Evolution/DgSubcell/ActiveGrid.hpp"
#include "Evolution/DgSubcell/Mesh.hpp"
#include "Evolution/DgSubcell/Projection.hpp"
#include "Evolution/Systems/ScalarAdvection/Subcell/InitialDataTci.hpp"
#include "Evolution/Systems/ScalarAdvection/Subcell/TciOptions.hpp"
#include "Evolution/Systems/ScalarAdvection/Tags.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/Spectral.hpp"
#include "Utilities/TMPL.hpp"

namespace {
template <size_t Dim>
void test() {
  // type aliases
  using Vars = Variables<tmpl::list<ScalarAdvection::Tags::U>>;

  // create DG mesh and subcell mesh for test
  const Mesh<Dim> dg_mesh{5, Spectral::Basis::Legendre,
                          Spectral::Quadrature::GaussLobatto};
  const Mesh<Dim> subcell_mesh = evolution::dg::subcell::fd::mesh(dg_mesh);
  const size_t number_of_dg_grid_points{dg_mesh.number_of_grid_points()};

  // create scalar field U for DG mesh
  Vars dg_vars{number_of_dg_grid_points, 1.0};

  // TCI parameters
  const double persson_exponent{4.0};
  const double rdmp_delta0{1.0e-4};
  const double rdmp_epsilon{1.0e-3};
  const ScalarAdvection::subcell::TciOptions tci_options{1.0e-8};

  const auto compute_expected_rdmp_tci_data = [&dg_vars, &dg_mesh,
                                               &subcell_mesh]() {
    const auto subcell_vars = evolution::dg::subcell::fd::project(
        dg_vars, dg_mesh, subcell_mesh.extents());
    using std::max;
    using std::min;
    evolution::dg::subcell::RdmpTciData rdmp_tci_data{
        {max(max(get(get<ScalarAdvection::Tags::U>(dg_vars))),
             max(get(get<ScalarAdvection::Tags::U>(subcell_vars))))},
        {min(min(get(get<ScalarAdvection::Tags::U>(dg_vars))),
             min(get(get<ScalarAdvection::Tags::U>(subcell_vars))))}};
    return rdmp_tci_data;
  };

  {
    INFO("TCI is happy");
    const auto result = ScalarAdvection::subcell::DgInitialDataTci<Dim>::apply(
        dg_vars, rdmp_delta0, rdmp_epsilon, persson_exponent, dg_mesh,
        subcell_mesh, tci_options);
    CHECK_FALSE(std::get<0>(result));
    CHECK(std::get<1>(result) == compute_expected_rdmp_tci_data());
  }

  {
    INFO("Two mesh RDMP fails");
    // set subcell_vars to be smooth but quite different from dg_vars
    // Test that the 2-mesh RDMP fails be setting an absurdly small epsilon
    // and delta_0 tolerance.
    get(get<ScalarAdvection::Tags::U>(
        dg_vars))[dg_mesh.number_of_grid_points() / 2] *=
        1.0 + std::numeric_limits<double>::epsilon() * 2.0;
    const auto result = ScalarAdvection::subcell::DgInitialDataTci<Dim>::apply(
        dg_vars, 1.0e-100, 1.0e-18, persson_exponent, dg_mesh, subcell_mesh,
        tci_options);
    CHECK(std::get<0>(result));
    CHECK(std::get<1>(result) == compute_expected_rdmp_tci_data());
    get(get<ScalarAdvection::Tags::U>(
        dg_vars))[dg_mesh.number_of_grid_points() / 2] /=
        1.0 + std::numeric_limits<double>::epsilon() * 2.0;
  }

  {
    INFO("Persson TCI fails");
    // set dg_vars to have a sharp peak
    get(get<ScalarAdvection::Tags::U>(dg_vars))[number_of_dg_grid_points / 2] +=
        1.0;
    // set rdmp_delta0 to be very large to ensure that it's the Persson TCI
    // which triggers alarm here
    const auto result = ScalarAdvection::subcell::DgInitialDataTci<Dim>::apply(
        dg_vars, 1.0e100, rdmp_epsilon, persson_exponent, dg_mesh, subcell_mesh,
        tci_options);
    CHECK(std::get<0>(result));
    CHECK(std::get<1>(result) == compute_expected_rdmp_tci_data());
  }

  {
    INFO("U is below absolute cutoff");
    // dg_vars is troubled but it is scaled to be smaller than the cutoff
    get(get<ScalarAdvection::Tags::U>(dg_vars)) *= 1.0e-10;
    const auto result = ScalarAdvection::subcell::DgInitialDataTci<Dim>::apply(
        dg_vars, 1.0e100, rdmp_epsilon, persson_exponent, dg_mesh, subcell_mesh,
        tci_options);
    CHECK_FALSE(std::get<0>(result));
    CHECK(std::get<1>(result) == compute_expected_rdmp_tci_data());
    get(get<ScalarAdvection::Tags::U>(dg_vars)) *= 1.0e+10;
  }

  {
    INFO("Test SetInitialRdmpData");
    // While the code is supposed to be used on the subcells, that doesn't
    // actually matter.
    evolution::dg::subcell::RdmpTciData rdmp_data{};
    ScalarAdvection::subcell::SetInitialRdmpData::apply(
        make_not_null(&rdmp_data), get<ScalarAdvection::Tags::U>(dg_vars),
        evolution::dg::subcell::ActiveGrid::Dg);
    CHECK(rdmp_data == evolution::dg::subcell::RdmpTciData{});
    ScalarAdvection::subcell::SetInitialRdmpData::apply(
        make_not_null(&rdmp_data), get<ScalarAdvection::Tags::U>(dg_vars),
        evolution::dg::subcell::ActiveGrid::Subcell);
    const evolution::dg::subcell::RdmpTciData expected_rdmp_data{
        {max(get(get<ScalarAdvection::Tags::U>(dg_vars)))},
        {min(get(get<ScalarAdvection::Tags::U>(dg_vars)))}};
    CHECK(rdmp_data == expected_rdmp_data);
  }
}
}  // namespace

SPECTRE_TEST_CASE(
    "Unit.Evolution.Systems.ScalarAdvection.Subcell.InitialDataTci",
    "[Unit][Evolution]") {
  test<1>();
  test<2>();
}
