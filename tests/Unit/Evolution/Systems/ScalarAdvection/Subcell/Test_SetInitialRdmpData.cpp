// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cstddef>

#include "DataStructures/Variables.hpp"
#include "Evolution/DgSubcell/ActiveGrid.hpp"
#include "Evolution/DgSubcell/Mesh.hpp"
#include "Evolution/DgSubcell/Projection.hpp"
#include "Evolution/Systems/ScalarAdvection/Subcell/SetInitialRdmpData.hpp"
#include "Evolution/Systems/ScalarAdvection/Subcell/TciOptions.hpp"
#include "Evolution/Systems/ScalarAdvection/Tags.hpp"
#include "NumericalAlgorithms/SpatialDiscretization/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/Spectral.hpp"
#include "Utilities/TMPL.hpp"

namespace {
template <size_t Dim>
void test() {
  // type aliases
  using Vars = Variables<tmpl::list<ScalarAdvection::Tags::U>>;

  // create DG mesh and subcell mesh for test
  const Mesh<Dim> dg_mesh{5, SpatialDiscretization::Basis::Legendre,
                          SpatialDiscretization::Quadrature::GaussLobatto};
  const Mesh<Dim> subcell_mesh = evolution::dg::subcell::fd::mesh(dg_mesh);
  const size_t number_of_dg_grid_points{dg_mesh.number_of_grid_points()};

  // create scalar field U for DG mesh
  Vars dg_vars{number_of_dg_grid_points, 1.0};

  // While the code is supposed to be used on the subcells, that doesn't
  // actually matter.
  const auto& dg_u = get<ScalarAdvection::Tags::U>(dg_vars);
  using std::max;
  using std::min;
  const auto subcell_u = evolution::dg::subcell::fd::project(
      get(dg_u), dg_mesh, subcell_mesh.extents());
  evolution::dg::subcell::RdmpTciData rdmp_data{};
  ScalarAdvection::subcell::SetInitialRdmpData<Dim>::apply(
      make_not_null(&rdmp_data), dg_u, evolution::dg::subcell::ActiveGrid::Dg,
      dg_mesh, subcell_mesh);
  const evolution::dg::subcell::RdmpTciData expected_dg_rdmp_data{
      {max(max(get(dg_u), max(subcell_u)))},
      {min(min(get(dg_u)), min(subcell_u))}};
  CHECK(rdmp_data == expected_dg_rdmp_data);

  ScalarAdvection::subcell::SetInitialRdmpData<Dim>::apply(
      make_not_null(&rdmp_data), dg_u,
      evolution::dg::subcell::ActiveGrid::Subcell, dg_mesh, subcell_mesh);
  const evolution::dg::subcell::RdmpTciData expected_rdmp_data{
      {max(get(dg_u))}, {min(get(dg_u))}};
  CHECK(rdmp_data == expected_rdmp_data);
}
}  // namespace

SPECTRE_TEST_CASE(
    "Unit.Evolution.Systems.ScalarAdvection.Subcell.SetInitialRdmpData",
    "[Unit][Evolution]") {
  test<1>();
  test<2>();
}
