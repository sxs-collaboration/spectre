// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cstddef>

#include "DataStructures/Variables.hpp"
#include "Evolution/DgSubcell/ActiveGrid.hpp"
#include "Evolution/DgSubcell/Mesh.hpp"
#include "Evolution/DgSubcell/Projection.hpp"
#include "Evolution/Systems/Burgers/Subcell/SetInitialRdmpData.hpp"
#include "Evolution/Systems/Burgers/Tags.hpp"
#include "NumericalAlgorithms/SpatialDiscretization/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/Spectral.hpp"
#include "Utilities/TMPL.hpp"

SPECTRE_TEST_CASE("Unit.Evolution.Systems.Burgers.Subcell.SetInitialRdmpData",
                  "[Unit][Evolution]") {
  using Vars = Variables<tmpl::list<Burgers::Tags::U>>;

  const Mesh<1> dg_mesh{5, SpatialDiscretization::Basis::Legendre,
                        SpatialDiscretization::Quadrature::GaussLobatto};
  const Mesh<1> subcell_mesh = evolution::dg::subcell::fd::mesh(dg_mesh);
  const size_t number_of_dg_grid_points{dg_mesh.number_of_grid_points()};

  Vars dg_vars{number_of_dg_grid_points, 1.0};

  // While the code is supposed to be used on the subcells, that doesn't
  // actually matter.
  const auto& dg_u = get<Burgers::Tags::U>(dg_vars);
  using std::max;
  using std::min;
  const auto subcell_u = evolution::dg::subcell::fd::project(
      get(dg_u), dg_mesh, subcell_mesh.extents());
  evolution::dg::subcell::RdmpTciData rdmp_data{};
  Burgers::subcell::SetInitialRdmpData::apply(
      make_not_null(&rdmp_data), get<Burgers::Tags::U>(dg_vars),
      evolution::dg::subcell::ActiveGrid::Dg, dg_mesh, subcell_mesh);
  const evolution::dg::subcell::RdmpTciData expected_dg_rdmp_data{
      {max(max(get(dg_u), max(subcell_u)))},
      {min(min(get(dg_u)), min(subcell_u))}};
  CHECK(rdmp_data == expected_dg_rdmp_data);

  Burgers::subcell::SetInitialRdmpData::apply(
      make_not_null(&rdmp_data), get<Burgers::Tags::U>(dg_vars),
      evolution::dg::subcell::ActiveGrid::Subcell, dg_mesh, subcell_mesh);
  const evolution::dg::subcell::RdmpTciData expected_subcell_rdmp_data{
      {max(get(dg_u))}, {min(get(dg_u))}};
  CHECK(rdmp_data == expected_subcell_rdmp_data);
}
