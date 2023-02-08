// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cstddef>
#include <random>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/Variables.hpp"
#include "DataStructures/VariablesTag.hpp"
#include "Evolution/DgSubcell/Mesh.hpp"
#include "Evolution/DgSubcell/Projection.hpp"
#include "Evolution/Systems/ScalarAdvection/Subcell/GhostData.hpp"
#include "Evolution/Systems/ScalarAdvection/Tags.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/DataStructures/MakeWithRandomValues.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

namespace {

const size_t num_of_pts_dg_grid = 4;

template <size_t Dim>
void test(const gsl::not_null<std::mt19937*> gen,
          const gsl::not_null<std::uniform_real_distribution<>*> dist) {
  // make random U on DG and subcell mesh
  const Mesh<Dim> dg_mesh{num_of_pts_dg_grid, Spectral::Basis::Legendre,
                          Spectral::Quadrature::GaussLobatto};
  const Mesh<Dim> subcell_mesh = evolution::dg::subcell::fd::mesh(dg_mesh);
  const auto random_vars_dg =
      make_with_random_values<Variables<tmpl::list<ScalarAdvection::Tags::U>>>(
          gen, dist, dg_mesh.number_of_grid_points());
  const auto random_vars_subcell =
      make_with_random_values<Variables<tmpl::list<ScalarAdvection::Tags::U>>>(
          gen, dist, subcell_mesh.number_of_grid_points());

  // add the random U on the subcell mesh to a databox, apply
  // GhostDataOnSubcells and compare with the returned vector
  auto box_subcell = db::create<db::AddSimpleTags<
      ::Tags::Variables<tmpl::list<ScalarAdvection::Tags::U>>>>(
      random_vars_subcell);
  DataVector retrieved_vars_subcell =
      db::mutate_apply<ScalarAdvection::subcell::GhostVariables>(
          make_not_null(&box_subcell), 2_st);
  REQUIRE(retrieved_vars_subcell.size() ==
          subcell_mesh.number_of_grid_points() + 2);
  CHECK_ITERABLE_APPROX(get(get<ScalarAdvection::Tags::U>(random_vars_subcell)),
                        DataVector(retrieved_vars_subcell.data(),
                                   retrieved_vars_subcell.size() - 2));
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Evolution.Systems.ScalarAdvection.Subcell.GhostData",
                  "[Unit][Evolution]") {
  MAKE_GENERATOR(gen);
  std::uniform_real_distribution<> dist(-1.0, 1.0);

  test<1>(make_not_null(&gen), make_not_null(&dist));
  test<2>(make_not_null(&gen), make_not_null(&dist));
}
