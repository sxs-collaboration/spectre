// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cstddef>
#include <limits>
#include <random>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "DataStructures/VariablesTag.hpp"
#include "Evolution/DgSubcell/Mesh.hpp"
#include "Evolution/Systems/ForceFree/FiniteDifference/Tags.hpp"
#include "Evolution/Systems/ForceFree/Subcell/GhostData.hpp"
#include "Evolution/Systems/ForceFree/System.hpp"
#include "Evolution/Systems/ForceFree/Tags.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/DataStructures/MakeWithRandomValues.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/Spectral.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

namespace ForceFree::fd {

namespace {
void test_ghost_data_on_subcells(
    const gsl::not_null<std::mt19937*> gen,
    const gsl::not_null<std::uniform_real_distribution<>*> dist) {
  const Mesh<3> dg_mesh{5, Spectral::Basis::Legendre,
                        Spectral::Quadrature::GaussLobatto};
  const Mesh<3> subcell_mesh = evolution::dg::subcell::fd::mesh(dg_mesh);

  using evolved_vars_tags = System::variables_tag::tags_list;
  const auto evolved_vars =
      make_with_random_values<Variables<evolved_vars_tags>>(
          gen, dist, subcell_mesh.number_of_grid_points());
  const auto tilde_j = make_with_random_values<ForceFree::Tags::TildeJ::type>(
      gen, dist, subcell_mesh.number_of_grid_points());

  auto box = db::create<db::AddSimpleTags<::Tags::Variables<evolved_vars_tags>,
                                          ForceFree::Tags::TildeJ>>(
      evolved_vars, tilde_j);
  DataVector recons_prims_rdmp =
      db::mutate_apply<ForceFree::subcell::GhostVariables>(make_not_null(&box),
                                                           2_st);
  const Variables<ForceFree::fd::tags_list_for_reconstruction> recons_vars{
      recons_prims_rdmp.data(), recons_prims_rdmp.size() - 2};

  tmpl::for_each<evolved_vars_tags>([&evolved_vars, &recons_vars](auto tag_v) {
    using tag = tmpl::type_from<decltype(tag_v)>;
    CHECK_ITERABLE_APPROX(get<tag>(recons_vars), get<tag>(evolved_vars));
  });
  CHECK_ITERABLE_APPROX(get<ForceFree::Tags::TildeJ>(recons_vars), tilde_j);
}

SPECTRE_TEST_CASE("Unit.Evolution.Systems.ForceFree.Subcell.GhostData",
                  "[Unit][Evolution]") {
  MAKE_GENERATOR(gen);
  std::uniform_real_distribution<> dist(-1.0, 1.0);
  test_ghost_data_on_subcells(make_not_null(&gen), make_not_null(&dist));
}
}  // namespace

}  // namespace ForceFree::fd
