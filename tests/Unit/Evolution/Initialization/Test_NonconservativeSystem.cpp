// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cstddef>

#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Domain/Tags.hpp"
#include "Evolution/Initialization/NonconservativeSystem.hpp"
#include "Framework/ActionTesting.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/Spectral.hpp"
#include "Parallel/Actions/SetupDataBox.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

namespace {
struct Var : db::SimpleTag {
  using type = Scalar<DataVector>;
};

template <size_t Dim>
struct System {
  static constexpr bool is_in_flux_conservative_form = false;
  static constexpr bool has_primitive_and_conservative_vars = false;
  static constexpr size_t volume_dim = Dim;
  using variables_tag = Tags::Variables<tmpl::list<Var>>;
};

template <size_t Dim, typename Metavariables>
struct component {
  using metavariables = Metavariables;
  using chare_type = ActionTesting::MockArrayChare;
  using array_index = size_t;
  using const_global_cache_tag_list = tmpl::list<>;

  using initial_tags = tmpl::list<domain::Tags::Mesh<Dim>>;

  using phase_dependent_action_list = tmpl::list<Parallel::PhaseActions<
      typename Metavariables::Phase, Metavariables::Phase::Initialization,
      tmpl::list<ActionTesting::InitializeDataBox<initial_tags>,
                 Actions::SetupDataBox,
                 Initialization::Actions::NonconservativeSystem<
                     typename Metavariables::system>>>>;
};

template <size_t Dim>
struct Metavariables {
  using component_list = tmpl::list<component<Dim, Metavariables>>;
  using system = System<Dim>;
  using const_global_cache_tag_list = tmpl::list<>;
  enum class Phase { Initialization, Exit };
};

template <size_t Dim>
void test() noexcept {
  using metavars = Metavariables<Dim>;
  using comp = component<Dim, metavars>;
  using MockRuntimeSystem = ActionTesting::MockRuntimeSystem<metavars>;
  MockRuntimeSystem runner{{}};
  Mesh<Dim> mesh{5, Spectral::Basis::Legendre,
                 Spectral::Quadrature::GaussLobatto};
  ActionTesting::emplace_component_and_initialize<comp>(&runner, 0, {mesh});
  // Invoke the NonconservativeSystem action on the runner
  for (size_t i = 0; i < 2; ++i) {
    ActionTesting::next_action<comp>(make_not_null(&runner), 0);
  }
  using vars_tag = Tags::Variables<tmpl::list<Var>>;
  // The numerical value that the vars are set to is undefined, but the number
  // of grid points must be correct.
  CHECK(ActionTesting::get_databox_tag<comp, vars_tag>(runner, 0)
            .number_of_grid_points() == mesh.number_of_grid_points());
}

SPECTRE_TEST_CASE("Unit.Evolution.Initialization.NonconservativeSystem",
                  "[Unit][Evolution][Actions]") {
  test<1>();
  test<2>();
  test<3>();
}
}  // namespace
