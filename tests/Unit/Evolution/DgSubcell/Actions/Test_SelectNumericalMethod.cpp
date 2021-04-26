// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cstddef>

#include "Evolution/DgSubcell/Actions/Labels.hpp"
#include "Evolution/DgSubcell/Actions/SelectNumericalMethod.hpp"
#include "Evolution/DgSubcell/ActiveGrid.hpp"
#include "Evolution/DgSubcell/Tags/ActiveGrid.hpp"
#include "Framework/ActionTesting.hpp"
#include "Parallel/Actions/Goto.hpp"
#include "Utilities/TMPL.hpp"

namespace {
// Use Actions::Label<DummyLabel> actions to space out the action list so we can
// check that SelectNumericalMethod is jumping to the right locations.
template <size_t Index>
struct DummyLabel {};

template <typename Metavariables>
struct component {
  using metavariables = Metavariables;
  using chare_type = ActionTesting::MockArrayChare;
  using array_index = size_t;
  using const_global_cache_tag_list = tmpl::list<>;

  using initial_tags = tmpl::list<evolution::dg::subcell::Tags::ActiveGrid>;

  using phase_dependent_action_list = tmpl::list<Parallel::PhaseActions<
      typename Metavariables::Phase, Metavariables::Phase::Initialization,
      tmpl::list<
          ActionTesting::InitializeDataBox<initial_tags>,
          evolution::dg::subcell::Actions::SelectNumericalMethod,
          Actions::Label<DummyLabel<0>>,
          Actions::Label<evolution::dg::subcell::Actions::Labels::BeginDg>,
          Actions::Label<DummyLabel<1>>,
          Actions::Label<evolution::dg::subcell::Actions::Labels::BeginSubcell>,
          Actions::Label<DummyLabel<2>>>>>;
};

struct Metavariables {
  using component_list = tmpl::list<component<Metavariables>>;
  enum class Phase { Initialization, Exit };
};

void test(const evolution::dg::subcell::ActiveGrid active_grid) {
  CAPTURE(active_grid);
  using metavars = Metavariables;
  using comp = component<metavars>;
  using MockRuntimeSystem = ActionTesting::MockRuntimeSystem<metavars>;
  MockRuntimeSystem runner{{}};
  ActionTesting::emplace_array_component_and_initialize<comp>(
      &runner, ActionTesting::NodeId{0}, ActionTesting::LocalCoreId{0}, 0,
      {active_grid});

  // Invoke the SelectNumericalMethod action on the runner
  ActionTesting::next_action<comp>(make_not_null(&runner), 0);
  // SelectNumericalMethod jumps to `index_of<Label>+1`
  const size_t expected_next_action_index =
      (active_grid == evolution::dg::subcell::ActiveGrid::Dg ? 4 : 6);
  CHECK(ActionTesting::get_next_action_index<comp>(runner, 0) ==
        expected_next_action_index);
}

SPECTRE_TEST_CASE("Unit.Evolution.Subcell.Actions.SelectNumericalMethod",
                  "[Evolution][Unit]") {
  test(evolution::dg::subcell::ActiveGrid::Dg);
  test(evolution::dg::subcell::ActiveGrid::Subcell);
}
}  // namespace
