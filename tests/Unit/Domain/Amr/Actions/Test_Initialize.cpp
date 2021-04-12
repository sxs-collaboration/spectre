// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cstddef>

#include "Domain/Amr/Actions/Initialize.hpp"
#include "Domain/Amr/Flag.hpp"
#include "Domain/Amr/Tags/Flags.hpp"
#include "Domain/Amr/Tags/NeighborFlags.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Framework/ActionTesting.hpp"
#include "Parallel/Actions/SetupDataBox.hpp"
#include "Parallel/PhaseDependentActionList.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeArray.hpp"
#include "Utilities/TMPL.hpp"

namespace {
template <size_t Dim, typename Metavariables>
struct Component {
  using metavariables = Metavariables;
  using chare_type = ActionTesting::MockArrayChare;
  using array_index = ElementId<Dim>;

  using phase_dependent_action_list = tmpl::list<Parallel::PhaseActions<
      typename Metavariables::Phase, Metavariables::Phase::Initialization,
      tmpl::list<Actions::SetupDataBox,
                 domain::amr::Actions::Initialize<Dim>>>>;
};

template <size_t Dim>
struct Metavariables {
  // static constexpr size_t volume_dim = Dim;
  using component_list = tmpl::list<Component<Dim, Metavariables>>;
  enum class Phase { Initialization, Exit };
};

template <size_t Dim>
void test() {
  using metavariables = Metavariables<Dim>;
  using component = Component<Dim, metavariables>;
  using MockRuntimeSystem = ActionTesting::MockRuntimeSystem<metavariables>;
  MockRuntimeSystem runner{{}};
  const ElementId<Dim> element_id{0};
  ActionTesting::emplace_array_component<component>(
      &runner, ActionTesting::NodeId{0}, ActionTesting::LocalCoreId{0},
      element_id);
  // Invoke SetupDataBox action
  ActionTesting::next_action<component>(make_not_null(&runner), element_id);
  // Invoke Initialize action
  ActionTesting::next_action<component>(make_not_null(&runner), element_id);
  CHECK(
      ActionTesting::get_databox_tag<component, domain::amr::Tags::Flags<Dim>>(
          runner, element_id) == make_array<Dim>(domain::amr::Flag::Undefined));
  CHECK(ActionTesting::get_databox_tag<component,
                                       domain::amr::Tags::NeighborFlags<Dim>>(
            runner, element_id)
            .empty());
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Domain.Amr.Actions.Initialize", "[Domain][Unit]") {
  test<1>();
  test<2>();
  test<3>();
}
