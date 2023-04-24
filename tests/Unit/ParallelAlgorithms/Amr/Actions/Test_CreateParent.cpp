// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <cstddef>
#include <deque>
#include <pup.h>
#include <vector>

#include "Domain/Structure/ElementId.hpp"
#include "Domain/Structure/SegmentId.hpp"
#include "Framework/ActionTesting.hpp"
#include "Parallel/GlobalCache.hpp"
#include "Parallel/Phase.hpp"
#include "Parallel/PhaseDependentActionList.hpp"
#include "ParallelAlgorithms/Amr/Actions/CollectDataFromChildren.hpp"
#include "ParallelAlgorithms/Amr/Actions/CreateParent.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

namespace {

struct MockCollectDataFromChildren {
  template <typename ParallelComponent, typename DataBox,
            typename Metavariables, typename... Tags>
  static void apply(
      DataBox& /*box*/, Parallel::GlobalCache<Metavariables>& /*cache*/,
      const ElementId<Metavariables::volume_dim>& /*child_id*/,
      const ElementId<Metavariables::volume_dim>& parent_id,
      std::deque<ElementId<Metavariables::volume_dim>> sibling_ids_to_collect) {
    CHECK(parent_id == ElementId<1>{0, std::array{SegmentId{2, 1}}});
    CHECK(sibling_ids_to_collect ==
          std::deque{ElementId<1>{0, std::array{SegmentId{3, 3}}}});
  }
};

template <typename Metavariables>
struct ArrayComponent {
  using metavariables = Metavariables;
  static constexpr size_t volume_dim = Metavariables::volume_dim;
  using chare_type = ActionTesting::MockArrayChare;
  using array_index = ElementId<volume_dim>;
  using const_global_cache_tags = tmpl::list<>;
  using simple_tags = tmpl::list<>;
  using phase_dependent_action_list = tmpl::list<Parallel::PhaseActions<
      Parallel::Phase::Initialization,
      tmpl::list<ActionTesting::InitializeDataBox<simple_tags>>>>;
  using replace_these_simple_actions =
      tmpl::list<amr::Actions::CollectDataFromChildren>;
  using with_these_simple_actions = tmpl::list<MockCollectDataFromChildren>;
};

template <typename Metavariables>
struct SingletonComponent {
  using metavariables = Metavariables;
  using array_index = int;
  static constexpr size_t volume_dim = Metavariables::volume_dim;
  using chare_type = ActionTesting::MockSingletonChare;
  using const_global_cache_tags = tmpl::list<>;
  using simple_tags = tmpl::list<>;
  using phase_dependent_action_list = tmpl::list<Parallel::PhaseActions<
      Parallel::Phase::Initialization,
      tmpl::list<ActionTesting::InitializeDataBox<simple_tags>>>>;
};

struct Metavariables {
  static constexpr size_t volume_dim = 1;
  using component_list = tmpl::list<ArrayComponent<Metavariables>,
                                    SingletonComponent<Metavariables>>;
  // NOLINTNEXTLINE(google-runtime-references)
  void pup(PUP::er& /*p*/) {}
};

void test() {
  using array_component = ArrayComponent<Metavariables>;
  using singleton_component = SingletonComponent<Metavariables>;
  const ElementId<1> lower_child_id{0, std::array{SegmentId{3, 2}}};
  const ElementId<1> upper_child_id{0, std::array{SegmentId{3, 3}}};
  const ElementId<1> parent_id{0, std::array{SegmentId{2, 1}}};

  ActionTesting::MockRuntimeSystem<Metavariables> runner{{}};
  ActionTesting::emplace_component<singleton_component>(&runner, 0);
  for (const auto& id :
       std::vector{lower_child_id, upper_child_id, parent_id}) {
    ActionTesting::emplace_component<array_component>(&runner, id);
    CHECK(ActionTesting::is_simple_action_queue_empty<array_component>(runner,
                                                                       id));
  }
  CHECK(ActionTesting::is_simple_action_queue_empty<singleton_component>(runner,
                                                                         0));

  auto& cache = ActionTesting::cache<array_component>(runner, parent_id);
  auto& element_proxy =
      Parallel::get_parallel_component<array_component>(cache);

  ActionTesting::simple_action<singleton_component, amr::Actions::CreateParent>(
      make_not_null(&runner), 0, element_proxy, parent_id, lower_child_id,
      std::deque{upper_child_id});
  for (const auto& id : std::vector{upper_child_id, parent_id}) {
    CHECK(ActionTesting::is_simple_action_queue_empty<array_component>(runner,
                                                                       id));
  }
  CHECK(ActionTesting::number_of_queued_simple_actions<array_component>(
            runner, lower_child_id) == 1);
  ActionTesting::invoke_queued_simple_action<array_component>(
      make_not_null(&runner), lower_child_id);
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Amr.Actions.CreateParent",
                  "[Unit][ParallelAlgorithms]") {
  test();
}
