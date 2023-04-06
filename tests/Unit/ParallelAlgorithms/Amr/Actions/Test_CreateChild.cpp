// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <pup.h>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "Domain/Amr/Helpers.hpp"
#include "Domain/Amr/Tags/Flags.hpp"
#include "Domain/Amr/Tags/NeighborFlags.hpp"
#include "Domain/Structure/Direction.hpp"
#include "Domain/Structure/DirectionMap.hpp"
#include "Domain/Structure/Element.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Domain/Structure/Neighbors.hpp"
#include "Domain/Structure/OrientationMap.hpp"
#include "Domain/Structure/SegmentId.hpp"
#include "Domain/Tags.hpp"
#include "Framework/ActionTesting.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "Parallel/Phase.hpp"
#include "Parallel/PhaseDependentActionList.hpp"
#include "ParallelAlgorithms/Amr/Actions/CreateChild.hpp"
#include "Utilities/Literals.hpp"

namespace {

struct MockSendDataToChildren {
  template <typename ParallelComponent, typename DataBox,
            typename Metavariables, typename... Tags>
  static void apply(DataBox& /*box*/,
                    const Parallel::GlobalCache<Metavariables>& /*cache*/,
                    const ElementId<Metavariables::volume_dim>& /*parent_id*/,
                    const std::vector<ElementId<Metavariables::volume_dim>>&
                        ids_of_children) {
    CHECK(ids_of_children ==
          std::vector{ElementId<1>{0, std::array{SegmentId{3, 2}}},
                      ElementId<1>{0, std::array{SegmentId{3, 3}}}});
  }
};

template <typename Metavariables>
struct ArrayComponent {
  using metavariables = Metavariables;
  static constexpr size_t volume_dim = Metavariables::volume_dim;
  using chare_type = ActionTesting::MockArrayChare;
  using array_index = ElementId<volume_dim>;
  using const_global_cache_tags = tmpl::list<>;
  using simple_tags =
      tmpl::list<domain::Tags::Element<volume_dim>,
                 domain::Tags::Mesh<volume_dim>, amr::Tags::Flags<volume_dim>,
                 amr::Tags::NeighborFlags<volume_dim>>;
  using phase_dependent_action_list = tmpl::list<Parallel::PhaseActions<
      Parallel::Phase::Initialization,
      tmpl::list<ActionTesting::InitializeDataBox<simple_tags>>>>;
  using replace_these_simple_actions =
      tmpl::list<amr::Actions::SendDataToChildren>;
  using with_these_simple_actions = tmpl::list<MockSendDataToChildren>;
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
  const ElementId<1> parent_id{0, std::array{SegmentId{2, 1}}};
  const ElementId<1> parent_lower_neighbor_id{0, std::array{SegmentId{2, 0}}};
  const ElementId<1> parent_upper_neighbor_id{0, std::array{SegmentId{2, 2}}};
  DirectionMap<1, Neighbors<1>> parent_neighbors{};
  OrientationMap<1> aligned{};
  parent_neighbors.emplace(
      Direction<1>::lower_xi(),
      Neighbors<1>{std::unordered_set{parent_lower_neighbor_id}, aligned});
  parent_neighbors.emplace(
      Direction<1>::upper_xi(),
      Neighbors<1>{std::unordered_set{parent_upper_neighbor_id}, aligned});
  Element<1> parent{parent_id, std::move(parent_neighbors)};
  Mesh<1> parent_mesh{3, Spectral::Basis::Legendre,
                      Spectral::Quadrature::GaussLobatto};
  std::array<amr::Flag, 1> parent_flags{amr::Flag::Split};
  std::unordered_map<ElementId<1>, std::array<amr::Flag, 1>>
      parent_neighbor_flags;
  parent_neighbor_flags.emplace(parent_lower_neighbor_id,
                                std::array{amr::Flag::DoNothing});
  parent_neighbor_flags.emplace(parent_upper_neighbor_id,
                                std::array{amr::Flag::DoNothing});
  const auto children_ids = amr::ids_of_children(parent_id, parent_flags);

  ActionTesting::MockRuntimeSystem<Metavariables> runner{{}};
  ActionTesting::emplace_component_and_initialize<array_component>(
      &runner, parent_id,
      {parent, parent_mesh, parent_flags, parent_neighbor_flags});
  ActionTesting::emplace_component<singleton_component>(&runner, 0);
  for (const auto& child_id : children_ids) {
    ActionTesting::emplace_component<array_component>(&runner, child_id);
    CHECK(ActionTesting::is_simple_action_queue_empty<array_component>(
        runner, child_id));
  }
  CHECK(ActionTesting::is_simple_action_queue_empty<singleton_component>(runner,
                                                                         0));
  CHECK(ActionTesting::is_simple_action_queue_empty<array_component>(
      runner, parent_id));

  auto& cache = ActionTesting::cache<array_component>(runner, parent_id);
  auto& element_proxy =
      Parallel::get_parallel_component<array_component>(cache);

  // Call CreateChild, creating the first child and queueing CreateChild on
  // the singleton component in order to create the second child
  ActionTesting::simple_action<singleton_component, amr::Actions::CreateChild>(
      make_not_null(&runner), 0, element_proxy, parent_id, children_ids, 0_st);
  for (const auto& child_id : children_ids) {
    CHECK(ActionTesting::is_simple_action_queue_empty<array_component>(
        runner, child_id));
  }
  CHECK(ActionTesting::number_of_queued_simple_actions<singleton_component>(
            runner, 0) == 1);
  CHECK(ActionTesting::is_simple_action_queue_empty<array_component>(
      runner, parent_id));

  // Call CreateChild, creating the second child and queueing SendDataToChildren
  // on the parent element in order to send data to the first child
  ActionTesting::invoke_queued_simple_action<singleton_component>(
      make_not_null(&runner), 0);
  for (const auto& child_id : children_ids) {
    CHECK(ActionTesting::is_simple_action_queue_empty<array_component>(
        runner, child_id));
  }
  CHECK(ActionTesting::is_simple_action_queue_empty<singleton_component>(runner,
                                                                         0));
  CHECK(ActionTesting::number_of_queued_simple_actions<array_component>(
            runner, parent_id) == 1);
  // Invoke the mock action to check that CreateChild sent the correct data to
  // SendDataToChildren
  ActionTesting::invoke_queued_simple_action<array_component>(
      make_not_null(&runner), parent_id);
  CHECK(ActionTesting::is_simple_action_queue_empty<array_component>(
      runner, parent_id));
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Amr.Actions.CreateChild",
                  "[Unit][ParallelAlgorithms]") {
  test();
}
