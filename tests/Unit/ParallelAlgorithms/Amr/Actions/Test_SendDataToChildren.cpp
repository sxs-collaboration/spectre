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
#include "Helpers/Domain/Amr/RegistrationHelpers.hpp"
#include "NumericalAlgorithms/SpatialDiscretization/Mesh.hpp"
#include "Parallel/Phase.hpp"
#include "Parallel/PhaseDependentActionList.hpp"
#include "Parallel/Protocols/RegistrationMetavariables.hpp"
#include "ParallelAlgorithms/Amr/Actions/SendDataToChildren.hpp"
#include "Utilities/ProtocolHelpers.hpp"

namespace {

Element<1> create_parent() {
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
  return Element<1>{parent_id, std::move(parent_neighbors)};
}

Mesh<1> create_parent_mesh() {
  return Mesh<1>{3, SpatialDiscretization::Basis::Legendre,
                 SpatialDiscretization::Quadrature::GaussLobatto};
}

std::array<amr::Flag, 1> create_parent_flags() {
  return std::array{amr::Flag::Split};
}

std::unordered_map<ElementId<1>, std::array<amr::Flag, 1>>
create_parent_neighbor_flags() {
  const ElementId<1> parent_lower_neighbor_id{0, std::array{SegmentId{2, 0}}};
  const ElementId<1> parent_upper_neighbor_id{0, std::array{SegmentId{2, 2}}};
  std::unordered_map<ElementId<1>, std::array<amr::Flag, 1>> result{};
  result.emplace(parent_lower_neighbor_id, std::array{amr::Flag::DoNothing});
  result.emplace(parent_upper_neighbor_id, std::array{amr::Flag::DoNothing});
  return result;
}

struct MockInitializeChild {
  template <typename ParallelComponent, typename DataBox,
            typename Metavariables, typename... Tags>
  static void apply(DataBox& /*box*/,
                    const Parallel::GlobalCache<Metavariables>& /*cache*/,
                    const ElementId<Metavariables::volume_dim>& /*child_id*/,
                    const tuples::TaggedTuple<Tags...>& parent_items) {
    CHECK(get<domain::Tags::Element<1>>(parent_items) == create_parent());
    CHECK(get<domain::Tags::Mesh<1>>(parent_items) == create_parent_mesh());
    CHECK(get<amr::Tags::Flags<1>>(parent_items) == create_parent_flags());
    CHECK(get<amr::Tags::NeighborFlags<1>>(parent_items) ==
          create_parent_neighbor_flags());
  }
};

template <typename Metavariables>
struct Component {
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
      tmpl::list<amr::Actions::InitializeChild>;
  using with_these_simple_actions = tmpl::list<MockInitializeChild>;
};

struct Metavariables {
  static constexpr size_t volume_dim = 1;
  using component_list = tmpl::list<Component<Metavariables>,
                                    TestHelpers::amr::Registrar<Metavariables>>;
  // NOLINTNEXTLINE(google-runtime-references)
  void pup(PUP::er& /*p*/) {}

  struct registration
      : tt::ConformsTo<Parallel::protocols::RegistrationMetavariables> {
    using element_registrars =
        tmpl::map<tmpl::pair<Component<Metavariables>,
                             tmpl::list<TestHelpers::amr::RegisterElement>>>;
  };
};

void test() {
  const auto parent = create_parent();
  const ElementId<1>& parent_id = parent.id();
  const auto parent_mesh = create_parent_mesh();
  const auto parent_flags = create_parent_flags();
  const auto parent_neighbor_flags = create_parent_neighbor_flags();
  const auto children_ids = amr::ids_of_children(parent_id, parent_flags);

  using array_component = Component<Metavariables>;
  using registrar = TestHelpers::amr::Registrar<Metavariables>;

  ActionTesting::MockRuntimeSystem<Metavariables> runner{{}};
  ActionTesting::emplace_component_and_initialize<array_component>(
      &runner, parent_id,
      {parent, parent_mesh, parent_flags, parent_neighbor_flags});
  for (const auto& child_id : children_ids) {
    ActionTesting::emplace_component<array_component>(&runner, child_id);
    CHECK(ActionTesting::number_of_queued_simple_actions<array_component>(
              runner, child_id) == 0);
  }
  ActionTesting::emplace_group_component_and_initialize<registrar>(
      &runner, std::unordered_set{parent_id});

  ActionTesting::simple_action<array_component,
                               amr::Actions::SendDataToChildren>(
      make_not_null(&runner), parent_id, children_ids);
  // SendDataToChildren calls InitializeChild on each child element
  for (const auto& child_id : children_ids) {
    CHECK(ActionTesting::number_of_queued_simple_actions<array_component>(
              runner, child_id) == 1);
    // Invoke the mock action to check that SendDataToChildren sent the correct
    // data
    ActionTesting::invoke_queued_simple_action<array_component>(
        make_not_null(&runner), child_id);
    CHECK(ActionTesting::is_simple_action_queue_empty<array_component>(
        runner, child_id));
  }
  CHECK(ActionTesting::number_of_queued_simple_actions<registrar>(runner, 0) ==
        1);
  ActionTesting::invoke_queued_simple_action<registrar>(make_not_null(&runner),
                                                        0);
  CHECK(ActionTesting::get_databox_tag<registrar,
                                       TestHelpers::amr::RegisteredElements<1>>(
            runner, 0) == std::unordered_set<ElementId<1>>{});
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Amr.Actions.SendDataToChildren",
                  "[Unit][ParallelAlgorithms]") {
  test();
}
