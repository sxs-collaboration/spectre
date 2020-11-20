// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <string>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/DataVector.hpp"
#include "Framework/ActionTesting.hpp"
#include "Parallel/Actions/SetupDataBox.hpp"  // IWYU pragma: keep
#include "Parallel/Actions/TerminatePhase.hpp"
#include "Parallel/PhaseDependentActionList.hpp"  // IWYU pragma: keep
#include "ParallelAlgorithms/Initialization/MutateAssign.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

namespace {
struct Label1;
struct Label2;

struct CounterTag : db::SimpleTag {
  using type = size_t;
};

struct VectorTag : db::SimpleTag {
  using type = DataVector;
};

struct SquareVectorTag : db::SimpleTag {
  using type = DataVector;
};

struct SquareVectorCompute : SquareVectorTag, db::ComputeTag {
  using argument_tags = tmpl::list<VectorTag>;
  static void function(const gsl::not_null<DataVector*> result,
                       const DataVector& vector) noexcept {
    *result = square(vector);
  }
  using return_type = DataVector;
  using base = SquareVectorTag;
};

/// [initialization_action]
struct InitializationAction {
  using simple_tags = tmpl::list<CounterTag, VectorTag>;
  using compute_tags = tmpl::list<SquareVectorCompute>;
  template <typename DbTagsList, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent>
  static std::tuple<db::DataBox<DbTagsList>&&> apply(
      db::DataBox<DbTagsList>& box,
      const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
      const Parallel::GlobalCache<Metavariables>& /*cache*/,
      const ArrayIndex& /*array_index*/, const ActionList /*meta*/,
      const ParallelComponent* const /*meta*/) noexcept {
    Initialization::mutate_assign<tmpl::list<VectorTag>>(make_not_null(&box),
                                                         DataVector{1.2, 3.0});
    return {std::move(box)};
  }
};
/// [initialization_action]

struct MutateAction {
  template <typename DbTagsList, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent>
  static std::tuple<db::DataBox<DbTagsList>&&> apply(
      db::DataBox<DbTagsList>& box,
      const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
      const Parallel::GlobalCache<Metavariables>& /*cache*/,
      const ArrayIndex& /*array_index*/, const ActionList /*meta*/,
      const ParallelComponent* const /*meta*/) noexcept {
    db::mutate<CounterTag>(
        make_not_null(&box),
        [](const gsl::not_null<size_t*> counter) noexcept { *counter = 3_st; });
    return {std::move(box)};
  }
};

template <typename Metavariables>
struct Component {
  using metavariables = Metavariables;
  using chare_type = ActionTesting::MockArrayChare;
  using array_index = int;

  using phase_dependent_action_list = tmpl::list<Parallel::PhaseActions<
      typename Metavariables::Phase, Metavariables::Phase::Initialization,
      tmpl::list<Actions::SetupDataBox, InitializationAction, MutateAction>>>;
};

struct Metavariables {
  using component_list = tmpl::list<Component<Metavariables>>;

  enum class Phase { Initialization, Exit };
};
}  // namespace

SPECTRE_TEST_CASE("Unit.Parallel.SetupDataBox", "[Unit][Parallel][Actions]") {
  using component = Component<Metavariables>;
  using MockRuntimeSystem = ActionTesting::MockRuntimeSystem<Metavariables>;
  MockRuntimeSystem runner{{}};
  ActionTesting::emplace_component<component>(&runner, 0);

  // Actions::SetupDataBox
  runner.next_action<component>(0);
  // check that all of the DataBox contents have been default-constructed
  CHECK(ActionTesting::get_databox_tag<component, CounterTag>(runner, 0_st) ==
        0_st);
  CHECK(ActionTesting::get_databox_tag<component, VectorTag>(runner, 0_st)
            .size() == 0_st);
  CHECK(ActionTesting::get_databox_tag<component, SquareVectorTag>(runner, 0_st)
            .size() == 0_st);
  // InitializationAction
  runner.next_action<component>(0);
  CHECK(ActionTesting::get_databox_tag<component, CounterTag>(runner, 0_st) ==
        0_st);
  CHECK(ActionTesting::get_databox_tag<component, VectorTag>(runner, 0_st) ==
        DataVector{1.2, 3.0});
  CHECK(ActionTesting::get_databox_tag<component, SquareVectorTag>(
            runner, 0_st) == DataVector{1.44, 9.0});
  // MutateAction
  runner.next_action<component>(0);
  CHECK(ActionTesting::get_databox_tag<component, CounterTag>(runner, 0_st) ==
        3_st);
  CHECK(ActionTesting::get_databox_tag<component, VectorTag>(runner, 0_st) ==
        DataVector{1.2, 3.0});
  CHECK(ActionTesting::get_databox_tag<component, SquareVectorTag>(
            runner, 0_st) == DataVector{1.44, 9.0});
}
