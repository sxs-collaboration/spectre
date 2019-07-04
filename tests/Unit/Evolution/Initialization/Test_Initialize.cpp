// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "tests/Unit/TestingFramework.hpp"

#include <string>
#include <tuple>
#include <utility>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/DataBoxTag.hpp"
#include "Evolution/Initialization/Initialize.hpp"
#include "Evolution/Initialization/MergeIntoDataBox.hpp"
#include "Evolution/Initialization/Tags.hpp"
#include "Parallel/Actions/TerminatePhase.hpp"
#include "Parallel/AddOptionsToDataBox.hpp"
#include "Parallel/ConstGlobalCache.hpp"
#include "Parallel/PhaseDependentActionList.hpp"  // IWYU pragma: keep
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"
#include "tests/Unit/ActionTesting.hpp"

namespace {
struct TemporalId {};

struct DummyTimeTag : db::SimpleTag {
  static std::string name() noexcept { return "DummyTime"; }
  using type = double;
};

template <typename Tag>
struct TagMultiplyByTwo : db::ComputeTag {
  static std::string name() noexcept { return "MultiplyByTwo"; }
  static double function(const double& t) noexcept { return t * 2.0; }
  using argument_tags = tmpl::list<Tag>;
};

struct Action0 {
  using initialization_option_tags =
      tmpl::list<Initialization::Tags::InitialTime>;

  template <typename DbTagsList, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent,
            Requires<tmpl::list_contains_v<
                DbTagsList, Initialization::Tags::InitialTime>> = nullptr>
  static auto apply(db::DataBox<DbTagsList>& box,
                    const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
                    const Parallel::ConstGlobalCache<Metavariables>& /*cache*/,
                    const ArrayIndex& /*array_index*/, ActionList /*meta*/,
                    const ParallelComponent* const /*meta*/) noexcept {
    const double initial_time_value =
        db::get<Initialization::Tags::InitialTime>(box);
    return std::make_tuple(
        Initialization::merge_into_databox<Action0,
                                           db::AddSimpleTags<DummyTimeTag>>(
            std::move(box), 3.0 * initial_time_value));
  }

  template <typename DbTagsList, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent,
            Requires<not tmpl::list_contains_v<
                DbTagsList, Initialization::Tags::InitialTime>> = nullptr>
  static std::tuple<db::DataBox<DbTagsList>&&> apply(
      db::DataBox<DbTagsList>& box,
      const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
      const Parallel::ConstGlobalCache<Metavariables>& /*cache*/,
      const ArrayIndex& /*array_index*/, ActionList /*meta*/,
      const ParallelComponent* const /*meta*/) noexcept {
    return {std::move(box)};
  }
};

struct Action1 {
  template <typename DbTagsList, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent>
  static auto apply(db::DataBox<DbTagsList>& box,
                    const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
                    const Parallel::ConstGlobalCache<Metavariables>& /*cache*/,
                    const ArrayIndex& /*array_index*/, ActionList /*meta*/,
                    const ParallelComponent* const /*meta*/) noexcept {
    return std::make_tuple(Initialization::merge_into_databox<
                           Action1, db::AddSimpleTags<>,
                           db::AddComputeTags<TagMultiplyByTwo<DummyTimeTag>>>(
        std::move(box)));
  }
};

template <typename Metavariables>
struct Component {
  using metavariables = Metavariables;
  using chare_type = ActionTesting::MockArrayChare;
  using array_index = int;
  using const_global_cache_tag_list = tmpl::list<>;

  /// [actions]
  using initialization_actions =
      tmpl::list<Action0, Action1,
                 Initialization::Actions::RemoveOptionsAndTerminatePhase>;
  /// [actions]

  /// [options_to_databox]
  using add_options_to_databox = Parallel::ForwardAllOptionsToDataBox<
      Initialization::option_tags<initialization_actions>>;
  /// [options_to_databox]

  using phase_dependent_action_list =
      tmpl::list<Parallel::PhaseActions<typename Metavariables::Phase,
                                        Metavariables::Phase::Initialization,
                                        initialization_actions>>;
};

struct Metavariables {
  using component_list = tmpl::list<Component<Metavariables>>;
  using const_global_cache_tag_list = tmpl::list<>;
  using temporal_id = TemporalId;

  enum class Phase { Initialization, Exit };
};
}  // namespace

SPECTRE_TEST_CASE("Unit.Evolution.Initialization.RemoveOptionsFromDataBox",
                  "[Unit][Evolution]") {
  using MockRuntimeSystem = ActionTesting::MockRuntimeSystem<Metavariables>;
  using component = Component<Metavariables>;

  MockRuntimeSystem runner{{}};
  const double initial_time = 3.7;
  ActionTesting::emplace_component<component>(&runner, 0, initial_time);
  runner.set_phase(Metavariables::Phase::Initialization);
  CHECK(ActionTesting::tag_is_retrievable<component,
                                          Initialization::Tags::InitialTime>(
      runner, 0));
  CHECK(not ActionTesting::tag_is_retrievable<component, DummyTimeTag>(runner,
                                                                       0));
  CHECK(not ActionTesting::tag_is_retrievable<component,
                                              TagMultiplyByTwo<DummyTimeTag>>(
      runner, 0));
  CHECK(ActionTesting::get_databox_tag<component,
                                       Initialization::Tags::InitialTime>(
            runner, 0) == initial_time);
  CHECK_FALSE(ActionTesting::get_terminate<component>(runner, 0));
  // Runs Action0
  runner.next_action<component>(0);
  CHECK(ActionTesting::tag_is_retrievable<component,
                                          Initialization::Tags::InitialTime>(
      runner, 0));
  CHECK(ActionTesting::tag_is_retrievable<component, DummyTimeTag>(runner, 0));
  CHECK(not ActionTesting::tag_is_retrievable<component,
                                              TagMultiplyByTwo<DummyTimeTag>>(
      runner, 0));
  CHECK(ActionTesting::get_databox_tag<component,
                                       Initialization::Tags::InitialTime>(
            runner, 0) == initial_time);
  CHECK(ActionTesting::get_databox_tag<component, DummyTimeTag>(runner, 0) ==
        3.0 * initial_time);
  CHECK_FALSE(ActionTesting::get_terminate<component>(runner, 0));
  // Runs Action1
  runner.next_action<component>(0);
  CHECK(ActionTesting::tag_is_retrievable<component,
                                          Initialization::Tags::InitialTime>(
      runner, 0));
  CHECK(ActionTesting::tag_is_retrievable<component, DummyTimeTag>(runner, 0));
  CHECK(ActionTesting::tag_is_retrievable<component,
                                          TagMultiplyByTwo<DummyTimeTag>>(
      runner, 0));
  CHECK(ActionTesting::get_databox_tag<component,
                                       Initialization::Tags::InitialTime>(
            runner, 0) == initial_time);
  CHECK(ActionTesting::get_databox_tag<component, DummyTimeTag>(runner, 0) ==
        3.0 * initial_time);
  CHECK(
      ActionTesting::get_databox_tag<component, TagMultiplyByTwo<DummyTimeTag>>(
          runner, 0) == 6.0 * initial_time);
  CHECK_FALSE(ActionTesting::get_terminate<component>(runner, 0));
  // Runs RemoveOptionsFromDataBox
  runner.next_action<component>(0);
  CHECK(not ActionTesting::tag_is_retrievable<
        component, Initialization::Tags::InitialTime>(runner, 0));
  CHECK(ActionTesting::tag_is_retrievable<component, DummyTimeTag>(runner, 0));
  CHECK(ActionTesting::tag_is_retrievable<component,
                                          TagMultiplyByTwo<DummyTimeTag>>(
      runner, 0));
  CHECK(ActionTesting::get_databox_tag<component, DummyTimeTag>(runner, 0) ==
        3.0 * initial_time);
  CHECK(
      ActionTesting::get_databox_tag<component, TagMultiplyByTwo<DummyTimeTag>>(
          runner, 0) == 6.0 * initial_time);
  CHECK(ActionTesting::get_terminate<component>(runner, 0));
}
