// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cstddef>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/Tag.hpp"
#include "Framework/ActionTesting.hpp"
#include "Framework/TestHelpers.hpp"
#include "Parallel/AlgorithmMetafunctions.hpp"
#include "Parallel/GlobalCache.hpp"
#include "Parallel/InboxInserters.hpp"
#include "Parallel/Invoke.hpp"
#include "Parallel/ParallelComponentHelpers.hpp"
#include "Parallel/PhaseDependentActionList.hpp"  // IWYU pragma: keep
#include "Utilities/Gsl.hpp"
#include "Utilities/Literals.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

namespace {
namespace Tags {
struct MapCounter : db::SimpleTag {
  using type = size_t;
};

struct MemberInsertCounter : db::SimpleTag {
  using type = size_t;
};

struct ValueCounter : db::SimpleTag {
  using type = size_t;
};

struct PushbackCounter : db::SimpleTag {
  using type = size_t;
};

// [map tag]
struct MapTag : public Parallel::InboxInserters::Map<MapTag> {
  using temporal_id = size_t;
  using type = std::unordered_map<temporal_id, std::unordered_map<int, int>>;
};
// [map tag]

// [member insert tag]
struct MemberInsertTag
    : public Parallel::InboxInserters::MemberInsert<MemberInsertTag> {
  using temporal_id = size_t;
  using type = std::unordered_map<temporal_id, std::unordered_multiset<int>>;
};
// [member insert tag]

// [value tag]
struct ValueTag : public Parallel::InboxInserters::Value<ValueTag> {
  using temporal_id = size_t;
  using type = std::unordered_map<temporal_id, int>;
};
// [value tag]

// [pushback tag]
struct PushbackTag : public Parallel::InboxInserters::Pushback<PushbackTag> {
  using temporal_id = size_t;
  using type = std::unordered_map<temporal_id, std::vector<int>>;
};
// [pushback tag]
}  // namespace Tags

namespace Actions {
struct SendMap {
  template <typename DbTagsList, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent>
  static auto apply(db::DataBox<DbTagsList>& box,
                    const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
                    Parallel::GlobalCache<Metavariables>& cache,
                    const ArrayIndex& /*array_index*/, ActionList /*meta*/,
                    const ParallelComponent* const /*meta*/) {
    Parallel::receive_data<Tags::MapTag>(
        Parallel::get_parallel_component<ParallelComponent>(cache)[0], 1_st,
        std::make_pair(10, 23));
    db::mutate<Tags::MapCounter>(
        make_not_null(&box),
        [](const gsl::not_null<size_t*> map_counter) { (*map_counter)++; });
    return std::forward_as_tuple(std::move(box));
  }
};

struct ReceiveMap {
  using inbox_tags = tmpl::list<Tags::MapTag>;

  template <typename DbTagsList, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent>
  static std::tuple<db::DataBox<DbTagsList>&&, Parallel::AlgorithmExecution>
  apply(db::DataBox<DbTagsList>& box,
        const tuples::TaggedTuple<InboxTags...>& inboxes,
        const Parallel::GlobalCache<Metavariables>& /*cache*/,
        const ArrayIndex& /*array_index*/, ActionList /*meta*/,
        const ParallelComponent* const /*meta*/) {
    if (get<Tags::MapTag>(inboxes).at(1_st).size() != 1) {
      return {std::move(box), Parallel::AlgorithmExecution::Retry};
    }

    db::mutate<Tags::MapCounter>(
        make_not_null(&box),
        [](const gsl::not_null<size_t*> map_counter) { (*map_counter)++; });
    return {std::move(box), Parallel::AlgorithmExecution::Continue};
  }
};

struct SendMemberInsert {
  template <typename DbTagsList, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent>
  static auto apply(db::DataBox<DbTagsList>& box,
                    const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
                    Parallel::GlobalCache<Metavariables>& cache,
                    const ArrayIndex& /*array_index*/, ActionList /*meta*/,
                    const ParallelComponent* const /*meta*/) {
    Parallel::receive_data<Tags::MemberInsertTag>(
        Parallel::get_parallel_component<ParallelComponent>(cache)[0], 1_st,
        23);
    db::mutate<Tags::MemberInsertCounter>(
        make_not_null(&box),
        [](const gsl::not_null<size_t*> member_insert_counter) {
          (*member_insert_counter)++;
        });
    return std::forward_as_tuple(std::move(box));
  }
};

struct ReceiveMemberInsert {
  using inbox_tags = tmpl::list<Tags::MemberInsertTag>;

  template <typename DbTagsList, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent>
  static std::tuple<db::DataBox<DbTagsList>&&, Parallel::AlgorithmExecution>
  apply(db::DataBox<DbTagsList>& box,
        const tuples::TaggedTuple<InboxTags...>& inboxes,
        const Parallel::GlobalCache<Metavariables>& /*cache*/,
        const ArrayIndex& /*array_index*/, ActionList /*meta*/,
        const ParallelComponent* const /*meta*/) {
    if (get<Tags::MemberInsertTag>(inboxes).at(1_st).size() != 1) {
      return {std::move(box), Parallel::AlgorithmExecution::Retry};
    }

    db::mutate<Tags::MemberInsertCounter>(
        make_not_null(&box),
        [](const gsl::not_null<size_t*> member_insert_counter) {
          (*member_insert_counter)++;
        });
    return {std::move(box), Parallel::AlgorithmExecution::Continue};
  }
};

struct SendValue {
  template <typename DbTagsList, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent>
  static auto apply(db::DataBox<DbTagsList>& box,
                    const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
                    Parallel::GlobalCache<Metavariables>& cache,
                    const ArrayIndex& /*array_index*/, ActionList /*meta*/,
                    const ParallelComponent* const /*meta*/) {
    Parallel::receive_data<Tags::ValueTag>(
        Parallel::get_parallel_component<ParallelComponent>(cache)[0], 1_st,
        23);
    db::mutate<Tags::ValueCounter>(
        make_not_null(&box),
        [](const gsl::not_null<size_t*> value_counter) { (*value_counter)++; });
    return std::forward_as_tuple(std::move(box));
  }
};

struct ReceiveValue {
  using inbox_tags = tmpl::list<Tags::ValueTag>;

  template <typename DbTagsList, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent>
  static std::tuple<db::DataBox<DbTagsList>&&, Parallel::AlgorithmExecution>
  apply(db::DataBox<DbTagsList>& box,
        const tuples::TaggedTuple<InboxTags...>& inboxes,
        const Parallel::GlobalCache<Metavariables>& /*cache*/,
        const ArrayIndex& /*array_index*/, ActionList /*meta*/,
        const ParallelComponent* const /*meta*/) {
    if (get<Tags::ValueTag>(inboxes).count(1_st) != 1) {
      return {std::move(box), Parallel::AlgorithmExecution::Retry};
    }

    db::mutate<Tags::ValueCounter>(
        make_not_null(&box),
        [](const gsl::not_null<size_t*> value_counter) { (*value_counter)++; });
    return {std::move(box), Parallel::AlgorithmExecution::Continue};
  }
};

struct SendPushback {
  template <typename DbTagsList, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent>
  static auto apply(db::DataBox<DbTagsList>& box,
                    const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
                    Parallel::GlobalCache<Metavariables>& cache,
                    const ArrayIndex& /*array_index*/, ActionList /*meta*/,
                    const ParallelComponent* const /*meta*/) {
    Parallel::receive_data<Tags::PushbackTag>(
        Parallel::get_parallel_component<ParallelComponent>(cache)[0], 1_st,
        23);
    db::mutate<Tags::PushbackCounter>(
        make_not_null(&box), [](const gsl::not_null<size_t*> pushback_counter) {
          (*pushback_counter)++;
        });
    return std::forward_as_tuple(std::move(box));
  }
};

struct ReceivePushback {
  using inbox_tags = tmpl::list<Tags::PushbackTag>;

  template <typename DbTagsList, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent>
  static std::tuple<db::DataBox<DbTagsList>&&, Parallel::AlgorithmExecution>
  apply(db::DataBox<DbTagsList>& box,
        const tuples::TaggedTuple<InboxTags...>& inboxes,
        const Parallel::GlobalCache<Metavariables>& /*cache*/,
        const ArrayIndex& /*array_index*/, ActionList /*meta*/,
        const ParallelComponent* const /*meta*/) {
    if (get<Tags::PushbackTag>(inboxes).at(1_st).size() != 1) {
      return {std::move(box), Parallel::AlgorithmExecution::Retry};
    }

    db::mutate<Tags::PushbackCounter>(
        make_not_null(&box), [](const gsl::not_null<size_t*> pushback_counter) {
          (*pushback_counter)++;
        });
    return {std::move(box), Parallel::AlgorithmExecution::Continue};
  }
};
}  // namespace Actions

template <typename Metavariables>
struct Component {
  using metavariables = Metavariables;
  using chare_type = ActionTesting::MockArrayChare;
  using array_index = size_t;

  using phase_dependent_action_list = tmpl::list<
      Parallel::PhaseActions<
          typename Metavariables::Phase, Metavariables::Phase::Initialization,
          tmpl::list<ActionTesting::InitializeDataBox<
              db::AddSimpleTags<Tags::MapCounter, Tags::MemberInsertCounter,
                                Tags::ValueCounter, Tags::PushbackCounter>>>>,

      Parallel::PhaseActions<
          typename Metavariables::Phase, Metavariables::Phase::Testing,
          tmpl::list<Actions::SendMap, Actions::ReceiveMap,
                     Actions::SendMemberInsert, Actions::ReceiveMemberInsert,
                     Actions::SendValue, Actions::ReceiveValue,
                     Actions::SendPushback, Actions::ReceivePushback>>>;
};

struct Metavariables {
  using component_list = tmpl::list<Component<Metavariables>>;

  enum class Phase { Initialization, Testing, Exit };
};

SPECTRE_TEST_CASE("Unit.Parallel.InboxInserters", "[Parallel][Unit]") {
  using metavars = Metavariables;
  using component = Component<metavars>;

  ActionTesting::MockRuntimeSystem<metavars> runner{{}};
  ActionTesting::emplace_component_and_initialize<component>(
      &runner, 0, {0_st, 0_st, 0_st, 0_st});
  ActionTesting::set_phase(make_not_null(&runner),
                           Metavariables::Phase::Testing);

  // Check map insertion
  CHECK(ActionTesting::get_next_action_index<component>(runner, 0) == 0);
  CHECK(ActionTesting::get_databox_tag<component, Tags::MapCounter>(runner,
                                                                    0) == 0);
  ActionTesting::next_action<component>(make_not_null(&runner), 0);
  CHECK(ActionTesting::get_inbox_tag<component, Tags::MapTag>(runner, 0)
            .at(1)
            .size() == 1);
  CHECK(
      ActionTesting::get_inbox_tag<component, Tags::MapTag>(runner, 0).at(1).at(
          10) == 23);
  CHECK(ActionTesting::get_next_action_index<component>(runner, 0) == 1);
  CHECK(ActionTesting::get_databox_tag<component, Tags::MapCounter>(runner,
                                                                    0) == 1);
  ActionTesting::next_action<component>(make_not_null(&runner), 0);
  CHECK(ActionTesting::get_databox_tag<component, Tags::MapCounter>(runner,
                                                                    0) == 2);
  ActionTesting::get_inbox_tag<component, Tags::MapTag>(make_not_null(&runner),
                                                        0)
      .clear();

  // Check member insertion
  CHECK(ActionTesting::get_next_action_index<component>(runner, 0) == 2);
  CHECK(ActionTesting::get_databox_tag<component, Tags::MemberInsertCounter>(
            runner, 0) == 0);
  ActionTesting::next_action<component>(make_not_null(&runner), 0);
  CHECK(
      ActionTesting::get_inbox_tag<component, Tags::MemberInsertTag>(runner, 0)
          .at(1)
          .size() == 1);
  CHECK(
      ActionTesting::get_inbox_tag<component, Tags::MemberInsertTag>(runner, 0)
          .at(1)
          .count(23) == 1);
  CHECK(ActionTesting::get_next_action_index<component>(runner, 0) == 3);
  CHECK(ActionTesting::get_databox_tag<component, Tags::MemberInsertCounter>(
            runner, 0) == 1);
  ActionTesting::next_action<component>(make_not_null(&runner), 0);
  CHECK(ActionTesting::get_databox_tag<component, Tags::MemberInsertCounter>(
            runner, 0) == 2);
  ActionTesting::get_inbox_tag<component, Tags::MemberInsertTag>(
      make_not_null(&runner), 0)
      .clear();

  // Check value insertion
  CHECK(ActionTesting::get_next_action_index<component>(runner, 0) == 4);
  CHECK(ActionTesting::get_databox_tag<component, Tags::ValueCounter>(runner,
                                                                      0) == 0);
  ActionTesting::next_action<component>(make_not_null(&runner), 0);
  CHECK(ActionTesting::get_inbox_tag<component, Tags::ValueTag>(runner, 0).at(
            1) == 23);
  CHECK(ActionTesting::get_next_action_index<component>(runner, 0) == 5);
  CHECK(ActionTesting::get_databox_tag<component, Tags::ValueCounter>(runner,
                                                                      0) == 1);
  ActionTesting::next_action<component>(make_not_null(&runner), 0);
  CHECK(ActionTesting::get_databox_tag<component, Tags::ValueCounter>(runner,
                                                                      0) == 2);
  ActionTesting::get_inbox_tag<component, Tags::ValueTag>(
      make_not_null(&runner), 0)
      .clear();

  // Check pushback insertion
  CHECK(ActionTesting::get_next_action_index<component>(runner, 0) == 6);
  CHECK(ActionTesting::get_databox_tag<component, Tags::PushbackCounter>(
            runner, 0) == 0);
  ActionTesting::next_action<component>(make_not_null(&runner), 0);
  CHECK(ActionTesting::get_inbox_tag<component, Tags::PushbackTag>(runner, 0)
            .at(1)
            .size() == 1);
  CHECK(
      ActionTesting::get_inbox_tag<component, Tags::PushbackTag>(runner, 0).at(
          1)[0] == 23);
  CHECK(ActionTesting::get_next_action_index<component>(runner, 0) == 7);
  CHECK(ActionTesting::get_databox_tag<component, Tags::PushbackCounter>(
            runner, 0) == 1);
  ActionTesting::next_action<component>(make_not_null(&runner), 0);
  CHECK(ActionTesting::get_databox_tag<component, Tags::PushbackCounter>(
            runner, 0) == 2);
  ActionTesting::get_inbox_tag<component, Tags::PushbackTag>(
      make_not_null(&runner), 0)
      .clear();
}
}  // namespace
