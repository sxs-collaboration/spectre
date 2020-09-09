// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cstddef>
#include <map>
#include <tuple>
#include <utility>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/Tag.hpp"
#include "Framework/ActionTesting.hpp"
#include "Parallel/Actions/Receive.hpp"
#include "Parallel/ExtractFromInbox.hpp"
#include "Parallel/GlobalCache.hpp"
#include "Parallel/InboxInserters.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

namespace {

/// [receive_action]
struct TemporalIdTag : db::SimpleTag {
  using type = size_t;
};

struct SampleDataTag : Parallel::InboxInserters::Value<SampleDataTag> {
  using temporal_id = size_t;
  using type = std::map<temporal_id, int>;
};

struct ReceiveSampleData
    : Parallel::Actions::Receive<SampleDataTag, TemporalIdTag> {
  template <typename DbTagsList, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent>
  static std::tuple<db::DataBox<DbTagsList>&&> apply(
      db::DataBox<DbTagsList>& box, tuples::TaggedTuple<InboxTags...>& inboxes,
      const Parallel::GlobalCache<Metavariables>& /*cache*/,
      const ArrayIndex& /*array_index*/, const ActionList /*meta*/,
      const ParallelComponent* const /*meta*/) noexcept {
    const int received_data =
        Parallel::extract_from_inbox<SampleDataTag, TemporalIdTag>(inboxes,
                                                                   box);
    CHECK(received_data == 1);
    return {std::move(box)};
  }
};
/// [receive_action]

template <typename Metavariables>
struct Component {
  using metavariables = Metavariables;
  using chare_type = ActionTesting::MockArrayChare;
  using array_index = int;
  using phase_dependent_action_list = tmpl::list<
      Parallel::PhaseActions<typename Metavariables::Phase,
                             Metavariables::Phase::Initialization,
                             tmpl::list<ActionTesting::InitializeDataBox<
                                 db::AddSimpleTags<TemporalIdTag>>>>,
      Parallel::PhaseActions<typename Metavariables::Phase,
                             Metavariables::Phase::Testing,
                             tmpl::list<ReceiveSampleData>>>;
};

struct Metavariables {
  using component_list = tmpl::list<Component<Metavariables>>;
  enum class Phase { Initialization, Testing, Exit };
};
}  // namespace

SPECTRE_TEST_CASE("Unit.Parallel.Actions.ReceiveSampleData",
                  "[Unit][Parallel][Actions]") {
  using component = Component<Metavariables>;

  ActionTesting::MockRuntimeSystem<Metavariables> runner{{}};
  const size_t temporal_id = 0;
  ActionTesting::emplace_component_and_initialize<component>(&runner, 0,
                                                             {temporal_id});

  ActionTesting::set_phase(make_not_null(&runner),
                           Metavariables::Phase::Testing);

  CHECK_FALSE(ActionTesting::is_ready<component>(runner, 0));
  ActionTesting::get_inbox_tag<component, SampleDataTag, Metavariables>(
      make_not_null(&runner), 0)
      .emplace(temporal_id, 1);
  CHECK(ActionTesting::is_ready<component>(runner, 0));
  ActionTesting::next_action<component>(make_not_null(&runner), 0);
}
