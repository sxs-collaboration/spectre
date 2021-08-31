// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <string>
#include <utility>
#include <vector>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/LinkedMessageQueue.hpp"
#include "Framework/ActionTesting.hpp"
#include "Parallel/PhaseDependentActionList.hpp"
#include "ParallelAlgorithms/Actions/UpdateMessageQueue.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

namespace Parallel {
template <typename Metavaiables>
class GlobalCache;
}  // namespace Parallel

namespace {
struct Queue1 {
  using type = double;
};

struct Queue2 {
  using type = double;
};

struct LinkedMessageQueueTag : db::SimpleTag {
  using type = LinkedMessageQueue<int, tmpl::list<Queue1, Queue2>>;
};

struct ProcessorCalls : db::SimpleTag {
  using type = std::vector<std::pair<int, tuples::TaggedTuple<Queue1, Queue2>>>;
};

struct Processor {
// [Processor::apply]
  template <typename DbTags, typename Metavariables, typename ArrayIndex>
  static void apply(const gsl::not_null<db::DataBox<DbTags>*> box,
                    Parallel::GlobalCache<Metavariables>& /*cache*/,
                    const ArrayIndex& /*array_index*/, const int id,
                    tuples::TaggedTuple<Queue1, Queue2> data) noexcept {
// [Processor::apply]
    db::mutate<ProcessorCalls>(
        box, [&id, &data](
                 const gsl::not_null<ProcessorCalls::type*> calls) noexcept {
          calls->emplace_back(id, std::move(data));
        });
  }
};

template <typename Metavariables>
struct Component {
  using metavariables = Metavariables;
  using chare_type = ActionTesting::MockArrayChare;
  using array_index = int;
  using initialization_tags = tmpl::list<LinkedMessageQueueTag, ProcessorCalls>;
  using phase_dependent_action_list =
      tmpl::list<Parallel::PhaseActions<typename Metavariables::Phase,
                                        Metavariables::Phase::Initialization,
                                        tmpl::list<>>>;
};

struct Metavariables {
  using component_list = tmpl::list<Component<Metavariables>>;
  enum class Phase { Initialization, Testing, Exit };
};
}  // namespace

SPECTRE_TEST_CASE("Unit.Actions.UpdateMessageQueue", "[Unit][Actions]") {
  using component = Component<Metavariables>;

  ActionTesting::MockRuntimeSystem<Metavariables> runner{{}};
  ActionTesting::emplace_component<component>(
      &runner, 0, LinkedMessageQueueTag::type{}, ProcessorCalls::type{});

  ActionTesting::set_phase(make_not_null(&runner),
                           Metavariables::Phase::Testing);

  const auto processed_by_call =
      [&runner](auto queue_v, const LinkedMessageId<int>& id,
                auto data) noexcept -> decltype(auto) {
    ActionTesting::simple_action<
        component, Actions::UpdateMessageQueue<
                       decltype(queue_v), LinkedMessageQueueTag, Processor>>(
        make_not_null(&runner), 0, id, std::move(data));
    return db::mutate<ProcessorCalls>(
        make_not_null(&ActionTesting::get_databox<component, tmpl::list<>>(
            make_not_null(&runner), 0)),
        [](const gsl::not_null<ProcessorCalls::type*> calls) noexcept {
          auto ret = std::move(*calls);
          calls->clear();
          return ret;
        });
  };

  CHECK(processed_by_call(Queue1{}, {0, {}}, 1.23).empty());
  {
    const auto processed = processed_by_call(Queue2{}, {0, {}}, 2.34);
    CHECK(processed.size() == 1);

    CHECK(processed[0].first == 0);
    CHECK(get<Queue1>(processed[0].second) == 1.23);
    CHECK(get<Queue2>(processed[0].second) == 2.34);
  }
  CHECK(processed_by_call(Queue1{}, {2, 1}, 2.2).empty());
  CHECK(processed_by_call(Queue2{}, {1, 0}, 1.1).empty());
  CHECK(processed_by_call(Queue2{}, {2, 1}, 2.2).empty());
  {
    const auto processed = processed_by_call(Queue1{}, {1, 0}, 1.1);
    CHECK(processed.size() == 2);

    CHECK(processed[0].first == 1);
    CHECK(get<Queue1>(processed[0].second) == 1.1);
    CHECK(get<Queue2>(processed[0].second) == 1.1);

    CHECK(processed[1].first == 2);
    CHECK(get<Queue1>(processed[1].second) == 2.2);
    CHECK(get<Queue2>(processed[1].second) == 2.2);
  }
}
