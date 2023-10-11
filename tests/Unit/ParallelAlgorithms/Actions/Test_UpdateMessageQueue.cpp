// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/LinkedMessageId.hpp"
#include "DataStructures/LinkedMessageQueue.hpp"
#include "Domain/Creators/Tags/FunctionsOfTime.hpp"
#include "Domain/FunctionsOfTime/FunctionOfTime.hpp"
#include "Domain/FunctionsOfTime/PiecewisePolynomial.hpp"
#include "Domain/FunctionsOfTime/RegisterDerivedWithCharm.hpp"
#include "Domain/FunctionsOfTime/Tags.hpp"
#include "Framework/ActionTesting.hpp"
#include "Parallel/GlobalCache.hpp"
#include "Parallel/Phase.hpp"
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
                    tuples::TaggedTuple<Queue1, Queue2> data) {
    // [Processor::apply]
    db::mutate<ProcessorCalls>(
        [&id, &data](const gsl::not_null<ProcessorCalls::type*> calls) {
          calls->emplace_back(id, std::move(data));
        },
        box);
  }
};

template <typename Metavariables>
struct Component {
  using metavariables = Metavariables;
  using chare_type = ActionTesting::MockArrayChare;
  using array_index = int;
  using simple_tags_from_options =
      tmpl::list<LinkedMessageQueueTag, ProcessorCalls>;
  using mutable_global_cache_tags =
      tmpl::list<domain::Tags::FunctionsOfTimeInitialize>;
  using phase_dependent_action_list = tmpl::list<
      Parallel::PhaseActions<Parallel::Phase::Initialization, tmpl::list<>>>;
};

struct Metavariables {
  using component_list = tmpl::list<Component<Metavariables>>;
};

using FunctionMap = domain::Tags::FunctionsOfTimeInitialize::type;
struct UpdateFoT {
  static void apply(const gsl::not_null<FunctionMap*> functions,
                    const std::string& name, const double expiration) {
    const double current_expiration = functions->at(name)->time_bounds()[1];
    // Update value doesn't matter
    (*functions)
        .at(name)
        ->update(current_expiration, DataVector{0.0}, expiration);
  }
};
}  // namespace

SPECTRE_TEST_CASE("Unit.Actions.UpdateMessageQueue", "[Unit][Actions]") {
  using component = Component<Metavariables>;
  domain::FunctionsOfTime::register_derived_with_charm();

  FunctionMap functions_of_time{};
  const std::string name{"Smaug"};
  functions_of_time[name] =
      std::make_unique<domain::FunctionsOfTime::PiecewisePolynomial<0>>(
          0.0, std::array{DataVector{0.0}}, 1.0);

  ActionTesting::MockRuntimeSystem<Metavariables> runner{
      {}, {std::move(functions_of_time)}};
  ActionTesting::emplace_component<component>(
      &runner, 0, LinkedMessageQueueTag::type{}, ProcessorCalls::type{});

  ActionTesting::set_phase(make_not_null(&runner), Parallel::Phase::Testing);

  auto& cache = ActionTesting::cache<component>(runner, 0);

  const auto processed_by_call = [&runner](auto queue_v,
                                           const LinkedMessageId<int>& id,
                                           auto data) -> decltype(auto) {
    ActionTesting::simple_action<
        component, Actions::UpdateMessageQueue<
                       decltype(queue_v), LinkedMessageQueueTag, Processor>>(
        make_not_null(&runner), 0, id, std::move(data));
    return db::mutate<ProcessorCalls>(
        [](const gsl::not_null<ProcessorCalls::type*> calls) {
          auto ret = std::move(*calls);
          calls->clear();
          return ret;
        },
        make_not_null(
            &ActionTesting::get_databox<component>(make_not_null(&runner), 0)));
  };

  CHECK(processed_by_call(Queue1{}, {0, {}}, 1.23).empty());
  {
    const auto processed = processed_by_call(Queue2{}, {0, {}}, 2.34);
    CHECK(processed.size() == 1);

    CHECK(processed[0].first == 0);
    CHECK(get<Queue1>(processed[0].second) == 1.23);
    CHECK(get<Queue2>(processed[0].second) == 2.34);
  }
  // Nothing should have been inserted because 2 is after expiration of 1.
  CHECK(processed_by_call(Queue1{}, {2, 1}, 2.2).empty());
  Parallel::mutate<domain::Tags::FunctionsOfTime, UpdateFoT>(cache, name, 5.0);
  CHECK(ActionTesting::number_of_queued_simple_actions<component>(runner, 0) ==
        1);
  // Now things should have been inserted
  ActionTesting::invoke_queued_simple_action<component>(make_not_null(&runner),
                                                        0);
  CHECK(ActionTesting::get_databox_tag<component, ProcessorCalls>(runner, 0)
            .empty());
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
