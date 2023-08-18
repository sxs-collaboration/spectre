// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <pup.h>
#include <string>
#include <tuple>
#include <vector>

#include "DataStructures/DataBox/DataBox.hpp"
#include "Evolution/Systems/Cce/Events/ObserveTimeStep.hpp"
#include "Framework/ActionTesting.hpp"
#include "Framework/TestHelpers.hpp"
#include "IO/Observer/ObserverComponent.hpp"
#include "Parallel/GlobalCache.hpp"
#include "Parallel/Phase.hpp"
#include "ParallelAlgorithms/EventsAndTriggers/Event.hpp"
#include "Time/Slab.hpp"
#include "Time/Time.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

namespace Cce {
namespace {
using ObserveTimeStep = Events::ObserveTimeStep;
constexpr double time = 1.3;
constexpr double step = 0.27;
static std::string filename = "Strawberry";

struct MockWriteReductionDataRow {
  template <typename ParallelComponent, typename DbTagsList,
            typename Metavariables, typename ArrayIndex>
  static void apply(db::DataBox<DbTagsList>& /*box*/,
                    const Parallel::GlobalCache<Metavariables>& /*cache*/,
                    const ArrayIndex& /*array_index*/,
                    const gsl::not_null<Parallel::NodeLock*> /*node_lock*/,
                    const std::string& subfile_name,
                    const std::vector<std::string>& file_legend,
                    const std::tuple<std::vector<double>>& data_row) {
    CHECK(subfile_name == "/Cce/" + filename);
    CHECK(file_legend == std::vector<std::string>{"Time", "Time Step"});
    CHECK(std::get<0>(data_row) == std::vector<double>{time, step});
  }
};

template <typename Metavariables>
struct MockObserverWriter {
  using metavariables = Metavariables;
  using chare_type = ActionTesting::MockNodeGroupChare;
  using array_index = int;
  using phase_dependent_action_list = tmpl::list<Parallel::PhaseActions<
      Parallel::Phase::Initialization,
      tmpl::list<ActionTesting::InitializeDataBox<tmpl::list<>>>>>;
  using component_being_mocked = observers::ObserverWriter<Metavariables>;

  using replace_these_threaded_actions =
      tmpl::list<observers::ThreadedActions::WriteReductionDataRow>;
  using with_these_threaded_actions = tmpl::list<MockWriteReductionDataRow>;
};

struct Metavars {
  void pup(PUP::er& /*p*/) {}
  using observed_reduction_data_tags = tmpl::list<>;
  using component_list = tmpl::list<MockObserverWriter<Metavars>>;
};

void test() {
  using metavars = Metavars;
  using obs_writer = MockObserverWriter<metavars>;

  // No need to print things to terminal here
  ObserveTimeStep observe_time_step{filename, false};
  ObserveTimeStep serialized_observe_time_step =
      serialize_and_deserialize(observe_time_step);

  Slab slab{time, time + step};
  TimeDelta time_delta{slab, 1};

  using MockRuntimeSystem = ActionTesting::MockRuntimeSystem<metavars>;
  MockRuntimeSystem runner{{}};
  ActionTesting::emplace_nodegroup_component_and_initialize<obs_writer>(
      make_not_null(&runner), {});

  auto& cache = ActionTesting::cache<obs_writer>(runner, 0);
  const int array_index = 0;
  const obs_writer* const component = nullptr;
  const Event::ObservationValue observation_value{"ObservationValue", time};

  serialized_observe_time_step(time_delta, cache, array_index, component,
                               observation_value);

  CHECK(ActionTesting::number_of_queued_threaded_actions<obs_writer>(runner,
                                                                     0) == 1);
  ActionTesting::invoke_queued_threaded_action<obs_writer>(
      make_not_null(&runner), 0);
}

SPECTRE_TEST_CASE("Unit.Evolution.Systems.Cce.Events.ObserveTimeStep",
                  "[Unit][Evolution]") {
  test();
}
}  // namespace
}  // namespace Cce
