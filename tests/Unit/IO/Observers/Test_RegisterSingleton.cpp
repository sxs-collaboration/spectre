// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cstddef>
#include <utility>

#include "Framework/ActionTesting.hpp"
#include "Helpers/IO/Observers/ObserverHelpers.hpp"
#include "IO/Observer/Actions/ObserverRegistration.hpp"
#include "IO/Observer/Actions/RegisterSingleton.hpp"
#include "IO/Observer/ArrayComponentId.hpp"
#include "IO/Observer/Initialize.hpp"
#include "IO/Observer/ObservationId.hpp"
#include "IO/Observer/ObserverComponent.hpp"
#include "IO/Observer/Tags.hpp"
#include "IO/Observer/TypeOfObservation.hpp"
#include "Parallel/ArrayIndex.hpp"
#include "Parallel/Invoke.hpp"

namespace {
struct RegistrationHelper {
  template <typename ParallelComponent, typename DbTagsList,
            typename ArrayIndex>
  static std::pair<observers::TypeOfObservation, observers::ObservationKey>
  register_info(const db::DataBox<DbTagsList>& /*box*/,
                const ArrayIndex& /*array_index*/) noexcept {
    return {observers::TypeOfObservation::Reduction,
            observers::ObservationKey{"Singleton"}};
  }
};

template <typename Metavariables>
struct Component {
  using metavariables = Metavariables;
  using component_being_mocked = void;
  using chare_type = ActionTesting::MockArrayChare;
  using array_index = int;
  using const_global_cache_tags = tmpl::list<>;
  using phase_dependent_action_list = tmpl::list<Parallel::PhaseActions<
      typename Metavariables::Phase, Metavariables::Phase::Testing,
      tmpl::list<observers::Actions::RegisterSingletonWithObserverWriter<
          RegistrationHelper>>>>;
};

struct MockRegisterReductionContributorWithObserverWriter {
  struct Result {
    observers::ObservationKey observation_key{};
    size_t caller_node_id{};
  };
  static Result result;

  template <typename ParallelComponent, typename DbTagList,
            typename Metavariables, typename ArrayIndex>
  static void apply(db::DataBox<DbTagList>& /*box*/,
                    Parallel::GlobalCache<Metavariables>& /*cache*/,
                    const ArrayIndex& /*array_index*/,
                    const observers::ObservationKey& observation_key,
                    const size_t caller_node_id) noexcept {
    result.observation_key = observation_key;
    result.caller_node_id = caller_node_id;
  }
};

MockRegisterReductionContributorWithObserverWriter::Result
    MockRegisterReductionContributorWithObserverWriter::result{};

template <typename Metavariables>
struct MockObserverWriterComponent {
  using metavariables = Metavariables;
  using chare_type = ActionTesting::MockArrayChare;
  using array_index = int;
  using const_global_cache_tags = tmpl::list<>;
  using phase_dependent_action_list = tmpl::list<
      Parallel::PhaseActions<typename Metavariables::Phase,
                             Metavariables::Phase::Testing, tmpl::list<>>>;

  using component_being_mocked = observers::ObserverWriter<Metavariables>;
  using replace_these_simple_actions =
      tmpl::list<observers::Actions::RegisterReductionNodeWithWritingNode>;
  using with_these_simple_actions =
      tmpl::list<MockRegisterReductionContributorWithObserverWriter>;
};

struct Metavariables {
  using component_list = tmpl::list<Component<Metavariables>,
                                    MockObserverWriterComponent<Metavariables>>;
  enum class Phase { Initialization, Testing, Exit };
};

SPECTRE_TEST_CASE("Unit.IO.Observers.RegisterSingleton", "[Unit][Observers]") {
  using my_component = Component<Metavariables>;
  using obs_component = MockObserverWriterComponent<Metavariables>;
  ActionTesting::MockRuntimeSystem<Metavariables> runner{{}};
  ActionTesting::emplace_component<my_component>(&runner, 0);
  ActionTesting::emplace_component<my_component>(&runner, 1);
  ActionTesting::emplace_component<obs_component>(&runner, 0);
  ActionTesting::set_phase(make_not_null(&runner),
                           Metavariables::Phase::Testing);

  ActionTesting::next_action<my_component>(make_not_null(&runner), 0);
  REQUIRE(not ActionTesting::is_simple_action_queue_empty<obs_component>(runner,
                                                                         0));
  ActionTesting::invoke_queued_simple_action<obs_component>(
      make_not_null(&runner), 0);
  REQUIRE(
      ActionTesting::is_simple_action_queue_empty<obs_component>(runner, 0));

  CHECK(MockRegisterReductionContributorWithObserverWriter::result
            .observation_key == observers::ObservationKey{"Singleton"});
  CHECK(MockRegisterReductionContributorWithObserverWriter::result
            .caller_node_id == 0);
}
}  // namespace
