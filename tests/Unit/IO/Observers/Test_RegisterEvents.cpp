// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cstddef>
#include <string>
#include <type_traits>
#include <utility>

#include "Framework/ActionTesting.hpp"
#include "Framework/TestCreation.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/IO/Observers/ObserverHelpers.hpp"
#include "IO/Observer/Actions/ObserverRegistration.hpp"
#include "IO/Observer/Actions/RegisterEvents.hpp"
#include "IO/Observer/ArrayComponentId.hpp"
#include "IO/Observer/Initialize.hpp"
#include "IO/Observer/ObservationId.hpp"
#include "IO/Observer/ObserverComponent.hpp"
#include "IO/Observer/Tags.hpp"
#include "IO/Observer/TypeOfObservation.hpp"
#include "Options/Protocols/FactoryCreation.hpp"
#include "Parallel/ArrayIndex.hpp"
#include "Parallel/Invoke.hpp"
#include "Parallel/RegisterDerivedClassesWithCharm.hpp"
#include "ParallelAlgorithms/EventsAndTriggers/Event.hpp"
#include "ParallelAlgorithms/EventsAndTriggers/EventsAndTriggers.hpp"
#include "ParallelAlgorithms/EventsAndTriggers/LogicalTriggers.hpp"
#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/ProtocolHelpers.hpp"
#include "Utilities/TaggedTuple.hpp"

namespace {
class SomeEvent : public Event {
 public:
  struct SubfileName {
    using type = std::string;
    static constexpr Options::String help = {
        "The The name of the subfile inside the HDF5 file"};
  };

  static constexpr Options::String help = "halp";

  using options = tmpl::list<SubfileName>;

  SomeEvent() = default;
  explicit SomeEvent(std::string subfile_path)
      : subfile_path_(std::move(subfile_path)) {}

  explicit SomeEvent(CkMigrateMessage* /*unused*/) {}
  using PUP::able::register_constructor;
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-function"
  WRAPPED_PUPable_decl_template(SomeEvent);  // NOLINT
#pragma GCC diagnostic pop

  using observation_registration_tags = tmpl::list<>;

  template <typename Metavariables, typename ArrayIndex,
            typename ParallelComponent>
  std::pair<observers::TypeOfObservation, observers::ObservationKey>
  get_observation_type_and_key_for_registration(
      const Parallel::GlobalCache<Metavariables>& /*cache*/,
      const ArrayIndex& /*array_index*/,
      const ParallelComponent* const /*meta*/) const {
    return {observers::TypeOfObservation::Reduction,
            observers::ObservationKey(subfile_path_ + ".dat")};
  }

  bool needs_evolved_variables() const override {
    ERROR("Should not be called");
  }

  // NOLINTNEXTLINE(google-runtime-references)
  void pup(PUP::er& p) override {
    Event::pup(p);
    p | subfile_path_;
  }

 private:
  std::string subfile_path_;
};

PUP::able::PUP_ID SomeEvent::my_PUP_ID = 0;  // NOLINT

template <typename Metavariables>
struct Component {
  using metavariables = Metavariables;
  using component_being_mocked = void;
  using chare_type = ActionTesting::MockArrayChare;
  using array_index = int;
  using const_global_cache_tags = tmpl::list<Tags::EventsAndTriggers>;
  using phase_dependent_action_list = tmpl::list<Parallel::PhaseActions<
      typename Metavariables::Phase, Metavariables::Phase::Testing,
      tmpl::list<observers::Actions::RegisterEventsWithObservers>>>;
};

struct MockRegisterContributorWithObserver {
  struct Result {
    observers::ObservationKey observation_key{};
    observers::ArrayComponentId array_component_id{};
    observers::TypeOfObservation type_of_observation{};
  };
  static Result result;

  template <typename ParallelComponent, typename DbTagList,
            typename Metavariables, typename ArrayIndex>
  static void apply(db::DataBox<DbTagList>& /*box*/,
                    Parallel::GlobalCache<Metavariables>& /*cache*/,
                    const ArrayIndex& /*array_index*/,
                    const observers::ObservationKey& observation_key,
                    const observers::ArrayComponentId& component_id,
                    const observers::TypeOfObservation& type_of_observation) {
    result.observation_key = observation_key;
    result.array_component_id = component_id;
    result.type_of_observation = type_of_observation;
  }
};

struct MockDeregisterContributorWithObserver {
  struct Result {
    observers::ObservationKey observation_key{};
    observers::ArrayComponentId array_component_id{};
    observers::TypeOfObservation type_of_observation{};
  };
  static Result result;

  template <typename ParallelComponent, typename DbTagList,
            typename Metavariables, typename ArrayIndex>
  static void apply(db::DataBox<DbTagList>& /*box*/,
                    Parallel::GlobalCache<Metavariables>& /*cache*/,
                    const ArrayIndex& /*array_index*/,
                    const observers::ObservationKey& observation_key,
                    const observers::ArrayComponentId& component_id,
                    const observers::TypeOfObservation& type_of_observation) {
    result.observation_key = observation_key;
    result.array_component_id = component_id;
    result.type_of_observation = type_of_observation;
  }
};

MockRegisterContributorWithObserver::Result
    MockRegisterContributorWithObserver::result{};

MockDeregisterContributorWithObserver::Result
    MockDeregisterContributorWithObserver::result{};

template <typename Metavariables>
struct MockObserverComponent {
  using metavariables = Metavariables;
  using chare_type = ActionTesting::MockGroupChare;
  using array_index = int;
  using const_global_cache_tags = tmpl::list<>;
  using phase_dependent_action_list = tmpl::list<
      Parallel::PhaseActions<typename Metavariables::Phase,
                             Metavariables::Phase::Testing, tmpl::list<>>>;

  using component_being_mocked = observers::Observer<Metavariables>;
  using replace_these_simple_actions =
      tmpl::list<observers::Actions::RegisterContributorWithObserver,
                 observers::Actions::DeregisterContributorWithObserver>;
  using with_these_simple_actions =
      tmpl::list<MockRegisterContributorWithObserver,
                 MockDeregisterContributorWithObserver>;
};

struct Metavariables {
  using component_list = tmpl::list<Component<Metavariables>,
                                    MockObserverComponent<Metavariables>>;
  struct factory_creation
      : tt::ConformsTo<Options::protocols::FactoryCreation> {
    using factory_classes =
        tmpl::map<tmpl::pair<Event, tmpl::list<SomeEvent>>,
                  tmpl::pair<Trigger, Triggers::logical_triggers>>;
  };
  enum class Phase { Initialization, Testing, Exit };
};

SPECTRE_TEST_CASE("Unit.IO.Observers.RegisterEvents", "[Unit][Observers]") {
  // Test pup
  Parallel::register_factory_classes_with_charm<Metavariables>();

  const auto events_and_triggers =
      TestHelpers::test_creation<EventsAndTriggers, Metavariables>(
          "? Not: Always\n"
          ": - SomeEvent:\n"
          "      SubfileName: element_data\n");

  using my_component = Component<Metavariables>;
  using obs_component = MockObserverComponent<Metavariables>;
  ActionTesting::MockRuntimeSystem<Metavariables> runner{
      {serialize_and_deserialize(events_and_triggers)}};
  ActionTesting::emplace_component<my_component>(&runner, 0);
  ActionTesting::emplace_component<my_component>(&runner, 1);
  ActionTesting::emplace_group_component<obs_component>(&runner);
  ActionTesting::set_phase(make_not_null(&runner),
                           Metavariables::Phase::Testing);

  ActionTesting::next_action<my_component>(make_not_null(&runner), 0);
  ActionTesting::next_action<my_component>(make_not_null(&runner), 1);
  for (size_t i = 0; i < 2; ++i) {
    CAPTURE(i);
    REQUIRE(not ActionTesting::is_simple_action_queue_empty<obs_component>(
        runner, 0));
    ActionTesting::invoke_queued_simple_action<obs_component>(
        make_not_null(&runner), 0);

    CHECK(MockRegisterContributorWithObserver::result.observation_key ==
          observers::ObservationKey("element_data.dat"));
    // Need an `or` because we don't know what order actions are run in
    CHECK((MockRegisterContributorWithObserver::result.array_component_id ==
               observers::ArrayComponentId(
                   std::add_pointer_t<my_component>{nullptr},
                   Parallel::ArrayIndex<int>{0}) or
           MockRegisterContributorWithObserver::result.array_component_id ==
               observers::ArrayComponentId(
                   std::add_pointer_t<my_component>{nullptr},
                   Parallel::ArrayIndex<int>{1})));
    CHECK(MockRegisterContributorWithObserver::result.type_of_observation ==
          observers::TypeOfObservation::Reduction);
  }
  {
    INFO("Deregistration");
    // call the deregistration functions for each element
    // note that these are not actions, because they are intended to be called
    // from pup functions.
    for (int i = 0; i < 2; ++i) {
      observers::Actions::RegisterEventsWithObservers::
          template perform_deregistration<my_component>(
              ActionTesting::get_databox<my_component, tmpl::list<>>(
                  make_not_null(&runner), i),
              ActionTesting::cache<my_component>(runner, i), i);
      ActionTesting::invoke_queued_simple_action<obs_component>(
          make_not_null(&runner), 0);

      // The deregistration mock action just records its arguments like the
      // registration mock action. The actual deregistration procedure is tested
      // by Unit.IO.Observers.RegisterElements
      CHECK(MockDeregisterContributorWithObserver::result.observation_key ==
            observers::ObservationKey("element_data.dat"));
      // Need an `or` because we don't know what order actions are run in
      CHECK((MockDeregisterContributorWithObserver::result.array_component_id ==
                 observers::ArrayComponentId(
                     std::add_pointer_t<my_component>{nullptr},
                     Parallel::ArrayIndex<int>{0}) or
             MockDeregisterContributorWithObserver::result.array_component_id ==
                 observers::ArrayComponentId(
                     std::add_pointer_t<my_component>{nullptr},
                     Parallel::ArrayIndex<int>{1})));
      CHECK(MockDeregisterContributorWithObserver::result.type_of_observation ==
            observers::TypeOfObservation::Reduction);
    }
  }
}
}  // namespace
