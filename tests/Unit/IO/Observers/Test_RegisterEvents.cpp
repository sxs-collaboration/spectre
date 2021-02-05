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
#include "Parallel/ArrayIndex.hpp"
#include "Parallel/Invoke.hpp"
#include "Parallel/RegisterDerivedClassesWithCharm.hpp"
#include "ParallelAlgorithms/EventsAndTriggers/Event.hpp"
#include "ParallelAlgorithms/EventsAndTriggers/EventsAndTriggers.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Registration.hpp"
#include "Utilities/TaggedTuple.hpp"

namespace {
template <typename EventRegistrars>
class SomeEvent : public Event<EventRegistrars> {
 public:
  struct SubfileName {
    using type = std::string;
    static constexpr Options::String help = {
        "The The name of the subfile inside the HDF5 file"};
  };

  static constexpr Options::String help = "halp";

  using options = tmpl::list<SubfileName>;

  SomeEvent() = default;
  explicit SomeEvent(std::string subfile_path) noexcept
      : subfile_path_(std::move(subfile_path)) {}

  /// \cond
  explicit SomeEvent(CkMigrateMessage* /*unused*/) noexcept {}
  using PUP::able::register_constructor;
  WRAPPED_PUPable_decl_template(SomeEvent);  // NOLINT
  /// \endcond

  using observation_registration_tags = tmpl::list<>;
  std::pair<observers::TypeOfObservation, observers::ObservationKey>
  get_observation_type_and_key_for_registration() const noexcept {
    return {observers::TypeOfObservation::Reduction,
            observers::ObservationKey(subfile_path_ + ".dat")};
  }

  // NOLINTNEXTLINE(google-runtime-references)
  void pup(PUP::er& p) override {
    Event<EventRegistrars>::pup(p);
    p | subfile_path_;
  }

 private:
  std::string subfile_path_;
};

template <typename EventRegistrars>
PUP::able::PUP_ID SomeEvent<EventRegistrars>::my_PUP_ID = 0;  // NOLINT

namespace Registrars {
using SomeEvent = ::Registration::Registrar<SomeEvent>;
}  // namespace Registrars

using events_and_triggers_tag =
    Tags::EventsAndTriggers<tmpl::list<Registrars::SomeEvent>, tmpl::list<>>;

template <typename Metavariables>
struct Component {
  using metavariables = Metavariables;
  using component_being_mocked = void;
  using chare_type = ActionTesting::MockArrayChare;
  using array_index = int;
  using const_global_cache_tags = tmpl::list<events_and_triggers_tag>;
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
  static void apply(
      db::DataBox<DbTagList>& /*box*/,
      Parallel::GlobalCache<Metavariables>& /*cache*/,
      const ArrayIndex& /*array_index*/,
      const observers::ObservationKey& observation_key,
      const observers::ArrayComponentId& component_id,
      const observers::TypeOfObservation& type_of_observation) noexcept {
    result.observation_key = observation_key;
    result.array_component_id = component_id;
    result.type_of_observation = type_of_observation;
  }
};

MockRegisterContributorWithObserver::Result
    MockRegisterContributorWithObserver::result{};

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
      tmpl::list<observers::Actions::RegisterContributorWithObserver>;
  using with_these_simple_actions =
      tmpl::list<MockRegisterContributorWithObserver>;
};

struct Metavariables {
  using component_list = tmpl::list<Component<Metavariables>,
                                    MockObserverComponent<Metavariables>>;
  enum class Phase { Initialization, Testing, Exit };
};

using EventsAndTriggersType =
    EventsAndTriggers<tmpl::list<Registrars::SomeEvent>, tmpl::list<>>;

SPECTRE_TEST_CASE("Unit.IO.Observers.RegisterEvents", "[Unit][Observers]") {
  // Test pup
  Parallel::register_derived_classes_with_charm<
      Event<tmpl::list<Registrars::SomeEvent>>>();
  Parallel::register_derived_classes_with_charm<Trigger<tmpl::list<>>>();

  const auto events_and_triggers =
      TestHelpers::test_creation<EventsAndTriggersType>(
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
}
}  // namespace
