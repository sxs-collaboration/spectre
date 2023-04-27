// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <cstddef>
#include <memory>
#include <optional>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <utility>

#include "ControlSystem/Actions/InitializeMeasurements.hpp"
#include "ControlSystem/Event.hpp"
#include "ControlSystem/Protocols/ControlSystem.hpp"
#include "ControlSystem/Protocols/Measurement.hpp"
#include "ControlSystem/Protocols/Submeasurement.hpp"
#include "ControlSystem/Tags/SystemTags.hpp"
#include "ControlSystem/Trigger.hpp"
#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataVector.hpp"
#include "Domain/Creators/Tags/FunctionsOfTime.hpp"
#include "Domain/FunctionsOfTime/FunctionOfTime.hpp"
#include "Domain/FunctionsOfTime/PiecewisePolynomial.hpp"
#include "Domain/FunctionsOfTime/Tags.hpp"
#include "Evolution/Actions/RunEventsAndDenseTriggers.hpp"
#include "Evolution/EventsAndDenseTriggers/DenseTrigger.hpp"
#include "Evolution/EventsAndDenseTriggers/EventsAndDenseTriggers.hpp"
#include "Evolution/EventsAndDenseTriggers/Tags.hpp"
#include "Framework/ActionTesting.hpp"
#include "Framework/MockDistributedObject.hpp"
#include "Framework/MockRuntimeSystem.hpp"
#include "Framework/MockRuntimeSystemFreeFunctions.hpp"
#include "Helpers/ControlSystem/TestStructs.hpp"
#include "IO/Logging/Verbosity.hpp"
#include "Options/Protocols/FactoryCreation.hpp"
#include "Parallel/Phase.hpp"
#include "Parallel/PhaseDependentActionList.hpp"
#include "ParallelAlgorithms/EventsAndTriggers/Event.hpp"
#include "Time/Tags.hpp"
#include "Utilities/GetOutput.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/ProtocolHelpers.hpp"
#include "Utilities/Serialization/RegisterDerivedClassesWithCharm.hpp"
#include "Utilities/TMPL.hpp"

template <typename Id>
struct LinkedMessageId;
namespace Parallel {
template <typename Metavariables>
class GlobalCache;
}  // namespace Parallel

namespace {
template <typename ExpectedSystems>
struct Submeasurement
    : tt::ConformsTo<control_system::protocols::Submeasurement> {
  template <typename ControlSystems>
  using interpolation_target_tag = void;

  using argument_tags = tmpl::list<>;

  template <typename Metavariables, typename ParallelComponent,
            typename ControlSystems>
  static void apply(const LinkedMessageId<double>& /*measurement_id*/,
                    Parallel::GlobalCache<Metavariables>& /*cache*/,
                    const int& /*array_index*/,
                    const ParallelComponent* /*meta*/,
                    ControlSystems /*meta*/) {
    // Test cases have either one or two systems, so this covers all
    // permutations.
    static_assert(
        std::is_same_v<ControlSystems, ExpectedSystems> or
        std::is_same_v<ControlSystems, tmpl::reverse<ExpectedSystems>>);
    ++call_count;
  }

  static size_t call_count;
};

template <typename ExpectedSystems>
size_t Submeasurement<ExpectedSystems>::call_count = 0;

template <typename ExpectedSystems>
struct Measurement : tt::ConformsTo<control_system::protocols::Measurement> {
  using submeasurements = tmpl::list<Submeasurement<ExpectedSystems>>;
};

struct SystemB;

struct SystemA : tt::ConformsTo<control_system::protocols::ControlSystem> {
  static std::string name() { return "A"; }
  static std::optional<std::string> component_name(
      const size_t i, const size_t /*num_components*/) {
    return get_output(i);
  }
  using simple_tags = tmpl::list<>;
  using measurement = Measurement<tmpl::list<SystemA, SystemB>>;
  using control_error = control_system::TestHelpers::ControlError<1>;
  struct process_measurement {
    template <typename Submeasurement>
    using argument_tags = tmpl::list<>;
  };
};

struct SystemB : tt::ConformsTo<control_system::protocols::ControlSystem> {
  static std::string name() { return "B"; }
  static std::optional<std::string> component_name(
      const size_t /*i*/, const size_t /*num_components*/) {
    return std::nullopt;
  }
  using simple_tags = tmpl::list<>;
  using measurement = Measurement<tmpl::list<SystemA, SystemB>>;
  using control_error = control_system::TestHelpers::ControlError<1>;
  struct process_measurement {
    template <typename Submeasurement>
    using argument_tags = tmpl::list<>;
  };
};

struct SystemC : tt::ConformsTo<control_system::protocols::ControlSystem> {
  static std::string name() { return "C"; }
  static std::optional<std::string> component_name(
      const size_t /*i*/, const size_t /*num_components*/) {
    return std::nullopt;
  }
  using simple_tags = tmpl::list<>;
  using measurement = Measurement<tmpl::list<SystemC>>;
  using control_error = control_system::TestHelpers::ControlError<1>;
  struct process_measurement {
    template <typename Submeasurement>
    using argument_tags = tmpl::list<>;
  };
};

static_assert(tt::assert_conforms_to_v<
              SystemA, control_system::protocols::ControlSystem>);
static_assert(tt::assert_conforms_to_v<
              SystemB, control_system::protocols::ControlSystem>);
static_assert(tt::assert_conforms_to_v<
              SystemC, control_system::protocols::ControlSystem>);

using control_systems = tmpl::list<SystemA, SystemB, SystemC>;

template <typename Metavariables>
struct Component {
  using metavariables = Metavariables;
  using chare_type = ActionTesting::MockArrayChare;
  using array_index = int;

  using simple_tags_from_options =
      tmpl::list<evolution::Tags::EventsAndDenseTriggers>;

  using simple_tags = tmpl::list<Tags::TimeStepId, Tags::Time>;
  using compute_tags = tmpl::list<Parallel::Tags::FromGlobalCache<
      ::domain::Tags::FunctionsOfTimeInitialize>>;
  using const_global_cache_tags = tmpl::list<control_system::Tags::Verbosity>;
  using mutable_global_cache_tags =
      tmpl::list<domain::Tags::FunctionsOfTimeInitialize>;

  using phase_dependent_action_list = tmpl::list<Parallel::PhaseActions<
      Parallel::Phase::Initialization,
      tmpl::list<
          ActionTesting::InitializeDataBox<simple_tags, compute_tags>,
          evolution::Actions::InitializeRunEventsAndDenseTriggers,
          control_system::Actions::InitializeMeasurements<control_systems>>>>;
};

struct Metavariables {
  using component_list = tmpl::list<Component<Metavariables>>;

  struct factory_creation
      : tt::ConformsTo<Options::protocols::FactoryCreation> {
    using factory_classes = tmpl::map<
        tmpl::pair<Event,
                   control_system::control_system_events<control_systems>>,
        tmpl::pair<DenseTrigger,
                   control_system::control_system_triggers<control_systems>>>;
  };
};

SPECTRE_TEST_CASE("Unit.ControlSystem.InitializeMeasurements",
                  "[ControlSystem][Unit]") {
  register_factory_classes_with_charm<Metavariables>();
  register_classes_with_charm<
      domain::FunctionsOfTime::PiecewisePolynomial<0>>();

  using MockRuntimeSystem = ActionTesting::MockRuntimeSystem<Metavariables>;
  using component = Component<Metavariables>;
  const component* const component_p = nullptr;

  // Details shouldn't matter
  const auto timescale =
      std::make_unique<domain::FunctionsOfTime::PiecewisePolynomial<0>>(
          0.0, std::array{DataVector{1.0}}, 2.0);
  std::unordered_map<std::string,
                     std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>
      timescales;
  timescales.emplace("A", timescale->get_clone());
  timescales.emplace("B", timescale->get_clone());
  timescales.emplace("C", timescale->get_clone());
  std::unordered_map<std::string,
                     std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>
      functions;
  functions.emplace("A", timescale->get_clone());
  functions.emplace("B", timescale->get_clone());
  functions.emplace("C", timescale->get_clone());

  MockRuntimeSystem runner{{::Verbosity::Silent},
                           {std::move(functions), std::move(timescales)}};
  ActionTesting::emplace_array_component_and_initialize<component>(
      make_not_null(&runner), ActionTesting::NodeId{0},
      ActionTesting::LocalCoreId{0}, 0,
      {Tags::TimeStepId::type{true, 0, {}}, Tags::Time::type{0.0}},
      evolution::Tags::EventsAndDenseTriggers::type{});

  // InitializeRunEventsAndDenseTriggers
  ActionTesting::next_action<component>(make_not_null(&runner), 0);
  // InitializeMeasurements
  ActionTesting::next_action<component>(make_not_null(&runner), 0);

  auto& cache = ActionTesting::cache<component>(runner, 0);
  auto& box = ActionTesting::get_databox<component>(make_not_null(&runner), 0);

  auto& events_and_dense_triggers =
      db::get_mutable_reference<evolution::Tags::EventsAndDenseTriggers>(
          make_not_null(&box));
  // This call initializes events_and_dense_triggers internals
  events_and_dense_triggers.next_trigger(box);
  CAPTURE(db::get<Tags::Time>(box));
  CHECK(events_and_dense_triggers.is_ready(box, cache, 0, component_p) ==
        evolution::EventsAndDenseTriggers::TriggeringState::
            NeedsEvolvedVariables);
  events_and_dense_triggers.run_events(box, cache, 0, component_p);

  CHECK(Submeasurement<tmpl::list<SystemA, SystemB>>::call_count == 1);
  CHECK(Submeasurement<tmpl::list<SystemC>>::call_count == 1);
}
}  // namespace
