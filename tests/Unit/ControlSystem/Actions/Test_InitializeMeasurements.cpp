// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <cstddef>
#include <memory>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <utility>

#include "ControlSystem/Actions/InitializeMeasurements.hpp"
#include "ControlSystem/Event.hpp"
#include "ControlSystem/Protocols/ControlSystem.hpp"
#include "ControlSystem/Protocols/Measurement.hpp"
#include "ControlSystem/Protocols/Submeasurement.hpp"
#include "ControlSystem/Trigger.hpp"
#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataVector.hpp"
#include "Domain/FunctionsOfTime/FunctionOfTime.hpp"
#include "Domain/FunctionsOfTime/PiecewisePolynomial.hpp"
#include "Evolution/Actions/RunEventsAndDenseTriggers.hpp"
#include "Evolution/EventsAndDenseTriggers/DenseTrigger.hpp"
#include "Evolution/EventsAndDenseTriggers/EventsAndDenseTriggers.hpp"
#include "Evolution/EventsAndDenseTriggers/Tags.hpp"
#include "Framework/ActionTesting.hpp"
#include "Framework/MockDistributedObject.hpp"
#include "Framework/MockRuntimeSystem.hpp"
#include "Framework/MockRuntimeSystemFreeFunctions.hpp"
#include "Options/Protocols/FactoryCreation.hpp"
#include "Parallel/Actions/SetupDataBox.hpp"
#include "Parallel/PhaseDependentActionList.hpp"
#include "Parallel/RegisterDerivedClassesWithCharm.hpp"
#include "ParallelAlgorithms/EventsAndTriggers/Event.hpp"
#include "Time/Tags.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/ProtocolHelpers.hpp"
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
  using simple_tags = tmpl::list<>;
  using measurement = Measurement<tmpl::list<SystemA, SystemB>>;
  struct process_measurement {
    template <typename Submeasurement>
    using argument_tags = tmpl::list<>;
  };
};

struct SystemB : tt::ConformsTo<control_system::protocols::ControlSystem> {
  static std::string name() { return "B"; }
  using simple_tags = tmpl::list<>;
  using measurement = Measurement<tmpl::list<SystemA, SystemB>>;
  struct process_measurement {
    template <typename Submeasurement>
    using argument_tags = tmpl::list<>;
  };
};

struct SystemC : tt::ConformsTo<control_system::protocols::ControlSystem> {
  static std::string name() { return "C"; }
  using simple_tags = tmpl::list<>;
  using measurement = Measurement<tmpl::list<SystemC>>;
  struct process_measurement {
    template <typename Submeasurement>
    using argument_tags = tmpl::list<>;
  };
};

static_assert(
    tt::assert_conforms_to<SystemA, control_system::protocols::ControlSystem>);
static_assert(
    tt::assert_conforms_to<SystemB, control_system::protocols::ControlSystem>);
static_assert(
    tt::assert_conforms_to<SystemC, control_system::protocols::ControlSystem>);

using control_systems = tmpl::list<SystemA, SystemB, SystemC>;

template <typename Metavariables>
struct Component {
  using metavariables = Metavariables;
  using chare_type = ActionTesting::MockArrayChare;
  using array_index = int;

  using initialization_tags =
      tmpl::list<Tags::TimeStepId, Tags::Time,
                 evolution::Tags::EventsAndDenseTriggers>;

  using phase_dependent_action_list = tmpl::list<Parallel::PhaseActions<
      typename Metavariables::Phase, Metavariables::Phase::Initialization,
      tmpl::list<
          Actions::SetupDataBox,
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

  enum class Phase { Initialization, Testing, Exit };
};

SPECTRE_TEST_CASE("Unit.ControlSystem.InitializeMeasurements",
                  "[ControlSystem][Unit]") {
  Parallel::register_factory_classes_with_charm<Metavariables>();
  Parallel::register_classes_with_charm<
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

  MockRuntimeSystem runner{{}, {std::move(timescales)}};
  ActionTesting::emplace_array_component<component>(
      make_not_null(&runner), ActionTesting::NodeId{0},
      ActionTesting::LocalCoreId{0}, 0, Tags::TimeStepId::type{true, 0, {}},
      Tags::Time::type{0.0}, evolution::Tags::EventsAndDenseTriggers::type{});

  // SetupDataBox
  ActionTesting::next_action<component>(make_not_null(&runner), 0);
  // InitializeRunEventsAndDenseTriggers
  ActionTesting::next_action<component>(make_not_null(&runner), 0);
  // InitializeMeasurements
  ActionTesting::next_action<component>(make_not_null(&runner), 0);

  using box_tags =
      tmpl::append<Actions::SetupDataBox::action_list_simple_tags<component>,
                   Actions::SetupDataBox::action_list_compute_tags<component>>;
  auto& cache = ActionTesting::cache<component>(runner, 0);
  auto& box = ActionTesting::get_databox<component, box_tags>(
      make_not_null(&runner), 0);

  auto& events_and_dense_triggers =
      db::get_mutable_reference<evolution::Tags::EventsAndDenseTriggers>(
          make_not_null(&box));
  // This call initializes events_and_dense_triggers internals
  events_and_dense_triggers.next_trigger(box);
  events_and_dense_triggers.run_events(box, cache, 0, component_p);

  CHECK(Submeasurement<tmpl::list<SystemA, SystemB>>::call_count == 1);
  CHECK(Submeasurement<tmpl::list<SystemC>>::call_count == 1);
}
}  // namespace
