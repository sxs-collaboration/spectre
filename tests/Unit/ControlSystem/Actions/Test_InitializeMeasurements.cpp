// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <cstddef>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <utility>

#include "ControlSystem/Actions/InitializeMeasurements.hpp"
#include "ControlSystem/Metafunctions.hpp"
#include "ControlSystem/Protocols/ControlSystem.hpp"
#include "ControlSystem/Protocols/Measurement.hpp"
#include "ControlSystem/Protocols/Submeasurement.hpp"
#include "ControlSystem/Tags/FutureMeasurements.hpp"
#include "ControlSystem/Tags/SystemTags.hpp"
#include "ControlSystem/Trigger.hpp"
#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataVector.hpp"
#include "Domain/Creators/Tags/FunctionsOfTime.hpp"
#include "Domain/FunctionsOfTime/FunctionOfTime.hpp"
#include "Domain/FunctionsOfTime/PiecewisePolynomial.hpp"
#include "Domain/FunctionsOfTime/Tags.hpp"
#include "Evolution/Actions/RunEventsAndDenseTriggers.hpp"
#include "Framework/ActionTesting.hpp"
#include "Framework/MockDistributedObject.hpp"
#include "Framework/MockRuntimeSystem.hpp"
#include "Framework/MockRuntimeSystemFreeFunctions.hpp"
#include "Helpers/ControlSystem/TestStructs.hpp"
#include "IO/Logging/Verbosity.hpp"
#include "Options/Protocols/FactoryCreation.hpp"
#include "Parallel/Phase.hpp"
#include "Parallel/PhaseDependentActionList.hpp"
#include "ParallelAlgorithms/EventsAndDenseTriggers/DenseTrigger.hpp"
#include "ParallelAlgorithms/EventsAndDenseTriggers/EventsAndDenseTriggers.hpp"
#include "ParallelAlgorithms/EventsAndDenseTriggers/Tags.hpp"
#include "ParallelAlgorithms/EventsAndTriggers/Event.hpp"
#include "Time/Slab.hpp"
#include "Time/Tags/Time.hpp"
#include "Time/Tags/TimeStep.hpp"
#include "Time/Tags/TimeStepId.hpp"
#include "Utilities/CartesianProduct.hpp"
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

  template <typename ControlSystems>
  using event = control_system::TestHelpers::TestEvent<ExpectedSystems,
                                                       ExpectedSystems, false>;
};

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

  using simple_tags = tmpl::list<Tags::TimeStepId, Tags::Time, Tags::TimeStep>;
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

template <bool LocalTimeStepping>
struct Metavariables {
  using component_list = tmpl::list<Component<Metavariables>>;
  static constexpr bool local_time_stepping = LocalTimeStepping;

  struct factory_creation
      : tt::ConformsTo<Options::protocols::FactoryCreation> {
    using factory_classes = tmpl::map<
        tmpl::pair<Event, control_system::metafunctions::control_system_events<
                              control_systems>>,
        tmpl::pair<DenseTrigger,
                   control_system::control_system_triggers<control_systems>>>;
  };
};

template <bool LocalTimeStepping>
void test_initialize_measurements(const bool ab_active, const bool c_active) {
  CAPTURE(LocalTimeStepping);
  CAPTURE(ab_active);
  CAPTURE(c_active);

  register_factory_classes_with_charm<Metavariables<LocalTimeStepping>>();
  register_classes_with_charm<
      domain::FunctionsOfTime::PiecewisePolynomial<0>>();

  using MockRuntimeSystem =
      ActionTesting::MockRuntimeSystem<Metavariables<LocalTimeStepping>>;
  using component = Component<Metavariables<LocalTimeStepping>>;
  const component* const component_p = nullptr;

  // Details shouldn't matter
  const double initial_time = 2.0;
  const size_t measurements_per_update = 6;
  const double initial_timescale = 1.5;
  const double fot_expiration = 4.0;

  const double step_to_expiration_ratio = c_active ? 3.0 : 0.2;

  const auto initial_slab = Slab::with_duration_from_start(
      initial_time, step_to_expiration_ratio * (fot_expiration - initial_time));
  const auto initial_time_step = initial_slab.duration();

  const auto timescale =
      std::make_unique<domain::FunctionsOfTime::PiecewisePolynomial<0>>(
          0.0, std::array{DataVector{initial_timescale}}, 2.0);
  const auto inactive_timescale =
      std::make_unique<domain::FunctionsOfTime::PiecewisePolynomial<0>>(
          0.0, std::array{DataVector{std::numeric_limits<double>::infinity()}},
          std::numeric_limits<double>::infinity());
  std::unordered_map<std::string,
                     std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>
      timescales;
  timescales.emplace("AB",
                     (ab_active ? timescale : inactive_timescale)->get_clone());
  timescales.emplace("C",
                     (c_active ? timescale : inactive_timescale)->get_clone());

  const auto fot =
      std::make_unique<domain::FunctionsOfTime::PiecewisePolynomial<0>>(
          0.0, std::array{DataVector{1.0}}, fot_expiration);

  std::unordered_map<std::string,
                     std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>
      functions;
  functions.emplace("A", fot->get_clone());
  functions.emplace("B", fot->get_clone());
  functions.emplace("C", fot->get_clone());

  MockRuntimeSystem runner{{::Verbosity::Silent, measurements_per_update},
                           {std::move(functions), std::move(timescales)}};
  ActionTesting::emplace_array_component_and_initialize<component>(
      make_not_null(&runner), ActionTesting::NodeId{0},
      ActionTesting::LocalCoreId{0}, 0,
      {Tags::TimeStepId::type{true, 0, {}}, initial_time, initial_time_step},
      evolution::Tags::EventsAndDenseTriggers::type{});

  // InitializeRunEventsAndDenseTriggers
  ActionTesting::next_action<component>(make_not_null(&runner), 0);
  // InitializeMeasurements
  ActionTesting::next_action<component>(make_not_null(&runner), 0);

  auto& cache = ActionTesting::cache<component>(runner, 0);
  auto& box = ActionTesting::get_databox<component>(make_not_null(&runner), 0);

  CHECK(db::get<control_system::Tags::FutureMeasurements<
            tmpl::list<SystemA, SystemB>>>(box)
            .next_measurement() ==
        std::optional(ab_active ? initial_time
                                : std::numeric_limits<double>::infinity()));
  CHECK(db::get<control_system::Tags::FutureMeasurements<tmpl::list<SystemC>>>(
            box)
            .next_measurement() ==
        std::optional(c_active ? initial_time
                               : std::numeric_limits<double>::infinity()));

  CHECK(db::get<control_system::Tags::FutureMeasurements<
            tmpl::list<SystemA, SystemB>>>(box)
            .next_update() ==
        (ab_active ? std::nullopt
                   : std::optional(std::numeric_limits<double>::infinity())));
  CHECK(db::get<control_system::Tags::FutureMeasurements<tmpl::list<SystemC>>>(
            box)
            .next_update() ==
        (c_active ? std::nullopt
                  : std::optional(std::numeric_limits<double>::infinity())));

  if (c_active) {
    // 4 is because of the 3.0 in the step_to_expiration_ratio above.
    const auto reduced_time_step =
        LocalTimeStepping ? initial_time_step / 4
                          : Slab(initial_time, fot_expiration).duration();
    CHECK(db::get<::Tags::TimeStep>(box) == reduced_time_step);
  } else {
    CHECK(db::get<::Tags::TimeStep>(box) == initial_time_step);
  }

  auto& events_and_dense_triggers =
      db::get_mutable_reference<evolution::Tags::EventsAndDenseTriggers>(
          make_not_null(&box));
  // This call initializes events_and_dense_triggers internals
  events_and_dense_triggers.next_trigger(box);
  CAPTURE(db::get<Tags::Time>(box));
  CHECK(events_and_dense_triggers.is_ready(make_not_null(&box), cache, 0,
                                           component_p) ==
        (ab_active or c_active
             ? evolution::EventsAndDenseTriggers::TriggeringState::
                   NeedsEvolvedVariables
             : evolution::EventsAndDenseTriggers::TriggeringState::Ready));

  using ListAB = tmpl::list<SystemA, SystemB>;
  using ListC = tmpl::list<SystemC>;
  Submeasurement<ListAB>::event<ListAB>::call_count = 0;
  Submeasurement<ListC>::event<ListC>::call_count = 0;
  events_and_dense_triggers.run_events(box, cache, 0, component_p);

  CHECK(Submeasurement<ListAB>::event<ListAB>::call_count ==
        (ab_active ? 1 : 0));
  CHECK(Submeasurement<ListC>::event<ListC>::call_count == (c_active ? 1 : 0));
}

SPECTRE_TEST_CASE("Unit.ControlSystem.InitializeMeasurements",
                  "[ControlSystem][Unit]") {
  for (const auto& [ab_active, c_active] :
       cartesian_product(std::array{true, false}, std::array{true, false})) {
    test_initialize_measurements<false>(ab_active, c_active);
    test_initialize_measurements<true>(ab_active, c_active);
  }
}
}  // namespace
