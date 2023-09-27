// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <cstddef>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "ControlSystem/Actions/LimitTimeStep.hpp"
#include "ControlSystem/FutureMeasurements.hpp"
#include "ControlSystem/Tags/FutureMeasurements.hpp"
#include "ControlSystem/Tags/MeasurementTimescales.hpp"
#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/DataVector.hpp"
#include "Domain/FunctionsOfTime/FunctionOfTime.hpp"
#include "Domain/FunctionsOfTime/PiecewisePolynomial.hpp"
#include "Domain/FunctionsOfTime/Tags.hpp"
#include "Framework/ActionTesting.hpp"
#include "Helpers/ControlSystem/TestStructs.hpp"
#include "Parallel/Phase.hpp"
#include "Parallel/PhaseDependentActionList.hpp"
#include "Time/History.hpp"
#include "Time/Slab.hpp"
#include "Time/Tags/AdaptiveSteppingDiagnostics.hpp"
#include "Time/Tags/HistoryEvolvedVariables.hpp"
#include "Time/Tags/TimeStep.hpp"
#include "Time/Tags/TimeStepId.hpp"
#include "Time/Tags/TimeStepper.hpp"
#include "Time/Time.hpp"
#include "Time/TimeStepId.hpp"
#include "Time/TimeSteppers/AdamsBashforth.hpp"
#include "Time/TimeSteppers/AdamsMoultonPc.hpp"
#include "Time/TimeSteppers/Rk3HesthavenSsp.hpp"
#include "Time/TimeSteppers/TimeStepper.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Serialization/RegisterDerivedClassesWithCharm.hpp"
#include "Utilities/TMPL.hpp"

namespace {
struct LabelA {};
struct LabelB {};
struct LabelC {};

using MeasurementA = control_system::TestHelpers::Measurement<LabelA>;
using MeasurementB = control_system::TestHelpers::Measurement<LabelB>;

using systemsA =
    tmpl::list<control_system::TestHelpers::System<0, LabelA, MeasurementA>>;
using systemsB =
    tmpl::list<control_system::TestHelpers::System<0, LabelB, MeasurementB>,
               control_system::TestHelpers::System<0, LabelC, MeasurementB>>;

using control_systems = tmpl::append<systemsA, systemsB>;

struct Var : db::SimpleTag {
  using type = double;
};

struct FunctionsOfTimeTag : domain::Tags::FunctionsOfTime, db::SimpleTag {
  using type = std::unordered_map<
      std::string, std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>;
};

template <typename Metavariables>
struct Component {
  using metavariables = Metavariables;
  using chare_type = ActionTesting::MockArrayChare;
  using array_index = int;
  using mutable_global_cache_tags =
      tmpl::list<control_system::Tags::MeasurementTimescales,
                 FunctionsOfTimeTag>;
  using simple_tags = db::AddSimpleTags<
      control_system::Tags::FutureMeasurements<systemsA>,
      control_system::Tags::FutureMeasurements<systemsB>,
      Tags::TimeStepper<TimeStepper>, Tags::TimeStepId,
      Tags::Next<Tags::TimeStepId>, Tags::TimeStep, Tags::Next<Tags::TimeStep>,
      Tags::AdaptiveSteppingDiagnostics, Tags::HistoryEvolvedVariables<Var>>;
  using compute_tags = db::AddComputeTags<>;
  using phase_dependent_action_list = tmpl::list<
      Parallel::PhaseActions<Parallel::Phase::Initialization,
                             tmpl::list<ActionTesting::InitializeDataBox<
                                 simple_tags, compute_tags>>>,
      Parallel::PhaseActions<
          Parallel::Phase::Testing,
          tmpl::list<control_system::Actions::LimitTimeStep<control_systems>>>>;
};

struct Metavariables {
  static constexpr bool local_time_stepping = false;
  using component_list = tmpl::list<Component<Metavariables>>;
};

void test(const std::string& test_label, const double initial_time,
          const double initial_step_end,
          const std::optional<double>& expected_step_end,
          const std::vector<std::pair<double, double>>& measurement_updatesA,
          const std::vector<std::pair<double, double>>& fot_updatesA,
          const std::vector<std::pair<double, double>>& measurement_updatesBC,
          const std::vector<std::pair<double, double>>& fot_updatesB,
          const std::vector<std::pair<double, double>>& fot_updatesC,
          std::unique_ptr<TimeStepper> stepper =
              std::make_unique<TimeSteppers::Rk3HesthavenSsp>(),
          TimeSteppers::History<Var::type> history = {}) {
  INFO(test_label);
  ASSERT(measurement_updatesA.size() > 1, "Bad argument");
  ASSERT(measurement_updatesBC.size() > 1, "Bad argument");
  ASSERT(fot_updatesA.size() > 1, "Bad argument");
  ASSERT(fot_updatesB.size() > 1, "Bad argument");
  ASSERT(fot_updatesC.size() > 1, "Bad argument");

  const Slab initial_slab(initial_time, initial_step_end);
  const TimeStepId initial_id(true, 0, initial_slab.start());

  const size_t measurements_per_update = 3;

  const auto setup_fot =
      [](const std::vector<std::pair<double, double>>& updates) {
        auto fot =
            std::make_unique<domain::FunctionsOfTime::PiecewisePolynomial<0>>(
                updates.front().first,
                std::array{DataVector{updates.front().second}},
                updates[1].first);
        for (size_t i = 1; i < updates.size() - 1; ++i) {
          fot->update(updates[i].first, DataVector{updates[i].second},
                      updates[i + 1].first);
        }
        return fot;
      };

  control_system::Tags::MeasurementTimescales::type timescales{};
  timescales["LabelA"] = setup_fot(measurement_updatesA);
  timescales["LabelBLabelC"] = setup_fot(measurement_updatesBC);
  FunctionsOfTimeTag::type functions_of_time{};
  functions_of_time["LabelA"] = setup_fot(fot_updatesA);
  functions_of_time["LabelB"] = setup_fot(fot_updatesB);
  functions_of_time["LabelC"] = setup_fot(fot_updatesC);

  const auto setup_measurements =
      [&initial_time](
          const domain::FunctionsOfTime::FunctionOfTime& timescale) {
        if (timescale.func(0.0)[0][0] ==
            std::numeric_limits<double>::infinity()) {
          return control_system::FutureMeasurements(
              1, std::numeric_limits<double>::infinity());
        }

        control_system::FutureMeasurements measurements(measurements_per_update,
                                                        0.0);
        measurements.update(timescale);
        while (measurements.next_measurement().value_or(
                   std::numeric_limits<double>::infinity()) <= initial_time) {
          measurements.pop_front();
        }
        return measurements;
      };

  auto measurementsA = setup_measurements(*timescales["LabelA"]);
  auto measurementsBC = setup_measurements(*timescales["LabelBLabelC"]);

  using MockRuntimeSystem = ActionTesting::MockRuntimeSystem<Metavariables>;
  using component = Component<Metavariables>;
  MockRuntimeSystem runner{
      {}, {std::move(timescales), std::move(functions_of_time)}};
  ActionTesting::emplace_array_component_and_initialize<component>(
      make_not_null(&runner), ActionTesting::NodeId{0},
      ActionTesting::LocalCoreId{0}, 0,
      {std::move(measurementsA), std::move(measurementsBC), std::move(stepper),
       initial_id, TimeStepId{}, initial_slab.duration(), TimeDelta{},
       AdaptiveSteppingDiagnostics{}, std::move(history)});

  ActionTesting::set_phase(make_not_null(&runner), Parallel::Phase::Testing);
  const bool ready =
      ActionTesting::next_action_if_ready<component>(make_not_null(&runner), 0);
  if (ready and expected_step_end.has_value()) {
    const Slab expected_slab(initial_time, *expected_step_end);
    CHECK(ActionTesting::get_databox_tag<component, Tags::TimeStep>(
              runner, 0) == expected_slab.duration());
  } else {
    CHECK(not ready);
    CHECK(not expected_step_end.has_value());
  }
}
}  // namespace

SPECTRE_TEST_CASE("Unit.ControlSystem.Actions.LimitTimeStep",
                  "[Unit][ControlSystem]") {
  register_classes_with_charm<
      TimeSteppers::AdamsBashforth, TimeSteppers::AdamsMoultonPc,
      TimeSteppers::Rk3HesthavenSsp,
      domain::FunctionsOfTime::PiecewisePolynomial<0>>();
  const double infinity = std::numeric_limits<double>::infinity();
  const double nan = std::numeric_limits<double>::signaling_NaN();
  const double arbitrary = -1234.0;

  // For each active control system below, the interval from the
  // measurement triggering the update to the FoT expiration time is
  // indicated as the goal interval.  A step must occur in this
  // interval to avoid a deadlock.

  // clang-format off
  test("No control systems", 1.0, 5.0, 5.0,
       {{0.0, infinity}, {infinity, nan}},
       {{0.0, arbitrary}, {infinity, nan}},
       {{0.0, infinity}, {infinity, nan}},
       {{0.0, arbitrary}, {infinity, nan}},
       {{0.0, arbitrary}, {infinity, nan}});
  test("Step much shorter than limit", 1.0, 5.0, 5.0,
       {{0.0, 10.0}, {50.0, nan}},
       {{0.0, arbitrary}, {50.0, nan}},  // goal range [20, 50]
       {{0.0, infinity}, {infinity, nan}},
       {{0.0, arbitrary}, {infinity, nan}},
       {{0.0, arbitrary}, {infinity, nan}});
  test("Limited by expiration", 1.0, 5.0, 4.0,
       {{0.0, 1.0}, {5.0, nan}},
       {{0.0, arbitrary}, {4.0, nan}},  // goal range [2, 4]
       {{0.0, infinity}, {infinity, nan}},
       {{0.0, arbitrary}, {infinity, nan}},
       {{0.0, arbitrary}, {infinity, nan}});
  test("Adjusted to keep steps even", 1.0, 8.0, 6.0,
       {{0.0, 5.0}, {20.0, nan}},
       {{0.0, arbitrary}, {11.0, nan}},  // goal range [10, 11]
       {{0.0, infinity}, {infinity, nan}},
       {{0.0, arbitrary}, {infinity, nan}},
       {{0.0, arbitrary}, {infinity, nan}});
  test("Past update but not to expiration", 1.0, 8.0, 8.0,
       {{0.0, 3.0}, {20.0, nan}},
       {{0.0, arbitrary}, {11.0, nan}},  // goal range [6, 11]
       {{0.0, infinity}, {infinity, nan}},
       {{0.0, arbitrary}, {infinity, nan}},
       {{0.0, arbitrary}, {infinity, nan}});
  test("Limited by expiration, 2 systems", 1.0, 7.0, 5.0,
       {{0.0, infinity}, {infinity, nan}},
       {{0.0, arbitrary}, {infinity, nan}},
       {{0.0, 1.0}, {5.0, nan}},
       {{0.0, arbitrary}, {5.0, nan}},  // goal range [2, 5]
       {{0.0, arbitrary}, {6.0, nan}});  // goal range [2, 6]
  test("Limited by expiration, 2 measurements", 1.0, 6.0, 5.0,
       {{0.0, 2.0}, {10.0, nan}},
       {{0.0, arbitrary}, {6.0, nan}},  // goal range [4, 6]
       {{0.0, 1.0}, {5.0, nan}},
       {{0.0, arbitrary}, {5.0, nan}},  // goal range [2, 5]
       {{0.0, arbitrary}, {infinity, nan}});
  test("Adjusted to keep steps even, 2 systems", 1.0, 11.0, 7.0,
       {{0.0, 6.0}, {20.0, nan}},
       {{0.0, arbitrary}, {13.0, nan}},  // goal range [12, 13]
       {{0.0, 2.0}, {20.0, nan}},
       {{0.0, arbitrary}, {15.0, nan}},  // goal range [4, 15]
       {{0.0, arbitrary}, {infinity, nan}});
  test("Adjusted to keep steps even, limited by update", 1.0, 11.0, 8.0,
       {{0.0, 6.0}, {20.0, nan}},
       {{0.0, arbitrary}, {13.0, nan}},  // goal range [12, 13]
       {{0.0, 4.0}, {20.0, nan}},
       {{0.0, arbitrary}, {15.0, nan}},  // goal range [8, 15]
       {{0.0, arbitrary}, {infinity, nan}});
  test("Would be limited by expiration as of now", 1.0, 5.0, 5.0,
       {{0.0, 2.0}, {5.0, nan}},
       {{0.0, arbitrary}, {2.0, arbitrary}, {20.0, nan}},  // goal range [4, 20]
       {{0.0, infinity}, {infinity, nan}},
       {{0.0, arbitrary}, {infinity, nan}},
       {{0.0, arbitrary}, {infinity, nan}});
  test("Insufficient timescale data", 1.0, 5.0, std::nullopt,
       {{0.0, 10.0}, {5.0, nan}},
       {{0.0, arbitrary}, {50.0, nan}},  // goal range [?, ?]
       {{0.0, infinity}, {infinity, nan}},
       {{0.0, arbitrary}, {infinity, nan}},
       {{0.0, arbitrary}, {infinity, nan}});
  test("Barely sufficient timescale data", 1.0, 5.0, 5.0,
       {{0.0, 10.0}, {10.0, nan}},
       {{0.0, arbitrary}, {50.0, nan}},  // goal range [20, 50]
       {{0.0, infinity}, {infinity, nan}},
       {{0.0, arbitrary}, {infinity, nan}},
       {{0.0, arbitrary}, {infinity, nan}});
  test("Insufficient FoT data", 1.0, 5.0, std::nullopt,
       {{0.0, 10.0}, {50.0, nan}},
       {{0.0, arbitrary}, {15.0, nan}},  // goal range [20, ?]
       {{0.0, infinity}, {infinity, nan}},
       {{0.0, arbitrary}, {infinity, nan}},
       {{0.0, arbitrary}, {infinity, nan}});
  test("Barely insufficient FoT data", 1.0, 5.0, std::nullopt,
       {{0.0, 10.0}, {50.0, nan}},
       {{0.0, arbitrary}, {20.0, nan}},  // goal range [20, ?]
       {{0.0, infinity}, {infinity, nan}},
       {{0.0, arbitrary}, {infinity, nan}},
       {{0.0, arbitrary}, {infinity, nan}});

  test("Does nothing with Adams-Bashforth", 1.0, 5.0, 5.0,
       {{0.0, 1.0}, {5.0, nan}},
       {{0.0, arbitrary}, {3.0, nan}},  // goal range [2, 3]
       {{0.0, infinity}, {infinity, nan}},
       {{0.0, arbitrary}, {infinity, nan}},
       {{0.0, arbitrary}, {infinity, nan}},
       std::make_unique<TimeSteppers::AdamsBashforth>(4));
  test("Doesn't need data with Adams-Bashforth", 1.0, 5.0, 5.0,
       {{0.0, 10.0}, {5.0, nan}},
       {{0.0, arbitrary}, {50.0, nan}},  // goal range [?, ?]
       {{0.0, infinity}, {infinity, nan}},
       {{0.0, arbitrary}, {infinity, nan}},
       {{0.0, arbitrary}, {infinity, nan}},
       std::make_unique<TimeSteppers::AdamsBashforth>(4));

  TimeSteppers::History<Var::type> history{};
  history.insert(TimeStepId(true, -1, Time(Slab(1.0, 5.0), {1, 2})), {}, {});
  history.insert(TimeStepId(true, 0, Slab(1.0, 5.0).start()), {}, {});
  test("Step can't change but is OK", 1.0, 5.0, 5.0,
       {{0.0, 10.0}, {50.0, nan}},
       {{0.0, arbitrary}, {50.0, nan}},  // goal range [20, 50]
       {{0.0, infinity}, {infinity, nan}},
       {{0.0, arbitrary}, {infinity, nan}},
       {{0.0, arbitrary}, {infinity, nan}},
       std::make_unique<TimeSteppers::AdamsMoultonPc>(4), history);
  CHECK_THROWS_WITH(
      test("Step can't change but is not OK", 1.0, 5.0, 0.0,
           {{0.0, 1.0}, {5.0, nan}},
           {{0.0, arbitrary}, {4.0, nan}},  // goal range [2, 4]
           {{0.0, infinity}, {infinity, nan}},
           {{0.0, arbitrary}, {infinity, nan}},
           {{0.0, arbitrary}, {infinity, nan}},
           std::make_unique<TimeSteppers::AdamsMoultonPc>(4), history),
      Catch::Matchers::ContainsSubstring(
          "Step must be decreased to avoid control-system deadlock, but "
          "time-stepper requires a fixed step size."));
  // clang-format on
}
