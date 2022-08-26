// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <memory>
#include <random>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"
#include "Domain/FunctionsOfTime/FunctionOfTime.hpp"
#include "Domain/FunctionsOfTime/PiecewisePolynomial.hpp"
#include "Domain/FunctionsOfTime/QuaternionFunctionOfTime.hpp"
#include "Domain/FunctionsOfTime/Tags.hpp"
#include "Framework/ActionTesting.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/DataStructures/MakeWithRandomValues.hpp"
#include "Parallel/Phase.hpp"
#include "Parallel/PhaseDependentActionList.hpp"
#include "Parallel/RegisterDerivedClassesWithCharm.hpp"
#include "Time/Actions/LimitTimeStepToExpirationTimes.hpp"
#include "Time/Slab.hpp"
#include "Time/Tags.hpp"
#include "Time/Time.hpp"
#include "Time/TimeStepId.hpp"
#include "Time/TimeSteppers/DormandPrince5.hpp"
#include "Time/TimeSteppers/RungeKutta4.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

class TimeStepper;

namespace {
template <typename Metavariables>
struct Component {
  using metavariables = Metavariables;
  using chare_type = ActionTesting::MockArrayChare;
  using array_index = int;
  using const_global_cache_tags = tmpl::list<Tags::TimeStepper<TimeStepper>>;
  using mutable_global_cache_tags =
      tmpl::list<domain::Tags::FunctionsOfTimeInitialize>;

  using simple_tags = db::AddSimpleTags<Tags::TimeStepId, Tags::TimeStep,
                                        Tags::Time, Tags::Next<Tags::TimeStep>,
                                        Tags::Next<Tags::TimeStepId>>;

  using phase_dependent_action_list =
      tmpl::list<Parallel::PhaseActions<
                     Parallel::Phase::Initialization,
                     tmpl::list<ActionTesting::InitializeDataBox<simple_tags>>>,
                 Parallel::PhaseActions<
                     Parallel::Phase::Testing,
                     tmpl::list<Actions::LimitTimeStepToExpirationTimes>>>;
};

struct Metavariables {
  using component_list = tmpl::list<Component<Metavariables>>;
};

template <typename TimeStepper>
void check(const Time& start, const Time& end, const TimeDelta& time_step) {
  MAKE_GENERATOR(generator);
  const double eps = 1.e-10;
  std::uniform_real_distribution<double> expiration_times_before_next_time_dist{
      start.value() + eps, start.value() + time_step.value() - eps};
  std::uniform_real_distribution<double> expiration_times_after_next_time_dist{
      start.value() + time_step.value() + eps, end.value() - eps};
  const std::array<double, 2> before_expiration_times =
      make_with_random_values<std::array<double, 2>>(
          make_not_null(&generator), expiration_times_before_next_time_dist,
          std::array<double, 2>{});
  const std::array<double, 1> after_expiration_times =
      make_with_random_values<std::array<double, 1>>(
          make_not_null(&generator), expiration_times_after_next_time_dist,
          std::array<double, 1>{});
  const double min_expiration_time{*std::min_element(
      before_expiration_times.begin(), before_expiration_times.end())};

  std::unordered_map<std::string,
                     std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>
      functions_of_time{};
  std::array<DataVector, 4> initial_expansion{
      {{{1.0}}, {{0.2}}, {{0.03}}, {{0.004}}}};
  std::array<DataVector, 4> initial_unity{{{{1.0}}, {{0.0}}, {{0.0}}, {{0.0}}}};
  const std::array<DataVector, 4> initial_rotation{{{3, 0.0},
                                                    {{0.0, 0.0, -0.1}},
                                                    {{0.0, 0.0, -0.02}},
                                                    {{0.0, 0.0, -0.003}}}};
  const std::array<DataVector, 1> initial_quaternion{{{1.0, 0.0, 0.0, 0.0}}};
  functions_of_time.insert(
      {"CubicScaleA",
       std::make_unique<domain::FunctionsOfTime::PiecewisePolynomial<3>>(
           start.value(), initial_expansion, before_expiration_times[0])});
  functions_of_time.insert(
      {"CubicScaleB",
       std::make_unique<domain::FunctionsOfTime::PiecewisePolynomial<3>>(
           start.value(), initial_unity, after_expiration_times[0])});
  functions_of_time.insert(
      {"Rotation",
       std::make_unique<domain::FunctionsOfTime::QuaternionFunctionOfTime<3>>(
           start.value(), initial_quaternion, initial_rotation,
           before_expiration_times[1])});

  using component = Component<Metavariables>;
  using MockRuntimeSystem = ActionTesting::MockRuntimeSystem<Metavariables>;
  MockRuntimeSystem runner{{std::make_unique<TimeStepper>()},
                           {std::move(functions_of_time)}};
  const TimeStepId& time_step_id{time_step.is_positive(), 8, start};
  const TimeStepId& next_time_step_id{time_step.is_positive(), 9,
                                      start + time_step};
  ActionTesting::emplace_component_and_initialize<component>(
      &runner, 0,
      {time_step_id, time_step, start.value(), time_step, next_time_step_id});
  ActionTesting::set_phase(make_not_null(&runner), Parallel::Phase::Testing);

  runner.next_action<component>(0);

  const auto& box = ActionTesting::get_databox<component>(runner, 0);
  const auto& new_time_step = db::get<Tags::TimeStep>(box);
  const auto& new_next_time_step = db::get<Tags::Next<Tags::TimeStep>>(box);
  const auto& new_time_step_id = db::get<Tags::TimeStepId>(box);
  const auto& new_next_time_step_id =
      db::get<Tags::Next<Tags::TimeStepId>>(box);
  const double new_slab_start_value{new_time_step.slab().start().value()};
  const double new_slab_end_value{new_time_step.slab().end().value()};

  CHECK(new_slab_start_value == time_step.slab().start().value());
  CHECK((min_expiration_time == new_slab_end_value));

  CHECK(new_slab_start_value + time_step.value() > min_expiration_time);
  CHECK(new_slab_start_value + new_time_step.value() == min_expiration_time);

  CHECK(new_time_step.fraction() == Rational{1, 1});
  CHECK(new_time_step.slab().start().value() == new_slab_start_value);
  CHECK(new_time_step.slab().end().value() == new_slab_end_value);

  CHECK(new_time_step_id.time_runs_forward() ==
        time_step_id.time_runs_forward());
  CHECK(new_time_step_id.slab_number() == time_step_id.slab_number());
  CHECK(new_time_step_id.step_time().fraction() ==
        time_step_id.step_time().fraction());
  CHECK(new_time_step_id.substep() == time_step_id.substep());
  CHECK(new_time_step_id.substep_time().fraction() ==
        time_step_id.substep_time().fraction());
  CHECK(new_time_step_id.step_time().slab().start().value() ==
        new_slab_start_value);
  CHECK(new_time_step_id.step_time().slab().end().value() ==
        new_slab_end_value);
  CHECK(new_time_step_id.substep_time().slab().start().value() ==
        new_slab_start_value);
  CHECK(new_time_step_id.substep_time().slab().end().value() ==
        new_slab_end_value);

  CHECK(new_next_time_step_id == db::get<Tags::TimeStepper<>>(box).next_time_id(
                                     new_time_step_id, new_time_step));
  CHECK(new_next_time_step.fraction() == time_step.fraction());
  CHECK(new_next_time_step.slab() == new_next_time_step_id.step_time().slab());
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Time.Actions.LimitTimeStepToExpirationTimes",
                  "[Unit][Time][Actions]") {
  Parallel::register_classes_with_charm<
      TimeSteppers::DormandPrince5, TimeSteppers::RungeKutta4,
      domain::FunctionsOfTime::PiecewisePolynomial<3>,
      domain::FunctionsOfTime::QuaternionFunctionOfTime<3>>();
  const Slab slab(0.1, 1.1);
  check<TimeSteppers::DormandPrince5>(slab.start(), slab.end(),
                                      slab.duration());
  check<TimeSteppers::RungeKutta4>(slab.start(), slab.end(), slab.duration());
}
