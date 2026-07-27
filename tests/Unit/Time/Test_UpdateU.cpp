// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <cstddef>
#include <memory>
#include <optional>
#include <utility>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/DataBox/Tag.hpp"
#include "Time/Slab.hpp"
#include "Time/StepperErrorTolerances.hpp"
#include "Time/Tags/HistoryEvolvedVariables.hpp"
#include "Time/Tags/StepperErrorEstimatesEnabled.hpp"
#include "Time/Tags/StepperErrorTolerances.hpp"
#include "Time/Tags/StepperErrors.hpp"
#include "Time/Tags/TimeStep.hpp"
#include "Time/Tags/TimeStepId.hpp"
#include "Time/Tags/TimeStepper.hpp"
#include "Time/Time.hpp"
#include "Time/TimeStepId.hpp"
#include "Time/TimeSteppers/AdamsBashforth.hpp"
#include "Time/TimeSteppers/Rk3HesthavenSsp.hpp"
#include "Time/TimeSteppers/TimeStepper.hpp"
#include "Time/UpdateU.hpp"
#include "Time/UpdateU.tpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Rational.hpp"
#include "Utilities/TMPL.hpp"

namespace {
struct Var : db::SimpleTag {
  using type = double;
};

struct AlternativeVar : db::SimpleTag {
  using type = double;
};

struct SingleVariableSystem {
  using variables_tag = Var;
};

struct TwoVariableSystem {
  using variables_tag = tmpl::list<Var, AlternativeVar>;
};

template <typename System, bool AlternativeUpdates>
void test_integration() {
  using history_tag = Tags::HistoryEvolvedVariables<Var>;
  using alternative_history_tag = Tags::HistoryEvolvedVariables<AlternativeVar>;

  const Slab slab(1., 3.);
  const TimeStepId initial_id(true, 0, slab.start());
  const TimeDelta time_step = slab.duration() / 2;
  std::unique_ptr<TimeStepper> time_stepper =
      std::make_unique<TimeSteppers::Rk3HesthavenSsp>();

  const auto rhs = [](const auto t, const auto y) {
    return 2. * t - 2. * (y - t * t);
  };

  auto box = db::create<
      db::AddSimpleTags<
          Tags::ConcreteTimeStepper<TimeStepper>, Tags::TimeStepId,
          Tags::Next<Tags::TimeStepId>, Tags::TimeStep,
          ::Tags::StepperErrorEstimatesEnabled, Var, history_tag,
          AlternativeVar, alternative_history_tag,
          Tags::StepperErrorTolerances<Var>, Tags::StepperErrors<Var>,
          Tags::StepperErrorTolerances<AlternativeVar>,
          Tags::StepperErrors<AlternativeVar>>,
      time_stepper_ref_tags<TimeStepper>>(
      std::move(time_stepper), initial_id,
      time_stepper->next_time_id(initial_id, time_step), time_step, false, 1.,
      typename history_tag::type{3}, 1.,
      typename alternative_history_tag::type{3},
      Tags::StepperErrorTolerances<Var>::type{},
      Tags::StepperErrors<Var>::type{},
      Tags::StepperErrorTolerances<AlternativeVar>::type{},
      Tags::StepperErrors<AlternativeVar>::type{});

  // The exact answer is y = x^2, but the integrator would need a
  // smaller step size to get that accurately.
  const std::array<double, 3> expected_values{{3., 3., 10. / 3.}};

  for (size_t substep = 0; substep < 3; ++substep) {
    db::mutate<history_tag, alternative_history_tag>(
        [&rhs](const gsl::not_null<typename history_tag::type*> history,
               const gsl::not_null<typename alternative_history_tag::type*>
                   alternative_history,
               const TimeStepId& time_step_id, const double vars) {
          history->insert(time_step_id, vars,
                          rhs(time_step_id.substep_time(), vars));
          *alternative_history = *history;
        },
        make_not_null(&box), db::get<Tags::TimeStepId>(box), db::get<Var>(box));

    db::mutate_apply<UpdateU<System>>(make_not_null(&box));
    CHECK(db::get<Var>(box) == approx(gsl::at(expected_values, substep)));
    if (AlternativeUpdates) {
      CHECK(db::get<AlternativeVar>(box) ==
            approx(gsl::at(expected_values, substep)));
    } else {
      CHECK(db::get<AlternativeVar>(box) == 1.0);
    }

    db::mutate<Tags::TimeStepId, Tags::Next<Tags::TimeStepId>, history_tag>(
        [&time_step](const gsl::not_null<TimeStepId*> time_step_id,
                     const gsl::not_null<TimeStepId*> next_time_step_id,
                     const gsl::not_null<typename history_tag::type*> history,
                     const TimeStepper& stepper) {
          *time_step_id = *next_time_step_id;
          *next_time_step_id = stepper.next_time_id(*time_step_id, time_step);
          stepper.clean_history(history);
        },
        make_not_null(&box), db::get<Tags::TimeStepper<TimeStepper>>(box));
  }
}

void test_stepper_error() {
  using variables_tag = Var;
  using history_tag = Tags::HistoryEvolvedVariables<variables_tag>;

  const Slab slab(1., 3.);
  const TimeStepId initial_id(true, 0, slab.start());
  const TimeDelta initial_time_step = slab.duration() / 2;
  std::unique_ptr<TimeStepper> time_stepper =
      std::make_unique<TimeSteppers::Rk3HesthavenSsp>();

  auto box = db::create<
      db::AddSimpleTags<Tags::ConcreteTimeStepper<TimeStepper>,
                        Tags::TimeStepId, Tags::Next<Tags::TimeStepId>,
                        Tags::TimeStep, ::Tags::StepperErrorEstimatesEnabled,
                        ::Tags::StepperErrorTolerances<variables_tag>,
                        variables_tag, history_tag,
                        Tags::StepperErrors<variables_tag>>,
      time_stepper_ref_tags<TimeStepper>>(
      std::move(time_stepper), initial_id,
      time_stepper->next_time_id(initial_id, initial_time_step),
      initial_time_step, true,
      StepperErrorTolerances{
          .estimates = StepperErrorTolerances::Estimates::StepperOrder,
          .absolute = 1.0,
          .relative = 0.0},
      1., history_tag::type{3}, Tags::StepperErrors<variables_tag>::type{});

  const auto do_substep = [&box](const bool repeat_substep = false) {
    db::mutate<history_tag>(
        [](const gsl::not_null<typename history_tag::type*> history,
           const TimeStepId& time_step_id,
           const double vars) { history->insert(time_step_id, vars, vars); },
        make_not_null(&box), db::get<Tags::TimeStepId>(box),
        db::get<variables_tag>(box));

    if (repeat_substep) {
      db::mutate_apply<UpdateU<SingleVariableSystem>>(make_not_null(&box));
    }

    db::mutate_apply<UpdateU<SingleVariableSystem>>(make_not_null(&box));

    db::mutate<Tags::TimeStepId, Tags::Next<Tags::TimeStepId>, Tags::TimeStep,
               history_tag>(
        [](const gsl::not_null<TimeStepId*> time_step_id,
           const gsl::not_null<TimeStepId*> next_time_step_id,
           const gsl::not_null<TimeDelta*> time_step,
           const gsl::not_null<typename history_tag::type*> history,
           const TimeStepper& stepper) {
          *time_step_id = *next_time_step_id;
          *time_step = time_step->with_slab(time_step_id->step_time().slab());
          *next_time_step_id = stepper.next_time_id(*time_step_id, *time_step);
          stepper.clean_history(history);
        },
        make_not_null(&box), db::get<Tags::TimeStepper<TimeStepper>>(box));
  };

  using error_tag = Tags::StepperErrors<variables_tag>;
  do_substep();
  CHECK(not db::get<error_tag>(box)[0].has_value());
  CHECK(not db::get<error_tag>(box)[1].has_value());
  do_substep();
  CHECK(not db::get<error_tag>(box)[0].has_value());
  CHECK(not db::get<error_tag>(box)[1].has_value());
  do_substep();
  CHECK(not db::get<error_tag>(box)[0].has_value());
  REQUIRE(db::get<error_tag>(box)[1].has_value());
  CHECK(db::get<error_tag>(box)[1]->step_time == slab.start());

  const auto first_step_errors = db::get<error_tag>(box)[1]->errors;
  const auto second_step = slab.start() + initial_time_step;
  do_substep();
  CHECK(not db::get<error_tag>(box)[0].has_value());
  REQUIRE(db::get<error_tag>(box)[1].has_value());
  CHECK(db::get<error_tag>(box)[1]->step_time == slab.start());
  do_substep();
  CHECK(not db::get<error_tag>(box)[0].has_value());
  REQUIRE(db::get<error_tag>(box)[1].has_value());
  CHECK(db::get<error_tag>(box)[1]->step_time == slab.start());
  do_substep(true);
  REQUIRE(db::get<error_tag>(box)[0].has_value());
  REQUIRE(db::get<error_tag>(box)[1].has_value());
  CHECK(db::get<error_tag>(box)[0]->step_time == slab.start());
  CHECK(db::get<error_tag>(box)[1]->step_time == second_step);
  CHECK(db::get<error_tag>(box)[0]->errors == first_step_errors);
  CHECK(db::get<error_tag>(box)[1]->errors != first_step_errors);
}

void test_errors_for_restart() {
  // We should get low-order errors if we have all of
  //
  // 1. variable-order
  // 2. error-based stepping
  // 3. stepping to the end of a slab.
  //
  // Otherwise, we should get whatever the error-based settings want.

  using Estimates = StepperErrorTolerances::Estimates;

  const auto which_errors = [](const std::optional<size_t>& stepper_order,
                               const StepperErrorTolerances& tolerances,
                               const Rational& step_fraction) {
    using variables_tag = Var;
    using history_tag = Tags::HistoryEvolvedVariables<variables_tag>;

    const Slab slab(1., 3.);
    const TimeStepId initial_id(true, 0, slab.start() + slab.duration() / 2);

    history_tag::type history{3};
    history.insert(TimeStepId(true, 0, slab.start()), 0.0, 0.0);
    history.insert(TimeStepId(true, 0, slab.start() + slab.duration() / 4), 0.0,
                   0.0);
    history.insert(TimeStepId(true, 0, slab.start() + slab.duration() / 2), 0.0,
                   0.0);

    std::unique_ptr<TimeStepper> time_stepper =
        std::make_unique<TimeSteppers::AdamsBashforth>(stepper_order);
    const auto time_step = slab.duration() * step_fraction;

    auto box = db::create<
        db::AddSimpleTags<Tags::ConcreteTimeStepper<TimeStepper>,
                          Tags::TimeStepId, Tags::Next<Tags::TimeStepId>,
                          Tags::TimeStep, ::Tags::StepperErrorEstimatesEnabled,
                          ::Tags::StepperErrorTolerances<variables_tag>,
                          variables_tag, history_tag,
                          Tags::StepperErrors<variables_tag>>,
        time_stepper_ref_tags<TimeStepper>>(
        std::move(time_stepper), initial_id,
        time_stepper->next_time_id(initial_id, time_step), time_step, true,
        tolerances, 1., std::move(history),
        Tags::StepperErrors<variables_tag>::type{});
    db::mutate_apply<UpdateU<SingleVariableSystem>>(make_not_null(&box));
    const auto& errors = db::get<Tags::StepperErrors<variables_tag>>(box)[1];
    if (not errors.has_value()) {
      return Estimates::None;
    }
    return errors->errors[0].has_value() ? Estimates::AllOrders
                                         : Estimates::StepperOrder;
  };

  const StepperErrorTolerances none{};
  const StepperErrorTolerances order{
      .estimates = Estimates::StepperOrder, .absolute = 1.0, .relative = 0.0};
  const StepperErrorTolerances all{
      .estimates = Estimates::AllOrders, .absolute = 1.0, .relative = 0.0};

  CHECK(which_errors(3, none, {1, 4}) == Estimates::None);
  CHECK(which_errors(3, order, {1, 4}) == Estimates::StepperOrder);
  CHECK(which_errors(3, all, {1, 4}) == Estimates::AllOrders);
  CHECK(which_errors(3, none, {1, 2}) == Estimates::None);
  CHECK(which_errors(3, order, {1, 2}) == Estimates::StepperOrder);
  CHECK(which_errors(3, all, {1, 2}) == Estimates::AllOrders);
  CHECK(which_errors(std::nullopt, none, {1, 4}) == Estimates::None);
  CHECK(which_errors(std::nullopt, order, {1, 4}) == Estimates::StepperOrder);
  CHECK(which_errors(std::nullopt, all, {1, 4}) == Estimates::AllOrders);
  CHECK(which_errors(std::nullopt, none, {1, 2}) == Estimates::None);
  // Interesting case:
  CHECK(which_errors(std::nullopt, order, {1, 2}) == Estimates::AllOrders);
  CHECK(which_errors(std::nullopt, all, {1, 2}) == Estimates::AllOrders);
}

SPECTRE_TEST_CASE("Unit.Time.UpdateU", "[Unit][Time]") {
  test_integration<SingleVariableSystem, false>();
  test_integration<TwoVariableSystem, true>();
  test_stepper_error();
  test_errors_for_restart();
}
}  // namespace
