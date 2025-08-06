// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <cstddef>
#include <iomanip>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/MetavariablesTag.hpp"
#include "DataStructures/DataBox/PrefixHelpers.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/TaggedVariant.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "DataStructures/VariablesTag.hpp"
#include "Framework/TestCreation.hpp"
#include "Helpers/DataStructures/TestTags.hpp"
#include "Options/Protocols/FactoryCreation.hpp"
#include "Parallel/Printf/Printf.hpp"
#include "ParallelAlgorithms/EventsAndTriggers/EventsAndTriggers.hpp"
#include "ParallelAlgorithms/EventsAndTriggers/Tags.hpp"
#include "Time/AdaptiveSteppingDiagnostics.hpp"
#include "Time/ChangeTimeStepperOrder.hpp"
#include "Time/ChooseLtsStepSize.hpp"
#include "Time/Slab.hpp"
#include "Time/StepChoosers/ErrorControl.hpp"
#include "Time/StepChoosers/StepChooser.hpp"
#include "Time/StepperErrorEstimate.hpp"
#include "Time/Tags/AdaptiveSteppingDiagnostics.hpp"
#include "Time/Tags/HistoryEvolvedVariables.hpp"
#include "Time/Tags/MinimumTimeStep.hpp"
#include "Time/Tags/StepChoosers.hpp"
#include "Time/Tags/StepperErrorTolerancesCompute.hpp"
#include "Time/Tags/StepperErrors.hpp"
#include "Time/Tags/Time.hpp"
#include "Time/Tags/TimeStep.hpp"
#include "Time/Tags/TimeStepId.hpp"
#include "Time/Tags/TimeStepper.hpp"
#include "Time/Tags/VariableOrderAlgorithm.hpp"
#include "Time/TakeStep.hpp"
#include "Time/Time.hpp"
#include "Time/TimeStepId.hpp"
#include "Time/TimeSteppers/AdamsBashforth.hpp"
#include "Time/TimeSteppers/AdamsMoultonPc.hpp"
#include "Time/TimeSteppers/LtsTimeStepper.hpp"
#include "Time/TimeSteppers/TimeStepper.hpp"
#include "Time/VariableOrderAlgorithm.hpp"
#include "Utilities/Algorithm.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/FileSystem.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeString.hpp"
#include "Utilities/ProtocolHelpers.hpp"
#include "Utilities/TMPL.hpp"

namespace {
using Vars1 = Tags::Variables<tmpl::list<TestHelpers::Tags::Vector<>>>;
using Vars2 = Tags::Variables<tmpl::list<TestHelpers::Tags::Scalar<>>>;

struct System1 {
  using variables_tag = Vars1;
  static constexpr bool two_vars = false;
};

struct System2 {
  using variables_tag = tmpl::list<Vars1, Vars2>;
  static constexpr bool two_vars = true;
};

template <typename System>
size_t test_system(VariableOrderAlgorithm algorithm, const size_t initial_order,
                   std::optional<StepperErrorEstimate> errors1,
                   std::optional<StepperErrorEstimate> errors2) {
  const Slab slab(0.0, 1.0);

  // First entry is for the previous step, which is only used for step
  // size control, not order.
  std::array<std::optional<StepperErrorEstimate>, 2> stepper_errors1{};
  std::array<std::optional<StepperErrorEstimate>, 2> stepper_errors2{};
  stepper_errors1[1] = std::move(errors1);
  stepper_errors2[1] = std::move(errors2);

  using history_tag1 = Tags::HistoryEvolvedVariables<Vars1>;
  using history_tag2 = Tags::HistoryEvolvedVariables<Vars2>;

  const TimeStepId next_time_step_id(true, 0, slab.end());

  auto box = db::create<
      db::AddSimpleTags<
          Tags::ConcreteTimeStepper<TimeStepper>, Tags::VariableOrderAlgorithm,
          history_tag1, history_tag2, Tags::StepperErrors<Vars1>,
          Tags::StepperErrors<Vars2>, Tags::Next<Tags::TimeStepId>>,
      time_stepper_ref_tags<TimeStepper>>(
      static_cast<std::unique_ptr<TimeStepper>>(
          std::make_unique<TimeSteppers::AdamsBashforth>(initial_order)),
      std::move(algorithm), history_tag1::type{initial_order},
      history_tag2::type{initial_order}, stepper_errors1, stepper_errors2,
      next_time_step_id);

  // Does nothing for fixed-order
  db::mutate_apply<ChangeTimeStepperOrder<System>>(make_not_null(&box));
  CHECK(db::get<history_tag1>(box).integration_order() == initial_order);
  CHECK(db::get<history_tag2>(box).integration_order() == initial_order);

  db::mutate<Tags::ConcreteTimeStepper<TimeStepper>,
             Tags::Next<Tags::TimeStepId>>(
      [&](const gsl::not_null<std::unique_ptr<TimeStepper>*> stepper,
          const gsl::not_null<TimeStepId*> id) {
        *stepper = make_unique<TimeSteppers::AdamsBashforth>(std::nullopt);
        *id = TimeStepId(true, 0, slab.start(), 1, slab.duration(), 1.0);
      },
      make_not_null(&box));

  // Does nothing on a substep
  db::mutate_apply<ChangeTimeStepperOrder<System>>(make_not_null(&box));
  CHECK(db::get<history_tag1>(box).integration_order() == initial_order);
  CHECK(db::get<history_tag2>(box).integration_order() == initial_order);

  db::mutate<Tags::Next<Tags::TimeStepId>>(
      [&](const gsl::not_null<TimeStepId*> id) {
        *id = TimeStepId(true, 0, slab.end());
      },
      make_not_null(&box));
  db::mutate_apply<ChangeTimeStepperOrder<System>>(make_not_null(&box));

  if constexpr (System::two_vars) {
    CHECK(db::get<history_tag2>(box).integration_order() ==
          db::get<history_tag1>(box).integration_order());
  } else {
    CHECK(db::get<history_tag2>(box).integration_order() == initial_order);
  }
  return db::get<history_tag1>(box).integration_order();
}

size_t goal_order(const size_t goal, const size_t initial_order) {
  const VariableOrderAlgorithm goal_algorithm{goal};
  const auto algorithm_from_options =
      TestHelpers::test_creation<VariableOrderAlgorithm>(
          MakeString{} << "GoalOrder: " << goal);
  CHECK(algorithm_from_options == goal_algorithm);
  CHECK(goal_algorithm.required_estimates() ==
        StepperErrorTolerances::Estimates::None);

  const size_t result = test_system<System1>(goal_algorithm, initial_order,
                                             std::nullopt, std::nullopt);
  CHECK(test_system<System2>(goal_algorithm, initial_order, std::nullopt,
                             std::nullopt) == result);
  CHECK(test_system<System1>(algorithm_from_options, initial_order,
                             std::nullopt, std::nullopt) == result);
  CHECK(test_system<System2>(algorithm_from_options, initial_order,
                             std::nullopt, std::nullopt) == result);
  return result;
}

void test_goal() {
  CHECK(goal_order(4, 1) == 2);
  CHECK(goal_order(4, 2) == 3);
  CHECK(goal_order(4, 3) == 4);
  CHECK(goal_order(4, 4) == 4);
  CHECK(goal_order(4, 5) == 4);
  CHECK(goal_order(4, 6) == 5);
  CHECK(goal_order(1, 1) == 1);
}

size_t falloff_order(const double falloff, const std::vector<double>& errors) {
  const VariableOrderAlgorithm falloff_algorithm{falloff};
  const auto algorithm_from_options =
      TestHelpers::test_creation<VariableOrderAlgorithm>(
          MakeString{} << std::setprecision(18) << "OrderFalloff: " << falloff);
  CHECK(algorithm_from_options == falloff_algorithm);
  CHECK(falloff_algorithm.required_estimates() ==
        StepperErrorTolerances::Estimates::AllOrders);

  const Slab slab(0.0, 1.0);
  // Time and step size are unused.
  StepperErrorEstimate falloff_errors(slab.start(), slab.duration(),
                                      errors.size() - 1, errors.back());
  for (size_t i = 0; i < errors.size() - 1; ++i) {
    gsl::at(falloff_errors.errors, i).emplace(errors[i]);
  }

  const double min_error = *alg::min_element(errors);
  StepperErrorEstimate constant_errors(slab.start(), slab.duration(),
                                       errors.size() - 1, min_error);
  for (size_t i = 0; i < errors.size() - 1; ++i) {
    gsl::at(constant_errors.errors, i).emplace(min_error);
  }

  const size_t result = test_system<System1>(falloff_algorithm, errors.size(),
                                             falloff_errors, std::nullopt);
  CHECK(test_system<System2>(falloff_algorithm, errors.size(), falloff_errors,
                             falloff_errors) == result);
  CHECK(test_system<System2>(falloff_algorithm, errors.size(), falloff_errors,
                             constant_errors) == result);
  CHECK(test_system<System2>(falloff_algorithm, errors.size(), constant_errors,
                             falloff_errors) == result);
  CHECK(test_system<System2>(falloff_algorithm, errors.size(), falloff_errors,
                             std::nullopt) == result);
  CHECK(test_system<System2>(falloff_algorithm, errors.size(), std::nullopt,
                             falloff_errors) == result);
  return result;
}

void test_falloff_without_errors() {
  const size_t initial_order = 4;
  const Slab slab(0.0, 1.0);

  std::array<std::optional<StepperErrorEstimate>, 2> stepper_errors{};

  using history_tag = Tags::HistoryEvolvedVariables<Vars1>;

  const TimeStepId next_time_step_id(true, 0, slab.end());

  auto box =
      db::create<db::AddSimpleTags<Tags::ConcreteTimeStepper<TimeStepper>,
                                   Tags::VariableOrderAlgorithm, history_tag,
                                   Tags::StepperErrors<Vars1>,
                                   Tags::Next<Tags::TimeStepId>>,
                 time_stepper_ref_tags<TimeStepper>>(
          static_cast<std::unique_ptr<TimeStepper>>(
              std::make_unique<TimeSteppers::AdamsBashforth>(initial_order)),
          VariableOrderAlgorithm{0.1}, history_tag::type{initial_order},
          stepper_errors, next_time_step_id);

  // Does nothing for fixed-order
  db::mutate_apply<ChangeTimeStepperOrder<System1>>(make_not_null(&box));
  CHECK(db::get<history_tag>(box).integration_order() == initial_order);

  db::mutate<Tags::ConcreteTimeStepper<TimeStepper>>(
      [&](const gsl::not_null<std::unique_ptr<TimeStepper>*> stepper) {
        *stepper = make_unique<TimeSteppers::AdamsBashforth>(std::nullopt);
      },
      make_not_null(&box));

  CHECK_THROWS_WITH(
      db::mutate_apply<ChangeTimeStepperOrder<System1>>(make_not_null(&box)),
      Catch::Matchers::ContainsSubstring(
          "OrderFalloff only implemented with error-based adaptive time "
          "stepping."));
}

void test_falloff() {
  // Euler's method should always increase to second order
  CHECK(falloff_order(0.1, {3.0}) == 2);

  // The algorithm can't decrease from second to first order, and will
  // only stay the same if second is actually worse.
  CHECK(falloff_order(0.1, {0.9, 0.899}) == 3);
  CHECK(falloff_order(0.1, {0.9, 0.901}) == 2);

  // Max order can't increase
  CHECK(falloff_order(0.1, {1.0e-1, 1.0e-2, 1.0e-3, 1.0e-4, 1.0e-5, 1.0e-6,
                            1.0e-7, 1.0e-8}) == 8);

  // Typical cases
  // Make the largest drop a factor of 4, falloff 0.5 then requires
  // factor of 2 decrease.
  CHECK(falloff_order(0.5, {0.9, 0.8, 0.2, 0.15, 1.0e-5}) == 4);
  CHECK(falloff_order(0.5, {0.9, 0.8, 0.2, 0.08, 0.041}) == 5);
  CHECK(falloff_order(0.5, {0.9, 0.8, 0.2, 0.08, 0.039}) == 6);
  CHECK(falloff_order(0.51, {0.9, 0.8, 0.2, 0.08, 0.04}) == 5);
  CHECK(falloff_order(0.49, {0.9, 0.8, 0.2, 0.08, 0.04}) == 6);

  test_falloff_without_errors();
}

// See Hairer II.4
//
// Evolves the system
// y1' = 1 + y1^2 y2 - 4 y1
// y2' = 3 y1 - y1^2 y2
//
// This is a popular test for various sorts of adaptive stepping
// because the solution has a sawtooth-like shape, with gradual
// changes alternating with rapid ones.
namespace brusselator {
struct Var1 : db::SimpleTag {
  using type = Scalar<DataVector>;
};

struct Var2 : db::SimpleTag {
  using type = Scalar<DataVector>;
};

struct System {
  using variables_tag = ::Tags::Variables<tmpl::list<Var1, Var2>>;

  struct compute_time_derivative {
    using return_tags = tmpl::list<::Tags::dt<Var1>, ::Tags::dt<Var2>>;
    using argument_tags = tmpl::list<Var1, Var2>;
    static void apply(const gsl::not_null<Scalar<DataVector>*> dt_var1,
                      const gsl::not_null<Scalar<DataVector>*> dt_var2,
                      const Scalar<DataVector>& var1,
                      const Scalar<DataVector>& var2) {
      tenex::evaluate(dt_var1, 1.0 + square(var1()) * var2() - 4.0 * var1());
      tenex::evaluate(dt_var2, 3.0 * var1() - square(var1()) * var2());
    }
  };
};

using ErrorControl =
    StepChoosers::ErrorControl<StepChooserUse::LtsStep, System::variables_tag>;

struct Metavariables {
  using system = System;

  struct factory_creation
      : tt::ConformsTo<Options::protocols::FactoryCreation> {
    using factory_classes =
        tmpl::map<tmpl::pair<StepChooser<StepChooserUse::LtsStep>,
                             tmpl::list<ErrorControl>>>;
  };
};

double run(std::unique_ptr<LtsTimeStepper> time_stepper, const double tolerance,
           const double initial_step_size,
           const std::optional<std::string>& output_file = {}) {
  // Prevent people from forgetting to turn this off after testing.
  CHECK(not output_file.has_value());

  const Slab slab(0.0, 20.0);

  const TimeStepId initial_time_step_id(true, 0, slab.start());
  const TimeDelta initial_time_step =
      choose_lts_step_size(slab.start(), initial_step_size);

  std::vector<std::unique_ptr<StepChooser<StepChooserUse::LtsStep>>>
      step_choosers{};
  step_choosers.push_back(
      std::make_unique<ErrorControl>(tolerance, tolerance, 2.0, 0.25, 0.95));

  const double minimum_time_step = 1.0e-7;

  System::variables_tag::type initial_vars{1};
  get(get<Var1>(initial_vars)) = 1.5;
  get(get<Var2>(initial_vars)) = 3.0;

  const auto stepper_order = time_stepper->order();
  REQUIRE(holds_alternative<TimeSteppers::Tags::VariableOrder>(stepper_order));
  const size_t initial_order =
      get<TimeSteppers::Tags::VariableOrder>(stepper_order).minimum;

  using dt_variables_tag =
      db::add_tag_prefix<::Tags::dt, System::variables_tag>;
  using history_tag = ::Tags::HistoryEvolvedVariables<System::variables_tag>;

  auto box = db::create<
      db::AddSimpleTags<
          ::Parallel::Tags::MetavariablesImpl<Metavariables>,
          ::Tags::ConcreteTimeStepper<LtsTimeStepper>,
          ::Tags::EventsAndTriggers<Triggers::WhenToCheck::AtSlabs>,
          Tags::VariableOrderAlgorithm, ::Tags::TimeStepId,
          ::Tags::Next<::Tags::TimeStepId>, ::Tags::TimeStep, ::Tags::Time,
          ::Tags::AdaptiveSteppingDiagnostics, ::Tags::StepChoosers,
          ::Tags::MinimumTimeStep, System::variables_tag, dt_variables_tag,
          history_tag, ::Tags::StepperErrors<System::variables_tag>>,
      tmpl::push_back<
          time_stepper_ref_tags<LtsTimeStepper>,
          ::Tags::StepperErrorEstimatesEnabledCompute<true>,
          ::Tags::StepperErrorTolerancesCompute<System::variables_tag, true>>>(
      Metavariables{}, std::move(time_stepper), EventsAndTriggers{},
      VariableOrderAlgorithm(0.1), initial_time_step_id,
      time_stepper->next_time_id(initial_time_step_id, initial_time_step),
      initial_time_step, slab.start().value(), ::AdaptiveSteppingDiagnostics{},
      std::move(step_choosers), minimum_time_step, std::move(initial_vars),
      dt_variables_tag::type{1}, history_tag::type{initial_order},
      ::Tags::StepperErrors<System::variables_tag>::type{});

  if (output_file.has_value()) {
    ::file_system::rm(*output_file, false);
  }

  size_t order_sum = 0;
  size_t num_steps = 0;

  for (;;) {
    if (output_file.has_value() and
        db::get<::Tags::TimeStepId>(box).substep() == 0) {
      Parallel::fprintf(
          *output_file, "%zu\t%.16g\t%.16g\t%.16g\t%.16g\t%zu\n",
          static_cast<size_t>(db::get<::Tags::AdaptiveSteppingDiagnostics>(box)
                                  .number_of_steps),
          db::get<::Tags::Time>(box), get(db::get<Var1>(box))[0],
          get(db::get<Var2>(box))[0], db::get<::Tags::TimeStep>(box).value(),
          static_cast<size_t>(db::get<history_tag>(box).integration_order()));
    }
    if (db::get<::Tags::TimeStepId>(box).slab_number() != 0) {
      break;
    }

    order_sum += db::get<history_tag>(box).integration_order();
    ++num_steps;

    db::mutate_apply<System::compute_time_derivative>(make_not_null(&box));
    take_step<System, true>(make_not_null(&box));
    db::mutate_apply<ChangeTimeStepperOrder<System>>(make_not_null(&box));

    db::mutate<::Tags::TimeStepId, ::Tags::Next<::Tags::TimeStepId>,
               ::Tags::Time, ::Tags::AdaptiveSteppingDiagnostics, history_tag>(
        [](const gsl::not_null<TimeStepId*> time_step_id,
           const gsl::not_null<TimeStepId*> next_time_step_id,
           const gsl::not_null<double*> time,
           const gsl::not_null<AdaptiveSteppingDiagnostics*> diags,
           const gsl::not_null<history_tag::type*> history,
           const TimeStepper& local_time_stepper, const TimeDelta& time_step) {
          if (next_time_step_id->substep() == 0) {
            ++diags->number_of_steps;
          }
          *time_step_id = *next_time_step_id;
          *next_time_step_id = local_time_stepper.next_time_id(
              *next_time_step_id,
              time_step.with_slab(time_step_id->step_time().slab()));
          *time = time_step_id->substep_time();
          local_time_stepper.clean_history(history);
        },
        make_not_null(&box), db::get<::Tags::TimeStepper<TimeStepper>>(box),
        db::get<::Tags::TimeStep>(box));
  }

  return static_cast<double>(order_sum) / static_cast<double>(num_steps);
}

void test() {
  const auto average_order_ab_3 =
      run(std::make_unique<TimeSteppers::AdamsBashforth>(std::nullopt), 1.0e-3,
          1.0e-3);
  const auto average_order_ab_8 =
      run(std::make_unique<TimeSteppers::AdamsBashforth>(std::nullopt), 1.0e-8,
          1.0e-6);
  const auto average_order_am_3 =
      run(std::make_unique<TimeSteppers::AdamsMoultonPc<true>>(std::nullopt),
          1.0e-3, 1.0e-3);
  const auto average_order_am_8 =
      run(std::make_unique<TimeSteppers::AdamsMoultonPc<true>>(std::nullopt),
          1.0e-8, 1.0e-6);

  CHECK(average_order_ab_3 < average_order_ab_8);
  CHECK(average_order_am_3 < average_order_am_8);
}
}  // namespace brusselator

SPECTRE_TEST_CASE("Unit.Time.ChangeTimeStepperOrder", "[Unit][Time]") {
  test_goal();
  test_falloff();
  brusselator::test();
}
}  // namespace
