// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cstddef>
#include <limits>
#include <memory>
#include <string>
#include <utility>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/PrefixHelpers.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "DataStructures/VariablesTag.hpp"
#include "Domain/Tags.hpp"
#include "Framework/TestCreation.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/DataStructures/MakeWithRandomValues.hpp"
#include "Options/Protocols/FactoryCreation.hpp"
#include "ParallelAlgorithms/EventsAndTriggers/Completion.hpp"
#include "ParallelAlgorithms/EventsAndTriggers/Event.hpp"
#include "ParallelAlgorithms/EventsAndTriggers/LogicalTriggers.hpp"
#include "ParallelAlgorithms/EventsAndTriggers/Tags.hpp"
#include "ParallelAlgorithms/EventsAndTriggers/Trigger.hpp"
#include "Time/Actions/ChangeSlabSize.hpp"
#include "Time/History.hpp"
#include "Time/StepChoosers/Constant.hpp"
#include "Time/StepChoosers/ErrorControl.hpp"
#include "Time/StepChoosers/Increase.hpp"
#include "Time/StepChoosers/PreventRapidIncrease.hpp"
#include "Time/StepChoosers/StepChooser.hpp"
#include "Time/Tags.hpp"
#include "Time/TimeStepId.hpp"
#include "Time/TimeSteppers/AdamsBashforth.hpp"
#include "Time/TimeSteppers/TimeStepper.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/ProtocolHelpers.hpp"
#include "Utilities/Serialization/RegisterDerivedClassesWithCharm.hpp"
#include "Utilities/TMPL.hpp"

namespace {
struct EvolvedVar1 : db::SimpleTag {
  using type = Scalar<DataVector>;
};

struct EvolvedVar2 : db::SimpleTag {
  using type = tnsr::i<DataVector, 2>;
};

using EvolvedVariablesTag =
    Tags::Variables<tmpl::list<EvolvedVar1, EvolvedVar2>>;

struct ErrorControlSelecter {
  static std::string name() {
    return "SelectorLabel";
  }
};

template <bool WithErrorControl>
struct Metavariables {
  template <typename Use>
  using step_choosers_without_error_control =
      tmpl::list<StepChoosers::Increase<Use>, StepChoosers::Constant<Use>,
                 StepChoosers::PreventRapidIncrease<Use>>;
  template <typename Use>
  using step_choosers = tmpl::conditional_t<
      WithErrorControl,
      tmpl::push_back<step_choosers_without_error_control<Use>,
                      StepChoosers::ErrorControl<Use, EvolvedVariablesTag>>,
      step_choosers_without_error_control<Use>>;

  struct factory_creation
      : tt::ConformsTo<Options::protocols::FactoryCreation> {
    using factory_classes =
        tmpl::map<tmpl::pair<Event, tmpl::list<Events::ChangeSlabSize,
                                               Events::Completion>>,
                  tmpl::pair<StepChooser<StepChooserUse::LtsStep>,
                             step_choosers<StepChooserUse::LtsStep>>,
                  tmpl::pair<StepChooser<StepChooserUse::Slab>,
                             step_choosers<StepChooserUse::Slab>>,
                  tmpl::pair<Trigger, tmpl::list<Triggers::Always>>>;
  };

  using component_list = tmpl::list<>;
};

using LtsErrorControl =
    StepChoosers::ErrorControl<StepChooserUse::LtsStep, EvolvedVariablesTag>;

template <typename EvolvedTags>
std::pair<double, bool> get_suggestion(
    const LtsErrorControl& error_control,
    const Variables<EvolvedTags>& step_values,
    const Variables<EvolvedTags>& error,
    const Variables<EvolvedTags>& previous_error, const double previous_step,
    const size_t stepper_order) {
  TimeSteppers::History<Variables<EvolvedTags>> history{stepper_order};
  history.insert(TimeStepId{true, 0, {{0.0, 1.0}, {0, 1}}}, step_values,
                 0.1 * step_values);
  history.discard_value(TimeStepId{true, 0, {{0.0, 1.0}, {0, 1}}});
  auto box =
      db::create<db::AddSimpleTags<
                     Parallel::Tags::MetavariablesImpl<Metavariables<true>>,
                     Tags::HistoryEvolvedVariables<EvolvedVariablesTag>,
                     Tags::StepperError<EvolvedVariablesTag>,
                     Tags::PreviousStepperError<EvolvedVariablesTag>,
                     Tags::StepperErrorUpdated, Tags::TimeStepper<TimeStepper>>,
                 db::AddComputeTags<>>(
          Metavariables<true>{}, history, error, previous_error,
          false,
          std::unique_ptr<TimeStepper>{
              std::make_unique<TimeSteppers::AdamsBashforth>(stepper_order)});

  const auto& time_stepper = get<Tags::TimeStepper<>>(box);
  const std::unique_ptr<StepChooser<StepChooserUse::LtsStep>>
      error_control_base = std::make_unique<LtsErrorControl>(error_control);

  // check that when the error is not declared updated, the step is accepted and
  // the step is infinity.
  CHECK(std::make_pair(std::numeric_limits<double>::infinity(), true) ==
        error_control(history, error, previous_error, false, time_stepper,
                      previous_step));
  CHECK(std::make_pair(std::numeric_limits<double>::infinity(), true) ==
        error_control_base->desired_step(previous_step, box));

  db::mutate<Tags::StepperErrorUpdated>(
      [](const gsl::not_null<bool*> stepper_updated) {
        *stepper_updated = true;
      },
      make_not_null(&box));
  const std::pair<double, bool> result = error_control(
      history, error, previous_error, true, time_stepper, previous_step);
  CHECK(error_control_base->desired_step(previous_step, box) == result);
  CHECK(serialize_and_deserialize(error_control)(
            history, error, previous_error, true, time_stepper,
            previous_step) == result);
  CHECK(serialize_and_deserialize(error_control_base)
            ->desired_step(previous_step, box) == result);
  return result;
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Time.StepChoosers.ErrorControl", "[Unit][Time]") {
  register_factory_classes_with_charm<Metavariables<true>>();

  MAKE_GENERATOR(generator);
  std::uniform_real_distribution<> step_dist{1.0, 2.0};
  Variables<tmpl::list<EvolvedVar1, EvolvedVar2>> step_values{5};
  fill_with_random_values(make_not_null(&step_values),
                          make_not_null(&generator), make_not_null(&step_dist));
  const DataVector flattened_step_values{step_values.data(), 15};
  std::uniform_real_distribution<> error_dist{1.0e-4, 2.0e-4};
  Variables<tmpl::list<EvolvedVar1, EvolvedVar2>> step_errors{5};
  fill_with_random_values(make_not_null(&step_errors),
                          make_not_null(&generator),
                          make_not_null(&error_dist));
  const DataVector flattened_step_errors{step_errors.data(), 15};
  const std::vector<size_t> stepper_orders{2_st, 5_st};
  for (size_t stepper_order : stepper_orders) {
    CAPTURE(stepper_order);
    {
      INFO("Test error control step fixed by absolute tolerance");
      const LtsErrorControl error_control{5.0e-4, 0.0, 2.0, 0.5, 0.95};
      const auto first_result = get_suggestion(
          error_control, step_values, step_errors, {}, 1.0, stepper_order);
      // manually calculated in the special case in question: only absolute
      // errors constrained
      const double expected_linf_error = max(flattened_step_errors) / 5.0e-4;
      CHECK(approx(first_result.first) ==
            0.95 / pow(expected_linf_error, 1.0 / stepper_order));
      CHECK(first_result.second);
      const auto second_result = get_suggestion(
          error_control, step_values, decltype(step_errors){1.2 * step_errors},
          step_errors, first_result.first, stepper_order);
      CHECK(approx(second_result.first) ==
            0.95 * first_result.first /
                (pow(expected_linf_error, 0.3 / stepper_order) *
                 pow(1.2, 0.7 / stepper_order)));
      CHECK(second_result.first);
      // Check that the suggested step size is smaller if the error in
      // increasing faster.
      const auto adjusted_second_result = get_suggestion(
          error_control, step_values, decltype(step_errors){1.2 * step_errors},
          decltype(step_errors){0.5 * step_errors}, first_result.first,
          stepper_order);
      CHECK(adjusted_second_result.first < second_result.first);
    }
    {
      INFO("Test error control step fixed by relative tolerance");
      const LtsErrorControl error_control{0.0, 3.0e-4, 2.0, 0.5, 0.95};
      const auto first_result = get_suggestion(
          error_control, step_values, step_errors, {}, 1.0, stepper_order);
      // manually calculated in the special case in question: only relative
      // errors constrained
      const double expected_linf_error =
          max(flattened_step_errors /
              (flattened_step_values + flattened_step_errors)) /
          3.0e-4;
      CHECK(approx(first_result.first) ==
            0.95 / pow(expected_linf_error, 1.0 / stepper_order));
      CHECK(first_result.second);
      const auto second_result = get_suggestion(
          error_control, step_values, decltype(step_errors){1.2 * step_errors},
          step_errors, first_result.first, stepper_order);
      const double new_expected_linf_error =
          max(1.2 * flattened_step_errors /
              (flattened_step_values + 1.2 * flattened_step_errors)) /
          3.0e-4;
      CHECK(approx(second_result.first) ==
            0.95 * first_result.first *
                pow(pow(new_expected_linf_error, 1.0 / stepper_order), -0.7) *
                pow(pow(expected_linf_error, 1.0 / stepper_order), 0.4));
      CHECK(second_result.first);
    }
    {
      INFO("Test error control step failure");
      const LtsErrorControl error_control{4.0e-5, 4.0e-5, 2.0, 0.5, 0.95};
      const auto first_result = get_suggestion(
          error_control, step_values, step_errors, {}, 1.0, stepper_order);
      // manually calculated in the special case in question: only absolute
      // errors constrained
      const double expected_linf_error =
          max(flattened_step_errors /
              (flattened_step_values + flattened_step_errors + 1.0)) /
          4.0e-5;
      CHECK(
          approx(first_result.first) ==
          std::max(0.95 / pow(expected_linf_error, 1.0 / stepper_order), 0.5));
      CHECK_FALSE(first_result.second);
    }
    {
      INFO("Test error control clamped minimum");
      const LtsErrorControl error_control{4.0e-5, 4.0e-5, 2.0, 0.9, 0.95};
      const auto first_result = get_suggestion(
          error_control, step_values, step_errors, {}, 1.0, stepper_order);
      // manually calculated in the special case in question: only absolute
      // errors constrained
      CHECK(first_result.first == 0.9);
    }
    {
      INFO("Test error control clamped minimum");
      const LtsErrorControl error_control{1.0e-1, 1.0e-1, 2.0, 0.5, 0.95};
      const auto first_result = get_suggestion(
          error_control, step_values, step_errors, {}, 1.0, stepper_order);
      CHECK(first_result == std::make_pair(2.0, true));
    }
  }
  // test option creation
  TestHelpers::test_factory_creation<
      StepChooser<StepChooserUse::LtsStep>,
      StepChoosers::ErrorControl<StepChooserUse::LtsStep, EvolvedVariablesTag>>(
      "ErrorControl:\n"
      "  SafetyFactor: 0.95\n"
      "  AbsoluteTolerance: 1.0e-5\n"
      "  RelativeTolerance: 1.0e-4\n"
      "  MaxFactor: 2.1\n"
      "  MinFactor: 0.5");
  TestHelpers::test_factory_creation<
      StepChooser<StepChooserUse::LtsStep>,
      StepChoosers::ErrorControl<StepChooserUse::LtsStep, EvolvedVariablesTag,
                                 ErrorControlSelecter>>(
      "ErrorControl(SelectorLabel):\n"
      "  SafetyFactor: 0.95\n"
      "  AbsoluteTolerance: 1.0e-5\n"
      "  RelativeTolerance: 1.0e-4\n"
      "  MaxFactor: 2.1\n"
      "  MinFactor: 0.5");

  CHECK(StepChoosers::ErrorControl<StepChooserUse::LtsStep, EvolvedVariablesTag,
                                   ErrorControlSelecter>{}
            .uses_local_data());

  // Test `IsUsingTimeSteppingErrorControlCompute` tag:
  {
    INFO("IsUsingTimeSteppingErrorControlCompute tag LTS test");
    auto box =
        db::create<db::AddSimpleTags<Tags::StepChoosers>,
                   db::AddComputeTags<
                       Tags::IsUsingTimeSteppingErrorControlCompute<true>>>();
    db::mutate<Tags::StepChoosers>(
        [](const gsl::not_null<Tags::StepChoosers::type*> choosers) {
          *choosers =
              TestHelpers::test_creation<typename Tags::StepChoosers::type,
                                         Metavariables<true>>(
                  "- ErrorControl:\n"
                  "    SafetyFactor: 0.95\n"
                  "    AbsoluteTolerance: 1.0e-5\n"
                  "    RelativeTolerance: 1.0e-4\n"
                  "    MaxFactor: 2.1\n"
                  "    MinFactor: 0.5\n"
                  "- Increase:\n"
                  "    Factor: 2\n"
                  "- Constant: 0.5");
        },
        make_not_null(&box));
    CHECK(db::get<Tags::IsUsingTimeSteppingErrorControl>(box));
    db::mutate<Tags::StepChoosers>(
        [](const gsl::not_null<Tags::StepChoosers::type*> choosers) {
          *choosers =
              TestHelpers::test_creation<typename Tags::StepChoosers::type,
                                         Metavariables<true>>(
                  "- PreventRapidIncrease:\n"
                  "- Increase:\n"
                  "    Factor: 2\n"
                  "- Constant: 0.5");
        },
        make_not_null(&box));
    CHECK_FALSE(db::get<Tags::IsUsingTimeSteppingErrorControl>(box));
    db::mutate<Tags::StepChoosers>(
        [](const gsl::not_null<Tags::StepChoosers::type*> choosers) {
          *choosers =
              TestHelpers::test_creation<typename Tags::StepChoosers::type,
                                         Metavariables<true>>("");
        },
        make_not_null(&box));
    CHECK_FALSE(db::get<Tags::IsUsingTimeSteppingErrorControl>(box));
  }
  {
    INFO("IsUsingTimeSteppingErrorControlCompute tag GTS test");
    auto box =
        db::create<db::AddSimpleTags<Tags::EventsAndTriggers>,
                   db::AddComputeTags<
                       Tags::IsUsingTimeSteppingErrorControlCompute<false>>>();
    db::mutate<Tags::EventsAndTriggers>(
        [](const gsl::not_null<Tags::EventsAndTriggers::type*> events) {
          *events = TestHelpers::test_creation<Tags::EventsAndTriggers::type,
                                               Metavariables<true>>(
              "- Trigger: Always\n"
              "  Events:\n"
              "    - Completion\n"
              "- Trigger: Always\n"
              "  Events:\n"
              "    - Completion\n"
              "    - ChangeSlabSize:\n"
              "        DelayChange: 0\n"
              "        StepChoosers:\n"
              "          - Increase:\n"
              "              Factor: 2\n"
              "          - ErrorControl:\n"
              "              SafetyFactor: 0.95\n"
              "              AbsoluteTolerance: 1.0e-5\n"
              "              RelativeTolerance: 1.0e-4\n"
              "              MaxFactor: 2.1\n"
              "              MinFactor: 0.5\n"
              "          - Constant: 0.5");
        },
        make_not_null(&box));
    CHECK(db::get<Tags::IsUsingTimeSteppingErrorControl>(box));
    db::mutate<Tags::EventsAndTriggers>(
        [](const gsl::not_null<Tags::EventsAndTriggers::type*> events) {
          *events = TestHelpers::test_creation<Tags::EventsAndTriggers::type,
                                               Metavariables<true>>(
              "- Trigger: Always\n"
              "  Events:\n"
              "    - Completion\n"
              "- Trigger: Always\n"
              "  Events:\n"
              "    - Completion\n"
              "    - ChangeSlabSize:\n"
              "        DelayChange: 0\n"
              "        StepChoosers:\n"
              "          - Increase:\n"
              "              Factor: 2\n"
              "          - Constant: 0.5");
        },
        make_not_null(&box));
    CHECK_FALSE(db::get<Tags::IsUsingTimeSteppingErrorControl>(box));
    db::mutate<Tags::EventsAndTriggers>(
        [](const gsl::not_null<Tags::EventsAndTriggers::type*> events) {
          *events = TestHelpers::test_creation<Tags::EventsAndTriggers::type,
                                               Metavariables<true>>(
              "- Trigger: Always\n"
              "  Events:\n"
              "    - Completion\n"
              "- Trigger: Always\n"
              "  Events:\n"
              "    - Completion");
        },
        make_not_null(&box));
    CHECK_FALSE(db::get<Tags::IsUsingTimeSteppingErrorControl>(box));
    db::mutate<Tags::EventsAndTriggers>(
        [](const gsl::not_null<Tags::EventsAndTriggers::type*> events) {
          *events = TestHelpers::test_creation<Tags::EventsAndTriggers::type,
                                               Metavariables<true>>("");
        },
        make_not_null(&box));
    CHECK_FALSE(db::get<Tags::IsUsingTimeSteppingErrorControl>(box));
  }
}
