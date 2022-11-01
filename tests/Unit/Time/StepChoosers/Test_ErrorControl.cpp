// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cstddef>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <utility>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/PrefixHelpers.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Domain/Tags.hpp"
#include "Framework/TestCreation.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/DataStructures/MakeWithRandomValues.hpp"
#include "Options/Protocols/FactoryCreation.hpp"
#include "Parallel/GlobalCache.hpp"
#include "Parallel/RegisterDerivedClassesWithCharm.hpp"
#include "Time/History.hpp"
#include "Time/StepChoosers/Constant.hpp"
#include "Time/StepChoosers/ErrorControl.hpp"
#include "Time/StepChoosers/Increase.hpp"
#include "Time/StepChoosers/PreventRapidIncrease.hpp"
#include "Time/StepChoosers/StepChooser.hpp"
#include "Time/Tags.hpp"
#include "Time/TimeStepId.hpp"
#include "Time/TimeSteppers/AdamsBashforthN.hpp"
#include "Time/TimeSteppers/TimeStepper.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/ProtocolHelpers.hpp"
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
  using step_choosers_without_error_control =
      tmpl::list<StepChoosers::Increase<StepChooserUse::LtsStep>,
                 StepChoosers::Constant<StepChooserUse::LtsStep>,
                 StepChoosers::PreventRapidIncrease<StepChooserUse::LtsStep>>;

  struct factory_creation
      : tt::ConformsTo<Options::protocols::FactoryCreation> {
    using factory_classes = tmpl::map<tmpl::pair<
        StepChooser<StepChooserUse::LtsStep>,
        tmpl::conditional_t<
            WithErrorControl,
            tmpl::push_back<step_choosers_without_error_control,
                            StepChoosers::ErrorControl<EvolvedVariablesTag>>,
            step_choosers_without_error_control>>>;
  };

  using component_list = tmpl::list<>;
};

template <typename EvolvedTags>
std::pair<double, bool> get_suggestion(
    const StepChoosers::ErrorControl<EvolvedVariablesTag>& error_control,
    const Variables<EvolvedTags>& step_values,
    const Variables<EvolvedTags>& error,
    const Variables<EvolvedTags>& previous_error, const double previous_step,
    const size_t stepper_order) {
  const Parallel::GlobalCache<Metavariables<true>> cache{};
  TimeSteppers::History<Variables<db::wrap_tags_in<::Tags::dt, EvolvedTags>>>
      history{stepper_order};
  history.insert(TimeStepId{true, 0, {{0.0, 1.0}, {0, 1}}}, 0.1 * step_values);
  auto box =
      db::create<db::AddSimpleTags<
                     Parallel::Tags::MetavariablesImpl<Metavariables<true>>,
                     Tags::HistoryEvolvedVariables<EvolvedVariablesTag>,
                     Tags::RollbackValue<EvolvedVariablesTag>,
                     Tags::StepperError<EvolvedVariablesTag>,
                     Tags::PreviousStepperError<EvolvedVariablesTag>,
                     Tags::StepperErrorUpdated, Tags::TimeStepper<TimeStepper>>,
                 db::AddComputeTags<>>(
          Metavariables<true>{}, std::move(history), step_values, error,
          previous_error, false,
          std::unique_ptr<TimeStepper>{
              std::make_unique<TimeSteppers::AdamsBashforthN>(stepper_order)});

  const auto& time_stepper = get<Tags::TimeStepper<>>(box);
  const std::unique_ptr<StepChooser<StepChooserUse::LtsStep>>
      error_control_base =
          std::make_unique<StepChoosers::ErrorControl<EvolvedVariablesTag>>(
              error_control);

  // check that when the error is not declared updated, the step is accepted and
  // the step is infinity.
  CHECK(std::make_pair(std::numeric_limits<double>::infinity(), true) ==
        error_control(step_values, error, previous_error, false, time_stepper,
                      previous_step, cache));
  CHECK(std::make_pair(std::numeric_limits<double>::infinity(), true) ==
        error_control_base->desired_step(make_not_null(&box), previous_step,
                                         cache));

  db::mutate<Tags::StepperErrorUpdated>(
      make_not_null(&box), [](const gsl::not_null<bool*> stepper_updated) {
        *stepper_updated = true;
      });
  const std::pair<double, bool> result =
      error_control(step_values, error, previous_error, true, time_stepper,
                    previous_step, cache);
  CHECK(error_control_base->desired_step(make_not_null(&box), previous_step,
                                         cache) == result);
  CHECK(serialize_and_deserialize(error_control)(
            step_values, error, previous_error, true, time_stepper,
            previous_step, cache) == result);
  CHECK(serialize_and_deserialize(error_control_base)
            ->desired_step(make_not_null(&box), previous_step, cache) ==
        result);
  return result;
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Time.StepChoosers.ErrorControl", "[Unit][Time]") {
  Parallel::register_factory_classes_with_charm<Metavariables<true>>();

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
      const StepChoosers::ErrorControl<EvolvedVariablesTag> error_control{
          5.0e-4, 0.0, 2.0, 0.5, 0.95};
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
      const StepChoosers::ErrorControl<EvolvedVariablesTag> error_control{
          0.0, 3.0e-4, 2.0, 0.5, 0.95};
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
      const StepChoosers::ErrorControl<EvolvedVariablesTag> error_control{
          4.0e-5, 4.0e-5, 2.0, 0.5, 0.95};
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
      const StepChoosers::ErrorControl<EvolvedVariablesTag> error_control{
          4.0e-5, 4.0e-5, 2.0, 0.9, 0.95};
      const auto first_result = get_suggestion(
          error_control, step_values, step_errors, {}, 1.0, stepper_order);
      // manually calculated in the special case in question: only absolute
      // errors constrained
      CHECK(first_result.first == 0.9);
    }
    {
      INFO("Test error control clamped minimum");
      const StepChoosers::ErrorControl<EvolvedVariablesTag> error_control{
          1.0e-1, 1.0e-1, 2.0, 0.5, 0.95};
      const auto first_result = get_suggestion(
          error_control, step_values, step_errors, {}, 1.0, stepper_order);
      CHECK(first_result == std::make_pair(2.0, true));
    }
  }
  // test option creation
  TestHelpers::test_factory_creation<
      StepChooser<StepChooserUse::LtsStep>,
      StepChoosers::ErrorControl<EvolvedVariablesTag>>(
      "ErrorControl:\n"
      "  SafetyFactor: 0.95\n"
      "  AbsoluteTolerance: 1.0e-5\n"
      "  RelativeTolerance: 1.0e-4\n"
      "  MaxFactor: 2.1\n"
      "  MinFactor: 0.5");
  TestHelpers::test_factory_creation<
      StepChooser<StepChooserUse::LtsStep>,
      StepChoosers::ErrorControl<EvolvedVariablesTag, ErrorControlSelecter>>(
      "ErrorControl(SelectorLabel):\n"
      "  SafetyFactor: 0.95\n"
      "  AbsoluteTolerance: 1.0e-5\n"
      "  RelativeTolerance: 1.0e-4\n"
      "  MaxFactor: 2.1\n"
      "  MinFactor: 0.5");

  CHECK(StepChoosers::ErrorControl<EvolvedVariablesTag, ErrorControlSelecter>{}
            .uses_local_data());

  // Test `IsUsingTimeSteppingErrorControlCompute` tag:
  {
    INFO("IsUsingTimeSteppingErrorControlCompute tag test");
    auto box = db::create<
        db::AddSimpleTags<Tags::StepChoosers>,
        db::AddComputeTags<Tags::IsUsingTimeSteppingErrorControlCompute>>();
    db::mutate<Tags::StepChoosers>(
        make_not_null(&box),
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
        });
    CHECK(db::get<Tags::IsUsingTimeSteppingErrorControlCompute>(box));
    db::mutate<Tags::StepChoosers>(
        make_not_null(&box),
        [](const gsl::not_null<Tags::StepChoosers::type*> choosers) {
          *choosers =
              TestHelpers::test_creation<typename Tags::StepChoosers::type,
                                         Metavariables<true>>(
                  "- PreventRapidIncrease:\n"
                  "- Increase:\n"
                  "    Factor: 2\n"
                  "- Constant: 0.5");
        });
    CHECK_FALSE(db::get<Tags::IsUsingTimeSteppingErrorControlCompute>(box));
    db::mutate<Tags::StepChoosers>(
        make_not_null(&box),
        [](const gsl::not_null<Tags::StepChoosers::type*> choosers) {
          *choosers =
              TestHelpers::test_creation<typename Tags::StepChoosers::type,
                                         Metavariables<true>>("");
        });
    CHECK_FALSE(db::get<Tags::IsUsingTimeSteppingErrorControlCompute>(box));
  }
}
