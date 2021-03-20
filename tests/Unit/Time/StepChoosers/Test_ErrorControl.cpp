// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cstddef>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <utility>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Domain/Tags.hpp"
#include "Framework/TestCreation.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/DataStructures/MakeWithRandomValues.hpp"
#include "Parallel/GlobalCache.hpp"
#include "Parallel/RegisterDerivedClassesWithCharm.hpp"
#include "Time/StepChoosers/Constant.hpp"
#include "Time/StepChoosers/ErrorControl.hpp"
#include "Time/StepChoosers/Increase.hpp"
#include "Time/StepChoosers/PreventRapidIncrease.hpp"
#include "Time/StepChoosers/StepChooser.hpp"
#include "Time/Tags.hpp"
#include "Time/TimeSteppers/AdamsBashforthN.hpp"
#include "Time/TimeSteppers/TimeStepper.hpp"
#include "Utilities/Gsl.hpp"
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

using StepChooserType = StepChooser<
    tmpl::list<StepChoosers::Registrars::ErrorControl<EvolvedVariablesTag>>>;

struct Metavariables {
  using component_list = tmpl::list<>;
};

template <typename EvolvedTags, typename ErrorTags>
std::pair<double, bool> get_suggestion(
    const gsl::not_null<std::optional<double>*> previous_step_error,
    const StepChoosers::ErrorControl<EvolvedVariablesTag>& error_control,
    const Variables<EvolvedTags>& step_values,
    const Variables<ErrorTags>& error, const double previous_step,
    const size_t stepper_order) noexcept {
  const Parallel::GlobalCache<Metavariables> cache{};
  TimeSteppers::History<
      Variables<EvolvedTags>,
      typename db::add_tag_prefix<::Tags::dt, EvolvedVariablesTag>::type>
      history{stepper_order};
  history.insert(TimeStepId{true, 0, {{0.0, 1.0}, {0, 1}}}, step_values,
                 0.1 * step_values);
  auto box = db::create<
      db::AddSimpleTags<
          Tags::HistoryEvolvedVariables<EvolvedVariablesTag>,
          db::add_tag_prefix<Tags::StepperError, EvolvedVariablesTag>,
          Tags::StepperErrorUpdated, Tags::TimeStepper<TimeStepper>,
          StepChoosers::Tags::PreviousStepError>,
      db::AddComputeTags<>>(
      std::move(history), error, false,
      std::unique_ptr<TimeStepper>{
          std::make_unique<TimeSteppers::AdamsBashforthN>(stepper_order)},
      *previous_step_error);

  const auto& time_stepper = get<Tags::TimeStepper<>>(box);
  const std::unique_ptr<StepChooserType> error_control_base =
      std::make_unique<StepChoosers::ErrorControl<EvolvedVariablesTag>>(
          error_control);

  // check that when the error is not declared updated, the step is accepted and
  // the step is infinity.
  CHECK(std::make_pair(std::numeric_limits<double>::infinity(), true) ==
        error_control(
            previous_step_error,
            db::get<Tags::HistoryEvolvedVariables<EvolvedVariablesTag>>(box),
            error, false, time_stepper, previous_step, cache));
  CHECK(*previous_step_error ==
        db::get<StepChoosers::Tags::PreviousStepError>(box));
  CHECK(std::make_pair(std::numeric_limits<double>::infinity(), true) ==
        error_control_base->desired_step(make_not_null(&box), previous_step,
                                         cache));

  db::mutate<Tags::StepperErrorUpdated>(
      make_not_null(&box),
      [](const gsl::not_null<bool*> stepper_updated) noexcept {
        *stepper_updated = true;
      });
  const std::pair<double, bool> result = error_control(
      previous_step_error,
      db::get<Tags::HistoryEvolvedVariables<EvolvedVariablesTag>>(box), error,
      true, time_stepper, previous_step, cache);
  // reset the previous step error so we can reuse the former state on the next
  // re-application
  *previous_step_error = db::get<StepChoosers::Tags::PreviousStepError>(box);
  CHECK(error_control_base->desired_step(make_not_null(&box), previous_step,
                                         cache) == result);
  db::mutate<StepChoosers::Tags::PreviousStepError>(
      make_not_null(&box),
      [previous_step_error](const gsl::not_null<std::optional<double>*>
                                previous_step_error_from_box) noexcept {
        *previous_step_error_from_box = *previous_step_error;
      });
  CHECK(serialize_and_deserialize(error_control)(
            previous_step_error,
            db::get<Tags::HistoryEvolvedVariables<EvolvedVariablesTag>>(box),
            error, true, time_stepper, previous_step, cache) == result);
  CHECK(serialize_and_deserialize(error_control_base)
            ->desired_step(make_not_null(&box), previous_step, cache) ==
        result);
  return result;
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Time.StepChoosers.ErrorControl", "[Unit][Time]") {
  Parallel::register_derived_classes_with_charm<StepChooserType>();

  MAKE_GENERATOR(generator);
  std::uniform_real_distribution<> step_dist{1.0, 2.0};
  Variables<tmpl::list<EvolvedVar1, EvolvedVar2>> step_values{5};
  fill_with_random_values(make_not_null(&step_values),
                          make_not_null(&generator), make_not_null(&step_dist));
  const DataVector flattened_step_values{step_values.data(), 15};
  std::uniform_real_distribution<> error_dist{1.0e-4, 2.0e-4};
  Variables<tmpl::list<Tags::StepperError<EvolvedVar1>,
                       Tags::StepperError<EvolvedVar2>>>
      step_errors{5};
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
      std::optional<double> previous_step_error;
      const auto first_result =
          get_suggestion(make_not_null(&previous_step_error), error_control,
                         step_values, step_errors, 1.0, stepper_order);
      // manually calculated in the special case in question: only absolute
      // errors constrained
      const double expected_linf_error = max(flattened_step_errors) / 5.0e-4;
      CHECK(approx(first_result.first) ==
            0.95 / pow(expected_linf_error, 1.0 / stepper_order));
      CHECK(first_result.second);
      const auto second_result =
          get_suggestion(make_not_null(&previous_step_error), error_control,
                         step_values, decltype(step_errors){1.2 * step_errors},
                         first_result.first, stepper_order);
      CHECK(approx(second_result.first) ==
            0.95 * first_result.first /
                (pow(expected_linf_error, 1.1 / stepper_order) *
                 pow(1.2, 0.7 / stepper_order)));
      CHECK(second_result.first);
    }
    {
      INFO("Test error control step fixed by relative tolerance");
      const StepChoosers::ErrorControl<EvolvedVariablesTag> error_control{
          0.0, 3.0e-4, 2.0, 0.5, 0.95};
      std::optional<double> previous_step_error;
      const auto first_result =
          get_suggestion(make_not_null(&previous_step_error), error_control,
                         step_values, step_errors, 1.0, stepper_order);
      // manually calculated in the special case in question: only relative
      // errors constrained
      const double expected_linf_error =
          max(flattened_step_errors /
              (flattened_step_values + flattened_step_errors)) /
          3.0e-4;
      CHECK(approx(first_result.first) ==
            0.95 / pow(expected_linf_error, 1.0 / stepper_order));
      CHECK(first_result.second);
      const auto second_result =
          get_suggestion(make_not_null(&previous_step_error), error_control,
                         step_values, decltype(step_errors){1.2 * step_errors},
                         first_result.first, stepper_order);
      const double new_expected_linf_error =
          max(1.2 * flattened_step_errors /
              (flattened_step_values + 1.2 * flattened_step_errors)) /
          3.0e-4;
      CHECK(approx(second_result.first) ==
            0.95 * first_result.first /
                (pow(pow(new_expected_linf_error, 1.0 / stepper_order), 0.7) *
                 pow(pow(expected_linf_error, 1.0 / stepper_order), 0.4)));
      CHECK(second_result.first);
    }
    {
      INFO("Test error control step failure");
      const StepChoosers::ErrorControl<EvolvedVariablesTag> error_control{
          4.0e-5, 4.0e-5, 2.0, 0.5, 0.95};
      std::optional<double> previous_step_error;
      const auto first_result =
          get_suggestion(make_not_null(&previous_step_error), error_control,
                         step_values, step_errors, 1.0, stepper_order);
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
      std::optional<double> previous_step_error;
      const auto first_result =
          get_suggestion(make_not_null(&previous_step_error), error_control,
                         step_values, step_errors, 1.0, stepper_order);
      // manually calculated in the special case in question: only absolute
      // errors constrained
      CHECK(first_result.first == 0.9);
    }
    {
      INFO("Test error control clamped minimum");
      const StepChoosers::ErrorControl<EvolvedVariablesTag> error_control{
          1.0e-1, 1.0e-1, 2.0, 0.5, 0.95};
      std::optional<double> previous_step_error;
      const auto first_result =
          get_suggestion(make_not_null(&previous_step_error), error_control,
                         step_values, step_errors, 1.0, stepper_order);
      CHECK(first_result == std::make_pair(2.0, true));
    }
  }
  // test option creation
  TestHelpers::test_creation<
      std::unique_ptr<StepChoosers::ErrorControl<EvolvedVariablesTag>>>(
      "ErrorControl:\n"
      "  SafetyFactor: 0.95\n"
      "  AbsoluteTolerance: 1.0e-5\n"
      "  RelativeTolerance: 1.0e-4\n"
      "  MaxFactor: 2.1\n"
      "  MinFactor: 0.5");
  TestHelpers::test_creation<std::unique_ptr<
      StepChoosers::ErrorControl<EvolvedVariablesTag, ErrorControlSelecter>>>(
      "ErrorControl(SelectorLabel):\n"
      "  SafetyFactor: 0.95\n"
      "  AbsoluteTolerance: 1.0e-5\n"
      "  RelativeTolerance: 1.0e-4\n"
      "  MaxFactor: 2.1\n"
      "  MinFactor: 0.5");

  // Test `IsUsingTimeSteppingErrorControl` tag:
  {
    INFO("IsUsingTimeSteppingErrorControl tag test");
    using step_choosers_with_error_control =
        tmpl::list<StepChoosers::Registrars::Increase,
                   StepChoosers::Registrars::Constant,
                   StepChoosers::Registrars::ErrorControl<EvolvedVariablesTag>,
                   StepChoosers::Registrars::PreventRapidIncrease>;
    using step_choosers_without_error_control =
        tmpl::list<StepChoosers::Registrars::Increase,
                   StepChoosers::Registrars::Constant,
                   StepChoosers::Registrars::PreventRapidIncrease>;
    const auto step_choosers_collection_0 = TestHelpers::test_creation<
        typename Tags::StepChoosers<step_choosers_with_error_control>::type>(
        "- ErrorControl:\n"
        "    SafetyFactor: 0.95\n"
        "    AbsoluteTolerance: 1.0e-5\n"
        "    RelativeTolerance: 1.0e-4\n"
        "    MaxFactor: 2.1\n"
        "    MinFactor: 0.5\n"
        "- Increase:\n"
        "    Factor: 2\n"
        "- Constant: 0.5");
    const auto step_choosers_collection_1 = TestHelpers::test_creation<
        typename Tags::StepChoosers<step_choosers_with_error_control>::type>(
        "- PreventRapidIncrease:\n"
        "- Increase:\n"
        "    Factor: 2\n"
        "- Constant: 0.5");
    const auto step_choosers_collection_2 = TestHelpers::test_creation<
        typename Tags::StepChoosers<step_choosers_without_error_control>::type>(
        "- PreventRapidIncrease:\n"
        "- Increase:\n"
        "    Factor: 2\n"
        "- Constant: 0.5");
    CHECK(Tags::IsUsingTimeSteppingErrorControl<
          step_choosers_with_error_control>::
              create_from_options<Metavariables>(step_choosers_collection_0));
    CHECK_FALSE(
        Tags::IsUsingTimeSteppingErrorControl<
            step_choosers_with_error_control>::
            create_from_options<Metavariables>(step_choosers_collection_1));
    CHECK_FALSE(
        Tags::IsUsingTimeSteppingErrorControl<
            step_choosers_without_error_control>::
            create_from_options<Metavariables>(step_choosers_collection_2));
    CHECK_FALSE(Tags::IsUsingTimeSteppingErrorControl<
                tmpl::list<>>::create_from_options());
  }
}
