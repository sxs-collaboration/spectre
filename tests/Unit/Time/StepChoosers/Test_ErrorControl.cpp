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
#include "Helpers/DataStructures/DataBox/TestHelpers.hpp"
#include "Helpers/DataStructures/MakeWithRandomValues.hpp"
#include "Options/Protocols/FactoryCreation.hpp"
#include "ParallelAlgorithms/EventsAndTriggers/Completion.hpp"
#include "ParallelAlgorithms/EventsAndTriggers/Event.hpp"
#include "ParallelAlgorithms/EventsAndTriggers/LogicalTriggers.hpp"
#include "ParallelAlgorithms/EventsAndTriggers/Tags.hpp"
#include "ParallelAlgorithms/EventsAndTriggers/Trigger.hpp"
#include "Time/ChangeSlabSize/Event.hpp"
#include "Time/History.hpp"
#include "Time/StepChoosers/Constant.hpp"
#include "Time/StepChoosers/ErrorControl.hpp"
#include "Time/StepChoosers/Increase.hpp"
#include "Time/StepChoosers/PreventRapidIncrease.hpp"
#include "Time/StepChoosers/StepChooser.hpp"
#include "Time/Tags/HistoryEvolvedVariables.hpp"
#include "Time/Tags/IsUsingTimeSteppingErrorControl.hpp"
#include "Time/Tags/StepChoosers.hpp"
#include "Time/Tags/StepperErrorTolerances.hpp"
#include "Time/Tags/StepperErrors.hpp"
#include "Time/TimeStepId.hpp"
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

struct EvolvedVar3 : db::SimpleTag {
  using type = Scalar<DataVector>;
};

using EvolvedVariablesTag =
    Tags::Variables<tmpl::list<EvolvedVar1, EvolvedVar2>>;

using AltEvolvedVariablesTag = Tags::Variables<tmpl::list<EvolvedVar3>>;

struct ErrorControlSelecter {
  static std::string name() {
    return "SelectorLabel";
  }
};

template <bool WithErrorControl, bool WithAltErrorControl = false>
struct Metavariables {
  template <typename Use>
  using step_choosers_without_error_control =
      tmpl::list<StepChoosers::Increase<Use>, StepChoosers::Constant<Use>,
                 StepChoosers::PreventRapidIncrease<Use>>;
  template <typename Use>
  using step_choosers = tmpl::append<
      step_choosers_without_error_control<Use>,
      tmpl::conditional_t<
          WithErrorControl,
          tmpl::list<StepChoosers::ErrorControl<Use, EvolvedVariablesTag>>,
          tmpl::list<>>,
      tmpl::conditional_t<
          WithAltErrorControl,
          tmpl::list<StepChoosers::ErrorControl<Use, AltEvolvedVariablesTag,
                                                ErrorControlSelecter>>,
          tmpl::list<>>>;

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
    const std::optional<StepperErrorEstimate<Variables<EvolvedTags>>>& error,
    const std::optional<StepperErrorEstimate<Variables<EvolvedTags>>>&
        previous_error,
    const double previous_step, const size_t stepper_order) {
  TimeSteppers::History<Variables<EvolvedTags>> history{stepper_order};
  history.insert(TimeStepId{true, 0, {{0.0, 1.0}, {0, 1}}}, step_values,
                 0.1 * step_values);
  history.discard_value(TimeStepId{true, 0, {{0.0, 1.0}, {0, 1}}});
  auto box = db::create<
      db::AddSimpleTags<Parallel::Tags::MetavariablesImpl<Metavariables<true>>,
                        Tags::HistoryEvolvedVariables<EvolvedVariablesTag>,
                        Tags::StepperErrors<EvolvedVariablesTag>>>(
      Metavariables<true>{}, history, std::array{previous_error, error});

  const std::unique_ptr<StepChooser<StepChooserUse::LtsStep>>
      error_control_base = std::make_unique<LtsErrorControl>(error_control);

  const std::pair<double, bool> result =
      error_control(history, std::array{previous_error, error}, previous_step);
  CHECK(error_control_base->desired_step(previous_step, box) == result);
  CHECK(serialize_and_deserialize(error_control)(
            history, std::array{previous_error, error}, previous_step) ==
        result);
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
  Variables<tmpl::list<EvolvedVar1, EvolvedVar2>> step_error_data{5};
  fill_with_random_values(make_not_null(&step_error_data),
                          make_not_null(&generator),
                          make_not_null(&error_dist));
  const DataVector flattened_step_errors{step_error_data.data(), 15};
  const std::vector<size_t> stepper_orders{2_st, 5_st};
  for (const bool time_runs_forward : {true, false}) {
    for (size_t stepper_order : stepper_orders) {
      CAPTURE(stepper_order);
      const auto step_errors = [&](const double error_time,
                                   const double error_multiplier = 1.0,
                                   const double step_size = 1.0) {
        const auto error_slab = time_runs_forward
                                    ? Slab(error_time, error_time + step_size)
                                    : Slab(error_time - step_size, error_time);
        return StepperErrorEstimate<decltype(step_error_data)>{
            time_runs_forward ? error_slab.start() : error_slab.end(),
            (time_runs_forward ? 1 : -1) * error_slab.duration(),
            stepper_order - 1, step_error_data * error_multiplier};
      };

      {
        INFO("No data available");
        const LtsErrorControl error_control{5.0e-4, 0.0, 2.0, 0.5, 0.95};
        const auto result = get_suggestion(error_control, step_values, {}, {},
                                           1.0, stepper_order);
        CHECK(result.first == std::numeric_limits<double>::infinity());
        CHECK(result.second);
      }
      {
        INFO("Test error control step fixed by absolute tolerance");
        const LtsErrorControl error_control{5.0e-4, 0.0, 2.0, 0.5, 0.95};
        const auto first_result =
            get_suggestion(error_control, step_values, {step_errors(0.0)}, {},
                           1.0, stepper_order);
        // manually calculated in the special case in question: only absolute
        // errors constrained
        const double expected_linf_error = max(flattened_step_errors) / 5.0e-4;
        CHECK(approx(first_result.first) ==
              0.95 / pow(expected_linf_error, 1.0 / stepper_order));
        CHECK(first_result.second);
        const auto second_result = get_suggestion(
            error_control, step_values,
            {step_errors(0.0, 1.2, first_result.first)}, {step_errors(-1.0)},
            first_result.first, stepper_order);
        CHECK(approx(second_result.first) ==
              0.95 * first_result.first /
                  (pow(expected_linf_error, 0.3 / stepper_order) *
                   pow(1.2, 0.7 / stepper_order)));
        CHECK(second_result.second);
        // Check that the suggested step size is smaller if the error in
        // increasing faster.
        const auto adjusted_second_result = get_suggestion(
            error_control, step_values,
            {step_errors(0.0, 1.2, first_result.first)},
            {step_errors(-1.0, 0.5)}, first_result.first, stepper_order);
        CHECK(adjusted_second_result.first < second_result.first);
      }
      {
        INFO("Test error control step fixed by relative tolerance");
        const LtsErrorControl error_control{0.0, 3.0e-4, 2.0, 0.5, 0.95};
        const auto first_result =
            get_suggestion(error_control, step_values, {step_errors(0.0)}, {},
                           1.0, stepper_order);
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
            error_control, step_values,
            {step_errors(0.0, 1.2, first_result.first)}, {step_errors(-1.0)},
            first_result.first, stepper_order);
        const double new_expected_linf_error =
            max(1.2 * flattened_step_errors /
                (flattened_step_values + 1.2 * flattened_step_errors)) /
            3.0e-4;
        CHECK(approx(second_result.first) ==
              0.95 * first_result.first *
                  pow(pow(new_expected_linf_error, 1.0 / stepper_order), -0.7) *
                  pow(pow(expected_linf_error, 1.0 / stepper_order), 0.4));
        CHECK(second_result.second);
      }
      {
        INFO("Test error control step failure");
        const LtsErrorControl error_control{4.0e-5, 4.0e-5, 2.0, 0.5, 0.95};
        const auto result_start =
            get_suggestion(error_control, step_values, {step_errors(0.0)}, {},
                           1.0, stepper_order);
        const auto result_end =
            get_suggestion(error_control, step_values, {step_errors(-1.0)}, {},
                           1.0, stepper_order);
        const auto result_end2 =
            get_suggestion(error_control, step_values, {step_errors(-1.0)}, {},
                           result_end.first, stepper_order);
        // manually calculated in the special case in question: only absolute
        // errors constrained
        const double expected_linf_error =
            max(flattened_step_errors /
                (flattened_step_values + flattened_step_errors + 1.0)) /
            4.0e-5;
        CHECK(approx(result_start.first) ==
              std::max(0.95 / pow(expected_linf_error, 1.0 / stepper_order),
                       0.5));
        CHECK(result_end.first == result_start.first);
        CHECK(result_end2.first == result_start.first);
        CHECK_FALSE(result_start.second);
        CHECK_FALSE(result_end.second);
        CHECK(result_end2.second);
      }
      {
        INFO("Test error control clamped minimum");
        const LtsErrorControl error_control{4.0e-5, 4.0e-5, 2.0, 0.9, 0.95};
        const auto first_result =
            get_suggestion(error_control, step_values, {step_errors(0.0)}, {},
                           1.0, stepper_order);
        // manually calculated in the special case in question: only absolute
        // errors constrained
        CHECK(first_result.first == 0.9);
      }
      {
        INFO("Test error control clamped minimum");
        const LtsErrorControl error_control{1.0e-1, 1.0e-1, 2.0, 0.5, 0.95};
        const auto first_result =
            get_suggestion(error_control, step_values, {step_errors(0.0)}, {},
                           1.0, stepper_order);
        CHECK(first_result == std::make_pair(2.0, true));
      }
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

  // Test compute tags
  TestHelpers::db::test_compute_tag<
      Tags::IsUsingTimeSteppingErrorControlCompute<true>>(
      "IsUsingTimeSteppingErrorControl");
  TestHelpers::db::test_compute_tag<
      Tags::StepperErrorTolerancesCompute<EvolvedVariablesTag, true>>(
      "StepperErrorTolerances(Variables(EvolvedVar1,EvolvedVar2))");

  {
    INFO("Compute tag LTS test");
    auto box = db::create<
        db::AddSimpleTags<Tags::StepChoosers>,
        db::AddComputeTags<
            Tags::IsUsingTimeSteppingErrorControlCompute<true>,
            Tags::StepperErrorTolerancesCompute<EvolvedVariablesTag, true>,
            Tags::StepperErrorTolerancesCompute<AltEvolvedVariablesTag,
                                                true>>>();
    db::mutate<Tags::StepChoosers>(
        [](const gsl::not_null<Tags::StepChoosers::type*> choosers) {
          *choosers =
              TestHelpers::test_creation<typename Tags::StepChoosers::type,
                                         Metavariables<true, true>>(
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
    CHECK(db::get<Tags::StepperErrorTolerances<EvolvedVariablesTag>>(box) ==
          StepperErrorTolerances{.absolute = 1.0e-5, .relative = 1.0e-4});
    CHECK(not db::get<Tags::StepperErrorTolerances<AltEvolvedVariablesTag>>(box)
                  .has_value());

    db::mutate<Tags::StepChoosers>(
        [](const gsl::not_null<Tags::StepChoosers::type*> choosers) {
          *choosers =
              TestHelpers::test_creation<typename Tags::StepChoosers::type,
                                         Metavariables<true, true>>(
                  "- PreventRapidIncrease:\n"
                  "- Increase:\n"
                  "    Factor: 2\n"
                  "- Constant: 0.5");
        },
        make_not_null(&box));
    CHECK_FALSE(db::get<Tags::IsUsingTimeSteppingErrorControl>(box));
    CHECK(not db::get<Tags::StepperErrorTolerances<EvolvedVariablesTag>>(box)
                  .has_value());
    CHECK(not db::get<Tags::StepperErrorTolerances<AltEvolvedVariablesTag>>(box)
                  .has_value());

    db::mutate<Tags::StepChoosers>(
        [](const gsl::not_null<Tags::StepChoosers::type*> choosers) {
          *choosers =
              TestHelpers::test_creation<typename Tags::StepChoosers::type,
                                         Metavariables<true, true>>("");
        },
        make_not_null(&box));
    CHECK_FALSE(db::get<Tags::IsUsingTimeSteppingErrorControl>(box));
    CHECK(not db::get<Tags::StepperErrorTolerances<EvolvedVariablesTag>>(box)
                  .has_value());
    CHECK(not db::get<Tags::StepperErrorTolerances<AltEvolvedVariablesTag>>(box)
                  .has_value());

    db::mutate<Tags::StepChoosers>(
        [](const gsl::not_null<Tags::StepChoosers::type*> choosers) {
          *choosers =
              TestHelpers::test_creation<typename Tags::StepChoosers::type,
                                         Metavariables<true, true>>(
                  "- ErrorControl:\n"
                  "    SafetyFactor: 0.95\n"
                  "    AbsoluteTolerance: 1.0e-5\n"
                  "    RelativeTolerance: 1.0e-4\n"
                  "    MaxFactor: 2.1\n"
                  "    MinFactor: 0.5\n"
                  "- ErrorControl:\n"
                  "    SafetyFactor: 0.8\n"
                  "    AbsoluteTolerance: 1.0e-5\n"
                  "    RelativeTolerance: 1.0e-4\n"
                  "    MaxFactor: 1.1\n"
                  "    MinFactor: 0.1\n"
                  "- Increase:\n"
                  "    Factor: 2\n"
                  "- Constant: 0.5");
        },
        make_not_null(&box));
    CHECK(db::get<Tags::IsUsingTimeSteppingErrorControl>(box));
    CHECK(db::get<Tags::StepperErrorTolerances<EvolvedVariablesTag>>(box) ==
          StepperErrorTolerances{.absolute = 1.0e-5, .relative = 1.0e-4});
    CHECK(not db::get<Tags::StepperErrorTolerances<AltEvolvedVariablesTag>>(box)
                  .has_value());

    db::mutate<Tags::StepChoosers>(
        [](const gsl::not_null<Tags::StepChoosers::type*> choosers) {
          *choosers =
              TestHelpers::test_creation<typename Tags::StepChoosers::type,
                                         Metavariables<true, true>>(
                  "- ErrorControl:\n"
                  "    SafetyFactor: 0.95\n"
                  "    AbsoluteTolerance: 1.0e-5\n"
                  "    RelativeTolerance: 1.0e-4\n"
                  "    MaxFactor: 2.1\n"
                  "    MinFactor: 0.5\n"
                  "- ErrorControl(SelectorLabel):\n"
                  "    SafetyFactor: 0.8\n"
                  "    AbsoluteTolerance: 1.0e-5\n"
                  "    RelativeTolerance: 1.0e-8\n"
                  "    MaxFactor: 1.1\n"
                  "    MinFactor: 0.1\n"
                  "- Increase:\n"
                  "    Factor: 2\n"
                  "- Constant: 0.5");
        },
        make_not_null(&box));
    CHECK(db::get<Tags::IsUsingTimeSteppingErrorControl>(box));
    CHECK(db::get<Tags::StepperErrorTolerances<EvolvedVariablesTag>>(box) ==
          StepperErrorTolerances{.absolute = 1.0e-5, .relative = 1.0e-4});
    CHECK(db::get<Tags::StepperErrorTolerances<AltEvolvedVariablesTag>>(box) ==
          StepperErrorTolerances{.absolute = 1.0e-5, .relative = 1.0e-8});

    db::mutate<Tags::StepChoosers>(
        [](const gsl::not_null<Tags::StepChoosers::type*> choosers) {
          *choosers =
              TestHelpers::test_creation<typename Tags::StepChoosers::type,
                                         Metavariables<true, true>>(
                  "- ErrorControl:\n"
                  "    SafetyFactor: 0.95\n"
                  "    AbsoluteTolerance: 1.0e-5\n"
                  "    RelativeTolerance: 1.0e-4\n"
                  "    MaxFactor: 2.1\n"
                  "    MinFactor: 0.5\n"
                  "- ErrorControl:\n"
                  "    SafetyFactor: 0.8\n"
                  "    AbsoluteTolerance: 1.0e-5\n"
                  "    RelativeTolerance: 1.0e-8\n"
                  "    MaxFactor: 1.1\n"
                  "    MinFactor: 0.1\n"
                  "- Increase:\n"
                  "    Factor: 2\n"
                  "- Constant: 0.5");
        },
        make_not_null(&box));
    CHECK_THROWS_WITH(
        db::get<Tags::StepperErrorTolerances<EvolvedVariablesTag>>(box),
        Catch::Matchers::ContainsSubstring("All ErrorControl events for ") and
            Catch::Matchers::ContainsSubstring("EvolvedVar1") and
            Catch::Matchers::ContainsSubstring(
                " must use the same tolerances."));
  }

  {
    INFO("Compute tag GTS test");
    auto box = db::create<
        db::AddSimpleTags<Tags::EventsAndTriggers>,
        db::AddComputeTags<
            Tags::IsUsingTimeSteppingErrorControlCompute<false>,
            Tags::StepperErrorTolerancesCompute<EvolvedVariablesTag, false>,
            Tags::StepperErrorTolerancesCompute<AltEvolvedVariablesTag,
                                                false>>>();
    db::mutate<Tags::EventsAndTriggers>(
        [](const gsl::not_null<Tags::EventsAndTriggers::type*> events) {
          *events = TestHelpers::test_creation<Tags::EventsAndTriggers::type,
                                               Metavariables<true, true>>(
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
    CHECK(db::get<Tags::StepperErrorTolerances<EvolvedVariablesTag>>(box) ==
          StepperErrorTolerances{.absolute = 1.0e-5, .relative = 1.0e-4});
    CHECK(not db::get<Tags::StepperErrorTolerances<AltEvolvedVariablesTag>>(box)
                  .has_value());

    db::mutate<Tags::EventsAndTriggers>(
        [](const gsl::not_null<Tags::EventsAndTriggers::type*> events) {
          *events = TestHelpers::test_creation<Tags::EventsAndTriggers::type,
                                               Metavariables<true, true>>(
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
    CHECK(not db::get<Tags::StepperErrorTolerances<EvolvedVariablesTag>>(box)
                  .has_value());
    CHECK(not db::get<Tags::StepperErrorTolerances<AltEvolvedVariablesTag>>(box)
                  .has_value());

    db::mutate<Tags::EventsAndTriggers>(
        [](const gsl::not_null<Tags::EventsAndTriggers::type*> events) {
          *events = TestHelpers::test_creation<Tags::EventsAndTriggers::type,
                                               Metavariables<true, true>>(
              "- Trigger: Always\n"
              "  Events:\n"
              "    - Completion\n"
              "- Trigger: Always\n"
              "  Events:\n"
              "    - Completion");
        },
        make_not_null(&box));
    CHECK_FALSE(db::get<Tags::IsUsingTimeSteppingErrorControl>(box));
    CHECK(not db::get<Tags::StepperErrorTolerances<EvolvedVariablesTag>>(box)
                  .has_value());
    CHECK(not db::get<Tags::StepperErrorTolerances<AltEvolvedVariablesTag>>(box)
                  .has_value());

    db::mutate<Tags::EventsAndTriggers>(
        [](const gsl::not_null<Tags::EventsAndTriggers::type*> events) {
          *events = TestHelpers::test_creation<Tags::EventsAndTriggers::type,
                                               Metavariables<true, true>>("");
        },
        make_not_null(&box));
    CHECK_FALSE(db::get<Tags::IsUsingTimeSteppingErrorControl>(box));
    CHECK(not db::get<Tags::StepperErrorTolerances<EvolvedVariablesTag>>(box)
                  .has_value());
    CHECK(not db::get<Tags::StepperErrorTolerances<AltEvolvedVariablesTag>>(box)
                  .has_value());
  }
}
