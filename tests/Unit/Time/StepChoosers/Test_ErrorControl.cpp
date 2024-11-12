// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <cmath>
#include <cstddef>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "DataStructures/VariablesTag.hpp"
#include "Framework/TestCreation.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/DataStructures/DataBox/TestHelpers.hpp"
#include "Options/Protocols/FactoryCreation.hpp"
#include "Parallel/Tags/Metavariables.hpp"
#include "ParallelAlgorithms/EventsAndTriggers/Completion.hpp"
#include "ParallelAlgorithms/EventsAndTriggers/Event.hpp"
#include "ParallelAlgorithms/EventsAndTriggers/LogicalTriggers.hpp"
#include "ParallelAlgorithms/EventsAndTriggers/Tags.hpp"
#include "ParallelAlgorithms/EventsAndTriggers/Trigger.hpp"
#include "ParallelAlgorithms/EventsAndTriggers/WhenToCheck.hpp"
#include "Time/ChangeSlabSize/Event.hpp"
#include "Time/Slab.hpp"
#include "Time/StepChoosers/Constant.hpp"
#include "Time/StepChoosers/ErrorControl.hpp"
#include "Time/StepChoosers/LimitIncrease.hpp"
#include "Time/StepChoosers/StepChooser.hpp"
#include "Time/StepperErrorEstimate.hpp"
#include "Time/StepperErrorTolerances.hpp"
#include "Time/Tags/IsUsingTimeSteppingErrorControl.hpp"
#include "Time/Tags/StepChoosers.hpp"
#include "Time/Tags/StepperErrorTolerances.hpp"
#include "Time/Tags/StepperErrors.hpp"
#include "Time/TimeStepRequest.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Literals.hpp"
#include "Utilities/ProtocolHelpers.hpp"
#include "Utilities/Serialization/RegisterDerivedClassesWithCharm.hpp"
#include "Utilities/TMPL.hpp"

class DataVector;

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
  using step_choosers_without_error_control =
      tmpl::list<StepChoosers::LimitIncrease, StepChoosers::Constant>;
  template <typename Use>
  using step_choosers = tmpl::append<
      step_choosers_without_error_control,
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

template <typename StepChooserUse>
std::pair<std::optional<double>, bool> get_suggestion(
    const StepChoosers::ErrorControl<StepChooserUse, EvolvedVariablesTag>&
        error_control,
    const std::optional<StepperErrorEstimate>& error,
    const std::optional<StepperErrorEstimate>& previous_error,
    const double previous_step) {
  auto box = db::create<
      db::AddSimpleTags<Parallel::Tags::MetavariablesImpl<Metavariables<true>>,
                        Tags::StepperErrors<EvolvedVariablesTag>>>(
      Metavariables<true>{}, std::array{previous_error, error});

  const std::unique_ptr<StepChooser<StepChooserUse>> error_control_base =
      std::make_unique<
          StepChoosers::ErrorControl<StepChooserUse, EvolvedVariablesTag>>(
          error_control);

  const auto result =
      error_control(std::array{previous_error, error}, previous_step);
  CHECK(result.first == TimeStepRequest{.size_goal = result.first.size_goal});
  CHECK(error_control_base->desired_step(previous_step, box) == result);
  CHECK(serialize_and_deserialize(error_control)(
            std::array{previous_error, error}, previous_step) == result);
  CHECK(serialize_and_deserialize(error_control_base)
            ->desired_step(previous_step, box) == result);
  return {result.first.size_goal, result.second};
}

template <typename StepChooserUse>
void test_chooser() {
  using ErrorControl =
      StepChoosers::ErrorControl<StepChooserUse, EvolvedVariablesTag>;

  const std::vector<size_t> stepper_orders{2_st, 5_st};
  for (const bool time_runs_forward : {true, false}) {
    const double unit_step = time_runs_forward ? 1.0 : -1.0;
    for (size_t stepper_order : stepper_orders) {
      CAPTURE(stepper_order);
      const auto step_errors = [&](const double error_time, const double error,
                                   const double step_size = 1.0) {
        const auto error_slab = time_runs_forward
                                    ? Slab(error_time, error_time + step_size)
                                    : Slab(error_time - step_size, error_time);
        return StepperErrorEstimate{
            time_runs_forward ? error_slab.start() : error_slab.end(),
            (time_runs_forward ? 1 : -1) * error_slab.duration(),
            stepper_order - 1, error};
      };

      {
        INFO("No data available");
        const ErrorControl error_control{5.0e-4, 0.0, 2.0, 0.5, 0.95};
        const auto result = get_suggestion(error_control, {}, {}, unit_step);
        CHECK(not result.first.has_value());
        CHECK(result.second);
      }
      {
        INFO("Test successful step");
        const ErrorControl error_control{5.0e-4, 1.0e-3, 2.0, 0.5, 0.95};
        CHECK(error_control.tolerances() ==
              StepperErrorTolerances{.absolute = 5.0e-4, .relative = 1.0e-3});
        const auto first_result = get_suggestion(
            error_control, {step_errors(0.0, 0.3)}, {}, unit_step);
        REQUIRE(first_result.first.has_value());
        CHECK(approx(*first_result.first) ==
              0.95 * unit_step / pow(0.3, 1.0 / stepper_order));
        CHECK(first_result.second);
        if constexpr (std::is_same_v<StepChooserUse, ::StepChooserUse::Slab>) {
          const auto second_result = get_suggestion(
              error_control, {step_errors(0.0, 0.31, abs(*first_result.first))},
              {step_errors(-1.0, 0.3)}, *first_result.first);
          REQUIRE(second_result.first.has_value());
          CHECK(approx(*second_result.first) ==
                0.95 * *first_result.first /
                    (pow(0.3, -0.4 / stepper_order) *
                     pow(0.31, 0.7 / stepper_order)));
          CHECK(second_result.second);
          // Check that the suggested step size is smaller if the error in
          // increasing faster.
          const auto adjusted_second_result = get_suggestion(
              error_control, {step_errors(0.0, 0.31, abs(*first_result.first))},
              {step_errors(-1.0, 0.1)}, *first_result.first);
          REQUIRE(adjusted_second_result.first.has_value());
          CHECK(abs(*adjusted_second_result.first) < abs(*second_result.first));
        } else {
          // Check that the result is independent of the old error
          const auto second_result =
              get_suggestion(error_control, {step_errors(0.0, 0.3)},
                             {step_errors(-1.0, 0.7)}, unit_step);
          CHECK(first_result == second_result);
        }
      }
      {
        INFO("Test error control step failure");
        const ErrorControl error_control{4.0e-5, 4.0e-5, 2.0, 0.5, 0.95};
        const auto result_start = get_suggestion(
            error_control, {step_errors(0.0, 1.2)}, {}, unit_step);
        REQUIRE(result_start.first.has_value());
        const auto result_end = get_suggestion(
            error_control, {step_errors(-1.0, 1.2)}, {}, unit_step);
        REQUIRE(result_end.first.has_value());
        const auto result_end2 = get_suggestion(
            error_control, {step_errors(-1.0, 1.2)}, {}, *result_end.first);
        REQUIRE(result_end2.first.has_value());
        CHECK(approx(*result_start.first) ==
              0.95 * unit_step / pow(1.2, 1.0 / stepper_order));
        CHECK(result_end.first == result_start.first);
        CHECK(result_end2.first == result_start.first);
        CHECK_FALSE(result_start.second);
        CHECK_FALSE(result_end.second);
        CHECK(result_end2.second);
      }
      {
        INFO("Test error control clamped minimum");
        const ErrorControl error_control{4.0e-5, 4.0e-5, 2.0, 0.9, 0.95};
        const auto first_result = get_suggestion(
            error_control, {step_errors(0.0, 10.0)}, {}, unit_step);
        CHECK(first_result.first == std::optional(0.9 * unit_step));
      }
      {
        INFO("Test error control clamped maximum");
        const ErrorControl error_control{1.0e-1, 1.0e-1, 2.0, 0.5, 0.95};
        const auto first_result = get_suggestion(
            error_control, {step_errors(0.0, 0.01)}, {}, unit_step);
        CHECK(first_result == std::pair(std::optional(2.0 * unit_step), true));
      }
    }
  }
  // test option creation
  TestHelpers::test_factory_creation<
      StepChooser<StepChooserUse>,
      StepChoosers::ErrorControl<StepChooserUse, EvolvedVariablesTag>>(
      "ErrorControl:\n"
      "  SafetyFactor: 0.95\n"
      "  AbsoluteTolerance: 1.0e-5\n"
      "  RelativeTolerance: 1.0e-4\n"
      "  MaxFactor: 2.1\n"
      "  MinFactor: 0.5");
  TestHelpers::test_factory_creation<
      StepChooser<StepChooserUse>,
      StepChoosers::ErrorControl<StepChooserUse, EvolvedVariablesTag,
                                 ErrorControlSelecter>>(
      "ErrorControl(SelectorLabel):\n"
      "  SafetyFactor: 0.95\n"
      "  AbsoluteTolerance: 1.0e-5\n"
      "  RelativeTolerance: 1.0e-4\n"
      "  MaxFactor: 2.1\n"
      "  MinFactor: 0.5");

  CHECK(StepChoosers::ErrorControl<StepChooserUse, EvolvedVariablesTag,
                                   ErrorControlSelecter>{}
            .uses_local_data());
}

void test_tags() {
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
                  "- LimitIncrease:\n"
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
                  "- LimitIncrease:\n"
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
                  "- LimitIncrease:\n"
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
                  "- LimitIncrease:\n"
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
                  "- LimitIncrease:\n"
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
        db::AddSimpleTags<
            Tags::EventsAndTriggers<Triggers::WhenToCheck::AtSlabs>>,
        db::AddComputeTags<
            Tags::IsUsingTimeSteppingErrorControlCompute<false>,
            Tags::StepperErrorTolerancesCompute<EvolvedVariablesTag, false>,
            Tags::StepperErrorTolerancesCompute<AltEvolvedVariablesTag,
                                                false>>>();
    db::mutate<Tags::EventsAndTriggers<Triggers::WhenToCheck::AtSlabs>>(
        [](const gsl::not_null<EventsAndTriggers*> events) {
          *events = TestHelpers::test_creation<EventsAndTriggers,
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
              "          - LimitIncrease:\n"
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

    db::mutate<Tags::EventsAndTriggers<Triggers::WhenToCheck::AtSlabs>>(
        [](const gsl::not_null<EventsAndTriggers*> events) {
          *events = TestHelpers::test_creation<EventsAndTriggers,
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
              "          - LimitIncrease:\n"
              "              Factor: 2\n"
              "          - Constant: 0.5");
        },
        make_not_null(&box));
    CHECK_FALSE(db::get<Tags::IsUsingTimeSteppingErrorControl>(box));
    CHECK(not db::get<Tags::StepperErrorTolerances<EvolvedVariablesTag>>(box)
                  .has_value());
    CHECK(not db::get<Tags::StepperErrorTolerances<AltEvolvedVariablesTag>>(box)
                  .has_value());

    db::mutate<Tags::EventsAndTriggers<Triggers::WhenToCheck::AtSlabs>>(
        [](const gsl::not_null<EventsAndTriggers*> events) {
          *events = TestHelpers::test_creation<EventsAndTriggers,
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

    db::mutate<Tags::EventsAndTriggers<Triggers::WhenToCheck::AtSlabs>>(
        [](const gsl::not_null<EventsAndTriggers*> events) {
          *events = TestHelpers::test_creation<EventsAndTriggers,
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

SPECTRE_TEST_CASE("Unit.Time.StepChoosers.ErrorControl", "[Unit][Time]") {
  register_factory_classes_with_charm<Metavariables<true>>();

  test_chooser<StepChooserUse::Slab>();
  test_chooser<StepChooserUse::LtsStep>();
  test_tags();
}
}  // namespace
