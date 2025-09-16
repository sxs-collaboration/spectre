// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cmath>
#include <cstddef>
#include <memory>
#include <optional>
#include <string>
#include <type_traits>
#include <typeinfo>
#include <vector>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/MetavariablesTag.hpp"
#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "DataStructures/VariablesTag.hpp"
#include "Framework/TestCreation.hpp"
#include "Framework/TestHelpers.hpp"
#include "Options/Protocols/FactoryCreation.hpp"
#include "Time/Slab.hpp"
#include "Time/StepChoosers/ErrorControl.hpp"
#include "Time/StepChoosers/StepChooser.hpp"
#include "Time/StepperErrorEstimate.hpp"
#include "Time/StepperErrorTolerances.hpp"
#include "Time/Tags/StepperErrors.hpp"
#include "Time/TimeStepRequest.hpp"
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

using EvolvedVariablesTag =
    Tags::Variables<tmpl::list<EvolvedVar1, EvolvedVar2>>;

struct ErrorControlSelecter {
  static std::string name() { return "SelectorLabel"; }
};

struct Metavariables {
  struct factory_creation
      : tt::ConformsTo<Options::protocols::FactoryCreation> {
    using factory_classes = tmpl::map<
        tmpl::pair<StepChooser<StepChooserUse::LtsStep>,
                   tmpl::list<StepChoosers::ErrorControl<
                       StepChooserUse::LtsStep, EvolvedVariablesTag>>>,
        tmpl::pair<StepChooser<StepChooserUse::Slab>,
                   tmpl::list<StepChoosers::ErrorControl<
                       StepChooserUse::Slab, EvolvedVariablesTag>>>>;
  };
};

template <typename StepChooserUse>
std::optional<double> get_suggestion(
    const StepChoosers::ErrorControl<StepChooserUse, EvolvedVariablesTag>&
        error_control,
    const std::optional<StepperErrorEstimate>& error,
    const double previous_step) {
  auto box = db::create<
      db::AddSimpleTags<Parallel::Tags::MetavariablesImpl<Metavariables>,
                        Tags::StepperErrors<EvolvedVariablesTag>>>(
      Metavariables{}, error);

  const std::unique_ptr<StepChooser<StepChooserUse>> error_control_base =
      std::make_unique<
          StepChoosers::ErrorControl<StepChooserUse, EvolvedVariablesTag>>(
          error_control);

  const auto result = error_control(error, previous_step);
  CHECK(result == TimeStepRequest{.size_goal = result.size_goal});
  CHECK(error_control_base->desired_step(previous_step, box) == result);
  CHECK(serialize_and_deserialize(error_control)(error, previous_step) ==
        result);
  CHECK(serialize_and_deserialize(error_control_base)
            ->desired_step(previous_step, box) == result);
  return result.size_goal;
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
        const auto result = get_suggestion(error_control, {}, unit_step);
        CHECK(not result.has_value());
      }
      {
        INFO("Test successful step");
        const ErrorControl error_control{5.0e-4, 1.0e-3, 2.0, 0.5, 0.95};
        CHECK(error_control.tolerances().at(typeid(EvolvedVariablesTag)) ==
              StepperErrorTolerances{
                  .estimates = StepperErrorTolerances::Estimates::StepperOrder,
                  .absolute = 5.0e-4,
                  .relative = 1.0e-3});
        const auto result =
            get_suggestion(error_control, {step_errors(0.0, 0.3)}, unit_step);
        REQUIRE(result.has_value());
        CHECK(approx(*result) ==
              0.95 * unit_step / pow(0.3, 1.0 / stepper_order));
      }
      {
        INFO("Test error control step failure");
        const ErrorControl error_control{4.0e-5, 4.0e-5, 2.0, 0.5, 0.95};
        const auto result_start =
            get_suggestion(error_control, {step_errors(0.0, 1.2)}, unit_step);
        REQUIRE(result_start.has_value());
        const auto result_end =
            get_suggestion(error_control, {step_errors(-1.0, 1.2)}, unit_step);
        REQUIRE(result_end.has_value());
        const auto result_end2 = get_suggestion(
            error_control, {step_errors(-1.0, 1.2)}, *result_end);
        REQUIRE(result_end2.has_value());
        CHECK(approx(*result_start) ==
              0.95 * unit_step / pow(1.2, 1.0 / stepper_order));
        CHECK(result_end == result_start);
        CHECK(result_end2 == result_start);
      }
      {
        INFO("Test error control clamped minimum");
        const ErrorControl error_control{4.0e-5, 4.0e-5, 2.0, 0.9, 0.95};
        const auto result =
            get_suggestion(error_control, {step_errors(0.0, 10.0)}, unit_step);
        CHECK(result == std::optional(0.9 * unit_step));
      }
      {
        INFO("Test error control clamped maximum");
        const ErrorControl error_control{1.0e-1, 1.0e-1, 2.0, 0.5, 0.95};
        const auto result =
            get_suggestion(error_control, {step_errors(0.0, 0.01)}, unit_step);
        CHECK(result == std::optional(2.0 * unit_step));
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

SPECTRE_TEST_CASE("Unit.Time.StepChoosers.ErrorControl", "[Unit][Time]") {
  register_factory_classes_with_charm<Metavariables>();

  test_chooser<StepChooserUse::Slab>();
  test_chooser<StepChooserUse::LtsStep>();
}
}  // namespace
