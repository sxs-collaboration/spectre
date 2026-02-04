// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cstddef>
#include <memory>
#include <optional>
#include <string>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/MetavariablesTag.hpp"
#include "Framework/TestCreation.hpp"
#include "Framework/TestHelpers.hpp"
#include "Options/Protocols/FactoryCreation.hpp"
#include "Options/String.hpp"
#include "Time/Slab.hpp"
#include "Time/StepChoosers/Constant.hpp"
#include "Time/StepChoosers/FixedLtsRatio.hpp"
#include "Time/StepChoosers/LimitIncrease.hpp"
#include "Time/StepChoosers/StepChooser.hpp"
#include "Time/Tags/FixedLtsRatio.hpp"
#include "Time/Tags/TimeStep.hpp"
#include "Time/TimeStepRequest.hpp"
#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/MakeVector.hpp"
#include "Utilities/ProtocolHelpers.hpp"
#include "Utilities/Serialization/CharmPupable.hpp"
#include "Utilities/Serialization/RegisterDerivedClassesWithCharm.hpp"
#include "Utilities/TMPL.hpp"

namespace {
class ErrorChooser : public StepChooser<StepChooserUse::LtsStep> {
 public:
  ErrorChooser() = default;
  using PUP::able::register_constructor;
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-function"
  WRAPPED_PUPable_decl_template(ErrorChooser);  // NOLINT
#pragma GCC diagnostic pop

  static constexpr Options::String help{""};
  using options = tmpl::list<>;

  using argument_tags = tmpl::list<>;

  TimeStepRequest operator()(double /*last_step*/) const {
    ERROR("StepChooser should not be called in fixed LTS region");
  }

  bool uses_local_data() const override { return false; }
  bool can_be_delayed() const override { return true; }
};

PUP::able::PUP_ID ErrorChooser::my_PUP_ID = 0;  // NOLINT

struct Metavariables {
  struct factory_creation
      : tt::ConformsTo<Options::protocols::FactoryCreation> {
    using factory_classes = tmpl::map<
        tmpl::pair<StepChooser<StepChooserUse::LtsStep>,
                   tmpl::list<StepChoosers::Constant,
                              StepChoosers::LimitIncrease, ErrorChooser>>,
        tmpl::pair<StepChooser<StepChooserUse::Slab>,
                   tmpl::list<StepChoosers::FixedLtsRatio>>>;
  };
};

void test(const std::optional<double>& expected_goal,
          const std::optional<double>& expected_size,
          const std::optional<size_t>& fixed_ratio,
          const std::string& lts_choosers) {
  CAPTURE(lts_choosers);
  const Slab slab(2.0, 6.0);
  const auto time_step = slab.duration() / 2;

  for (const auto& time_sign : {1, -1}) {
    CAPTURE(time_sign);
    auto box = db::create<
        db::AddSimpleTags<Parallel::Tags::MetavariablesImpl<Metavariables>,
                          Tags::TimeStep, Tags::FixedLtsRatio>>(
        Metavariables{}, time_sign * time_step, fixed_ratio);

    const auto chooser = TestHelpers::test_creation<
        std::unique_ptr<StepChooser<StepChooserUse::Slab>>, Metavariables>(
        "FixedLtsRatio:\n"
        "  StepChoosers:\n" +
        lts_choosers);

    const auto set_sign = [&](const std::optional<double>& opt) {
      if (opt.has_value()) {
        return std::optional<double>(time_sign * *opt);
      } else {
        return std::optional<double>{};
      }
    };

    const double current_step = time_sign * slab.duration().value();
    const TimeStepRequest expected{.size_goal = set_sign(expected_goal),
                                   .size = set_sign(expected_size)};
    CHECK(chooser->desired_step(current_step, box) == expected);
    CHECK(serialize_and_deserialize(chooser)->desired_step(current_step, box) ==
          expected);
  }
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Time.StepChoosers.FixedLtsRatio", "[Unit][Time]") {
  register_factory_classes_with_charm<Metavariables>();

  test({}, {}, {}, "    - ErrorChooser");
  test({}, {}, {8}, "");
  test({40.0}, {}, {8},
       "    - Constant: 5.0\n"
       "    - Constant: 7.0");
  // Initial step size used in test is 2.0
  test({}, {64.0}, {8},
       "    - LimitIncrease:\n"
       "        Factor: 4.0\n"
       "    - LimitIncrease:\n"
       "        Factor: 9.0");
  test({40.0}, {32.0}, {8},
       "    - Constant: 5.0\n"
       "    - LimitIncrease:\n"
       "        Factor: 2.0");
  // Should never give a limit larger than the goal.
  test({40.0}, {}, {8},
       "    - Constant: 5.0\n"
       "    - LimitIncrease:\n"
       "        Factor: 4.0");

  CHECK(StepChoosers::FixedLtsRatio{}.uses_local_data());
  CHECK(StepChoosers::FixedLtsRatio{}.can_be_delayed());

  StepChoosers::FixedLtsRatio{}.for_each_step_chooser(
      [](const auto& /*unused*/) { ERROR("Should not be called"); });
  {
    const StepChoosers::FixedLtsRatio two_choosers{
        make_vector<std::unique_ptr<StepChooser<StepChooserUse::LtsStep>>>(
            std::make_unique<StepChoosers::Constant>(1.0),
            std::make_unique<StepChoosers::Constant>(2.0))};
    int count = 0;
    two_choosers.for_each_step_chooser(
        [&](const auto& /*unused*/) { ++count; });
    CHECK(count == 2);
  }
}
