// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <optional>
#include <string>
#include <type_traits>

#include "DataStructures/DataBox/DataBox.hpp"
#include "Helpers/DataStructures/DataBox/TestHelpers.hpp"
#include "Time/Tags/TimeStepper.hpp"
#include "Time/TimeSteppers/AdamsBashforth.hpp"
#include "Time/TimeSteppers/LtsError.hpp"
#include "Time/TimeSteppers/LtsTimeStepper.hpp"
#include "Time/TimeSteppers/Rk3HesthavenSsp.hpp"
#include "Time/TimeSteppers/TimeStepper.hpp"
#include "Utilities/Serialization/RegisterDerivedClassesWithCharm.hpp"
#include "Utilities/TMPL.hpp"

namespace {
struct DummyType {};

class FakeTimeStepper {
 public:
  using provided_time_stepper_interfaces = tmpl::list<FakeTimeStepper>;
};

class MoreSpecificFakeTimeStepper : public FakeTimeStepper {
 public:
  using provided_time_stepper_interfaces =
      tmpl::list<MoreSpecificFakeTimeStepper, FakeTimeStepper>;
};

static_assert(std::is_same_v<
              time_stepper_ref_tags<FakeTimeStepper>,
              tmpl::list<Tags::TimeStepperRef<FakeTimeStepper, FakeTimeStepper>,
                         Tags::LtsOrError>>);
static_assert(
    std::is_same_v<time_stepper_ref_tags<MoreSpecificFakeTimeStepper>,
                   tmpl::list<Tags::TimeStepperRef<MoreSpecificFakeTimeStepper,
                                                   MoreSpecificFakeTimeStepper>,
                              Tags::TimeStepperRef<FakeTimeStepper,
                                                   MoreSpecificFakeTimeStepper>,
                              Tags::LtsOrError>>);
}  // namespace

SPECTRE_TEST_CASE("Unit.Time.Tags.TimeStepper", "[Unit][Time]") {
  TestHelpers::db::test_simple_tag<Tags::ConcreteTimeStepper<DummyType>>(
      "ConcreteTimeStepper");
  TestHelpers::db::test_simple_tag<Tags::TimeStepper<DummyType>>("TimeStepper");
  TestHelpers::db::test_reference_tag<
      Tags::TimeStepperRef<DummyType, DummyType>>("TimeStepper");
  TestHelpers::db::test_reference_tag<Tags::LtsOrError>("TimeStepper");

  register_classes_with_charm<TimeSteppers::AdamsBashforth>();
  // Check that these are allowed...
  Tags::ConcreteTimeStepper<TimeStepper>::create_from_options(
      std::make_unique<TimeSteppers::AdamsBashforth>(3));
  Tags::ConcreteTimeStepper<LtsTimeStepper>::create_from_options(
      std::make_unique<TimeSteppers::AdamsBashforth>(3));
  Tags::ConcreteTimeStepper<LtsTimeStepper>::create_from_options(
      std::make_unique<TimeSteppers::AdamsBashforth>(std::nullopt));
  // ...but this isn't.
  CHECK_THROWS_WITH(
      Tags::ConcreteTimeStepper<TimeStepper>::create_from_options(
          std::make_unique<TimeSteppers::AdamsBashforth>(std::nullopt)),
      Catch::Matchers::ContainsSubstring(
          "Variable-order TimeSteppers are only supported in evolutions with "
          "local time-stepping."));

  // Test LtsOrError
  {
    auto box =
        db::create<db::AddSimpleTags<Tags::ConcreteTimeStepper<TimeStepper>>,
                   time_stepper_ref_tags<TimeStepper>>(
            static_cast<std::unique_ptr<TimeStepper>>(
                std::make_unique<TimeSteppers::AdamsBashforth>(3)));
    const LtsTimeStepper& lts_stepper =
        db::get<Tags::TimeStepper<LtsTimeStepper>>(box);
    CHECK(dynamic_cast<const TimeSteppers::LtsError*>(&lts_stepper) == nullptr);
    CHECK(&lts_stepper == &db::get<Tags::TimeStepper<TimeStepper>>(box));
  }
  {
    auto box =
        db::create<db::AddSimpleTags<Tags::ConcreteTimeStepper<TimeStepper>>,
                   time_stepper_ref_tags<TimeStepper>>(
            static_cast<std::unique_ptr<TimeStepper>>(
                std::make_unique<TimeSteppers::Rk3HesthavenSsp>()));
    const LtsTimeStepper& lts_stepper =
        db::get<Tags::TimeStepper<LtsTimeStepper>>(box);
    CHECK(dynamic_cast<const TimeSteppers::LtsError*>(&lts_stepper) != nullptr);
    CHECK(&lts_stepper != &db::get<Tags::TimeStepper<TimeStepper>>(box));
  }
}
