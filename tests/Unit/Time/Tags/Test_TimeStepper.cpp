// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <string>

#include "Helpers/DataStructures/DataBox/TestHelpers.hpp"
#include "Time/Tags/TimeStepper.hpp"

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

static_assert(
    std::is_same_v<
        time_stepper_ref_tags<FakeTimeStepper>,
        tmpl::list<Tags::TimeStepperRef<FakeTimeStepper, FakeTimeStepper>>>);
static_assert(std::is_same_v<
              time_stepper_ref_tags<MoreSpecificFakeTimeStepper>,
              tmpl::list<Tags::TimeStepperRef<MoreSpecificFakeTimeStepper,
                                              MoreSpecificFakeTimeStepper>,
                         Tags::TimeStepperRef<FakeTimeStepper,
                                              MoreSpecificFakeTimeStepper>>>);
}  // namespace

SPECTRE_TEST_CASE("Unit.Time.Tags.TimeStepper", "[Unit][Time]") {
  TestHelpers::db::test_simple_tag<Tags::ConcreteTimeStepper<DummyType>>(
      "ConcreteTimeStepper");
  TestHelpers::db::test_simple_tag<Tags::TimeStepper<DummyType>>("TimeStepper");
  TestHelpers::db::test_reference_tag<
      Tags::TimeStepperRef<DummyType, DummyType>>("TimeStepper");
}
