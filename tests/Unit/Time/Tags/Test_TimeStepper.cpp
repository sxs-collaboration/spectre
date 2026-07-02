// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <optional>
#include <string>
#include <type_traits>
#include <utility>

#include "DataStructures/DataBox/DataBox.hpp"
#include "Helpers/DataStructures/DataBox/TestHelpers.hpp"
#include "Options/Protocols/FactoryCreation.hpp"
#include "Time/Tags/TimeStepper.hpp"
#include "Time/TimeSteppers/AdamsMoultonPc.hpp"
#include "Time/TimeSteppers/LtsError.hpp"
#include "Time/TimeSteppers/LtsTimeStepper.hpp"
#include "Time/TimeSteppers/Rk3HesthavenSsp.hpp"
#include "Time/TimeSteppers/TimeStepper.hpp"
#include "Utilities/ProtocolHelpers.hpp"
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
              tmpl::list<Tags::TimeStepperRef<FakeTimeStepper, FakeTimeStepper,
                                              std::bool_constant<false>>,
                         Tags::LtsOrError>>);
static_assert(std::is_same_v<
              time_stepper_ref_tags<FakeTimeStepper, true>,
              tmpl::list<Tags::TimeStepperRef<FakeTimeStepper, FakeTimeStepper,
                                              std::bool_constant<true>>,
                         Tags::LtsOrError>>);
static_assert(
    std::is_same_v<time_stepper_ref_tags<MoreSpecificFakeTimeStepper>,
                   tmpl::list<Tags::TimeStepperRef<MoreSpecificFakeTimeStepper,
                                                   MoreSpecificFakeTimeStepper,
                                                   std::bool_constant<false>>,
                              Tags::TimeStepperRef<FakeTimeStepper,
                                                   MoreSpecificFakeTimeStepper,
                                                   std::bool_constant<false>>,
                              Tags::LtsOrError>>);

template <bool LocalTimeStepping>
struct Metavariables {
  static constexpr bool local_time_stepping = LocalTimeStepping;

  struct factory_creation
      : tt::ConformsTo<Options::protocols::FactoryCreation> {
    using factory_classes = tmpl::map<
        tmpl::pair<TimeStepper, tmpl::list<TimeSteppers::AdamsMoultonPc<false>,
                                           TimeSteppers::AdamsMoultonPc<true>,
                                           TimeSteppers::Rk3HesthavenSsp>>>;
  };
};

template <bool LocalTimeStepping, bool MonotonicLts, typename Stepper,
          typename... Args>
void try_create(Args&&... args) {
  Tags::ConcreteTimeStepper<TimeStepper, MonotonicLts>::
      template create_from_options<Metavariables<LocalTimeStepping>>(
          std::make_unique<Stepper>(std::forward<Args>(args)...));
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Time.Tags.TimeStepper", "[Unit][Time]") {
  TestHelpers::db::test_simple_tag<Tags::ConcreteTimeStepper<DummyType>>(
      "ConcreteTimeStepper");
  TestHelpers::db::test_simple_tag<Tags::TimeStepper<DummyType>>("TimeStepper");
  TestHelpers::db::test_reference_tag<
      Tags::TimeStepperRef<DummyType, DummyType, std::bool_constant<false>>>(
      "TimeStepper");
  TestHelpers::db::test_reference_tag<Tags::LtsOrError>("TimeStepper");

  register_factory_classes_with_charm<Metavariables<false>>();

  // Check that everything is allowed in GTS except variable-order
  try_create<false, false, TimeSteppers::Rk3HesthavenSsp>();
  try_create<false, false, TimeSteppers::AdamsMoultonPc<false>>(3);
  try_create<false, false, TimeSteppers::AdamsMoultonPc<true>>(3);
  try_create<false, true, TimeSteppers::Rk3HesthavenSsp>();
  try_create<false, true, TimeSteppers::AdamsMoultonPc<false>>(3);
  try_create<false, true, TimeSteppers::AdamsMoultonPc<true>>(3);
  CHECK_THROWS_WITH(
      (try_create<false, false, TimeSteppers::AdamsMoultonPc<false>>(
          std::nullopt)),
      Catch::Matchers::ContainsSubstring(
          "Variable-order TimeSteppers are only supported in evolutions with "
          "local time-stepping."));

  // Check that LTS rejects non-LTS steppers
  CHECK_THROWS_WITH(
      (try_create<true, false, TimeSteppers::Rk3HesthavenSsp>()),
      Catch::Matchers::ContainsSubstring(
          "Chosen TimeStepper does not support conservative local "
          "time-stepping.  Valid time steppers for your settings: "
          "(AdamsMoultonPc,AdamsMoultonPcMonotonic)"));
  CHECK_THROWS_WITH(
      (try_create<true, true, TimeSteppers::Rk3HesthavenSsp>()),
      Catch::Matchers::ContainsSubstring(
          "Chosen TimeStepper does not support conservative local "
          "time-stepping.  Valid time steppers for your settings: "
          "(AdamsMoultonPcMonotonic)"));

  // Check that LTS rejects non-monotonic steppers if that's required.
  try_create<true, false, TimeSteppers::AdamsMoultonPc<false>>(3);
  try_create<true, false, TimeSteppers::AdamsMoultonPc<true>>(3);
  try_create<true, true, TimeSteppers::AdamsMoultonPc<true>>(3);
  CHECK_THROWS_WITH(
      (try_create<true, true, TimeSteppers::AdamsMoultonPc<false>>(3)),
      Catch::Matchers::ContainsSubstring(
          "Local time-stepping with control systems requires a monotonic "
          "TimeStepper to avoid deadlocks.  Valid time steppers for your "
          "settings: (AdamsMoultonPcMonotonic)"));
  try_create<true, false, TimeSteppers::AdamsMoultonPc<false>>(std::nullopt);
  try_create<true, false, TimeSteppers::AdamsMoultonPc<true>>(std::nullopt);
  try_create<true, true, TimeSteppers::AdamsMoultonPc<true>>(std::nullopt);
  CHECK_THROWS_WITH(
      (try_create<true, true, TimeSteppers::AdamsMoultonPc<false>>(
          std::nullopt)),
      Catch::Matchers::ContainsSubstring(
          "Local time-stepping with control systems requires a monotonic "
          "TimeStepper to avoid deadlocks.  Valid time steppers for your "
          "settings: (AdamsMoultonPcMonotonic)"));

  // Test LtsOrError
  {
    auto box =
        db::create<db::AddSimpleTags<Tags::ConcreteTimeStepper<TimeStepper>>,
                   time_stepper_ref_tags<TimeStepper>>(
            static_cast<std::unique_ptr<TimeStepper>>(
                std::make_unique<TimeSteppers::AdamsMoultonPc<true>>(3)));
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
