// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <string>
#include <type_traits>

#include "Time/TimeSteppers/Factory.hpp"
#include "Time/TimeSteppers/ImexTimeStepper.hpp"
#include "Time/TimeSteppers/LtsTimeStepper.hpp"
#include "Time/TimeSteppers/TimeStepper.hpp"
#include "Utilities/TMPL.hpp"

namespace {
template <typename StepperList, typename InclusionPredicate>
void check_list_contents(const std::string& description,
                         InclusionPredicate&& inclusion_predicate) {
  INFO(description);
  static_assert(std::is_same_v<
                tmpl::list_difference<StepperList, TimeSteppers::time_steppers>,
                tmpl::list<>>);
  tmpl::for_each<StepperList>(
      [&]<typename Stepper>(tmpl::type_<Stepper> /*meta*/) {
        INFO(pretty_type::get_name<Stepper>());
        CHECK(inclusion_predicate(tmpl::type_<Stepper>{}));
      });
  tmpl::for_each<
      tmpl::list_difference<TimeSteppers::time_steppers, StepperList>>(
      [&]<typename Stepper>(tmpl::type_<Stepper> /*meta*/) {
        INFO(pretty_type::get_name<Stepper>());
        CHECK(not inclusion_predicate(tmpl::type_<Stepper>{}));
      });
}

SPECTRE_TEST_CASE("Unit.Time.TimeSteppers.Factory", "[Unit][Time]") {
  check_list_contents<TimeSteppers::time_steppers>(
      "time_steppers", []<typename Stepper>(tmpl::type_<Stepper> /*meta*/) {
        return std::is_convertible_v<Stepper*, TimeStepper*>;
      });
  check_list_contents<TimeSteppers::lts_time_steppers>(
      "lts_time_steppers", []<typename Stepper>(tmpl::type_<Stepper> /*meta*/) {
        return std::is_convertible_v<Stepper*, LtsTimeStepper*>;
      });
  check_list_contents<TimeSteppers::imex_time_steppers>(
      "imex_time_steppers",
      []<typename Stepper>(tmpl::type_<Stepper> /*meta*/) {
        return std::is_convertible_v<Stepper*, ImexTimeStepper*>;
      });
}
}  // namespace
