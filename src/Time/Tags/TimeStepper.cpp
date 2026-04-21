// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Time/Tags/TimeStepper.hpp"

#include <initializer_list>
#include <memory>
#include <string>
#include <typeinfo>
#include <vector>

#include "DataStructures/TaggedVariant.hpp"
#include "Time/LtsMode.hpp"
#include "Time/TimeSteppers/Factory.hpp"
#include "Time/TimeSteppers/ImexTimeStepper.hpp"
#include "Time/TimeSteppers/LtsError.hpp"
#include "Time/TimeSteppers/LtsTimeStepper.hpp"
#include "Time/TimeSteppers/TimeStepper.hpp"
#include "Utilities/Algorithm.hpp"
#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/PrettyType.hpp"
#include "Utilities/TMPL.hpp"

namespace Tags {
namespace {
// Get a list of the valid time steppers for given options.  This
// assumes that none of the validity criteria depend on the options
// passed to the time-stepper constructors.
template <typename... F>
std::vector<std::string> list_valid_time_steppers(const F&... checks) {
  std::vector<std::string> valid_names{};
  tmpl::for_each<TimeSteppers::time_steppers>(
      [&]<typename Stepper>(tmpl::type_<Stepper> /*meta*/) {
        if ((... and checks(Stepper{}))) {
          valid_names.push_back(pretty_type::name<Stepper>());
        }
      });
  return valid_names;
}

// Use structs instead of lambdas for these requirements to avoid an nvcc bug.
struct ConservationRequirement {
  ::LtsMode lts_mode;

  bool operator()(const ::TimeStepper& local_time_stepper) const {
    return lts_mode != ::LtsMode::Conservative or
           dynamic_cast<const LtsTimeStepper*>(&local_time_stepper) != nullptr;
  }
};

struct MonotonicityRequirement {
  bool monotonic_lts;
  ::LtsMode lts_mode;

  bool operator()(const ::TimeStepper& local_time_stepper) const {
    return not monotonic_lts or lts_mode == ::LtsMode::Off or
           local_time_stepper.monotonic();
  }
};

struct FactoryRequirement {
  std::initializer_list<const std::type_info*> factory_types;

  template <typename LocalStepper>
  bool operator()(const LocalStepper& /*local_time_stepper*/) const {
    return alg::found_if(factory_types, [](const std::type_info* const info) {
      return *info == typeid(LocalStepper);
    });
  }
};
}  // namespace

template <typename StepperType, bool MonotonicLts>
void ConcreteTimeStepper<StepperType, MonotonicLts>::validate_time_stepper(
    const StepperType& time_stepper, const ::LtsMode lts_mode,
    std::initializer_list<const std::type_info*> factory_types) {
  if (lts_mode == ::LtsMode::Off and
      variants::holds_alternative<TimeSteppers::Tags::VariableOrder>(
          time_stepper.order())) {
    ERROR_NO_TRACE(
        "Variable-order TimeSteppers are only supported in evolutions with "
        "local time-stepping.");
  }

  const ConservationRequirement conservation_requirement{lts_mode};
  const MonotonicityRequirement monotonicity_requirement{MonotonicLts,
                                                         lts_mode};
  // We don't have to check this one because the option parser
  // enforces it, but we should try not to print invalid values in the
  // suggestion lists in the errors.
  const FactoryRequirement factory_requirement{factory_types};

  if (not conservation_requirement(time_stepper)) {
    ERROR_NO_TRACE(
        "Chosen TimeStepper does not support conservative local "
        "time-stepping.  Valid time steppers for your settings: "
        << list_valid_time_steppers(conservation_requirement,
                                    monotonicity_requirement,
                                    factory_requirement));
  }
  if (not monotonicity_requirement(time_stepper)) {
    ERROR_NO_TRACE(
        "Local time-stepping with control systems requires a monotonic "
        "TimeStepper to avoid deadlocks.  Valid time steppers for your "
        "settings: "
        << list_valid_time_steppers(conservation_requirement,
                                    monotonicity_requirement,
                                    factory_requirement));
  }
}

const LtsTimeStepper& LtsOrError::get(const ::TimeStepper& stepper) {
  if (const auto* const lts_stepper =
          dynamic_cast<const LtsTimeStepper*>(&stepper)) {
    return *lts_stepper;
  } else {
    static const TimeSteppers::LtsError lts_error{};
    return lts_error;
  }
}

#define STEPPER_TYPE(data) BOOST_PP_TUPLE_ELEM(0, data)
#define MONOTONIC_LTS(data) BOOST_PP_TUPLE_ELEM(1, data)

#define INSTANTIATE(_, data) \
  template struct ConcreteTimeStepper<STEPPER_TYPE(data), MONOTONIC_LTS(data)>;

GENERATE_INSTANTIATIONS(INSTANTIATE,
                        (::TimeStepper, LtsTimeStepper, ImexTimeStepper),
                        (true, false))

#undef INSTANTIATE
#undef MONOTONIC_LTS
#undef STEPPER_TYPE
}  // namespace Tags
