// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Time/StepChoosers/PreventRapidIncrease.hpp"

#include <cmath>
#include <optional>

#include "DataStructures/MathWrapper.hpp"
#include "Time/History.hpp"
#include "Time/SlabRoundingError.hpp"
#include "Time/Time.hpp"
#include "Utilities/GenerateInstantiations.hpp"

namespace StepChoosers::PreventRapidIncrease_detail {
template <typename T>
bool limit_from_history(const TimeSteppers::ConstUntypedHistory<T>& history,
                        const double last_step) {
  if (history.size() < 2) {
    return false;
  }

  const double sloppiness =
      slab_rounding_error(history.front().time_step_id.step_time());
  std::optional<Time> previous_time{};
  double newer_step = abs(last_step);
  for (auto record = history.rbegin(); record != history.rend(); ++record) {
    const Time time = record->time_step_id.step_time();
    if (previous_time.has_value()) {
      const double this_step = abs(*previous_time - time).value();
      // Potential roundoff error comes from the inability to make
      // slabs exactly the same length.
      if (this_step < newer_step - sloppiness) {
        return true;
      }
      newer_step = this_step;
    }
    previous_time.emplace(time);
  }
  return false;
}

#define MATH_WRAPPER_TYPE(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATE(_, data)                                            \
  template bool limit_from_history(                                     \
      const TimeSteppers::ConstUntypedHistory<MATH_WRAPPER_TYPE(data)>& \
          history,                                                      \
      double last_step);

GENERATE_INSTANTIATIONS(INSTANTIATE, (MATH_WRAPPER_TYPES))

#undef INSTANTIATE
#undef MATH_WRAPPER_TYPE
}  // namespace StepChoosers::PreventRapidIncrease_detail
