// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Time/ChooseLtsStepSize.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>

#include "Time/Time.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"

TimeDelta choose_lts_step_size(const Time& time, const double desired_step) {
  ASSERT((time.fraction().denominator() &
          (time.fraction().denominator() - 1)) == 0,
         "Not at a binary-fraction time within slab: " << time.fraction());
  const TimeDelta full_slab =
      desired_step > 0.0 ? time.slab().duration() : -time.slab().duration();
  const double desired_step_count = full_slab.value() / desired_step;
  const size_t desired_step_power =
      desired_step_count <= 1
          ? 0
          : static_cast<size_t>(std::ceil(std::log2(desired_step_count)));

  // Ensure we will hit the slab boundary if we continue taking
  // constant-sized steps.
  const auto step_count =
      std::max(static_cast<decltype(time.fraction().denominator())>(
                   two_to_the(desired_step_power)),
               time.fraction().denominator());

  return full_slab / step_count;
}
