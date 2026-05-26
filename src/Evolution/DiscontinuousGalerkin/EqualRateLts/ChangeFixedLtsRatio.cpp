// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/DiscontinuousGalerkin/EqualRateLts/ChangeFixedLtsRatio.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <map>
#include <optional>
#include <vector>

#include "Time/TimeStepId.hpp"
#include "Utilities/Algorithm.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/Gsl.hpp"

namespace evolution::dg::Actions {
bool ChangeFixedLtsRatio::Impl::apply(
    const gsl::not_null<std::optional<size_t>*> fixed_lts_ratio,
    const gsl::not_null<std::map<StepId, size_t>*> expected_messages_map,
    const gsl::not_null<std::map<StepId, std::vector<double>>*>
        new_step_size_messages_map,
    const TimeStepId& time_step_id, const uint64_t step_number_within_slab) {
  if (time_step_id.substep() != 0) {
    return true;
  }

  const StepId step_id{time_step_id.slab_number(), step_number_within_slab};

  if (expected_messages_map->empty() or
      expected_messages_map->begin()->first > step_id) {
    return true;
  }

  double desired_step_size = std::numeric_limits<double>::infinity();
  // We don't keep a continuous count of the step number across slabs,
  // and it's not easy to predict exactly when a slab change is going
  // to happen, so just treat anything past the end of a slab as at
  // the start of the next slab.
  for (const auto& [receive_id, expected_messages] : *expected_messages_map) {
    if (receive_id > step_id) {
      break;
    }

    ASSERT(expected_messages > 0,
           "Should only create map entries when sending messages.");

    const auto new_step_size_messages =
        new_step_size_messages_map->find(receive_id);
    if (new_step_size_messages == new_step_size_messages_map->end() or
        new_step_size_messages->second.size() < expected_messages) {
      return false;
    }

    ASSERT(new_step_size_messages->second.size() == expected_messages,
           "Received too many messages at step " << receive_id);

    desired_step_size = std::min(
        desired_step_size, *alg::min_element(new_step_size_messages->second));
  }

  // We have all the messages.  Actually modify things.

  const double desired_ratio =
      time_step_id.step_time().slab().duration().value() / desired_step_size;
  const size_t new_step_ratio = desired_ratio < 1.0
                                    ? 1
                                    : two_to_the(static_cast<size_t>(
                                          std::ceil(std::log2(desired_ratio))));

  ASSERT(fixed_lts_ratio->has_value(),
         "Attempting to adjust FixedLtsRatio when not set.");
  **fixed_lts_ratio = new_step_ratio;

  for (auto expected_messages = expected_messages_map->begin();
       expected_messages != expected_messages_map->end() and
       expected_messages->first <= step_id;
       expected_messages = expected_messages_map->erase(expected_messages)) {
    const auto first_new_step_size_messages =
        new_step_size_messages_map->begin();
    ASSERT(first_new_step_size_messages->first == expected_messages->first,
           "Received unexpected change for step "
               << first_new_step_size_messages->first);
    new_step_size_messages_map->erase(first_new_step_size_messages);
  }
  return true;
}
}  // namespace evolution::dg::Actions
