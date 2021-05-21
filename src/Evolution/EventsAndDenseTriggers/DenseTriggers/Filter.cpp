// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/EventsAndDenseTriggers/DenseTriggers/Filter.hpp"

#include <pup_stl.h>
#include <utility>

namespace DenseTriggers {
Filter::Filter(std::unique_ptr<DenseTrigger> trigger,
               std::unique_ptr<Trigger> filter) noexcept
    : trigger_(std::move(trigger)), filter_(std::move(filter)) {}

void Filter::pup(PUP::er& p) noexcept {
  DenseTrigger::pup(p);
  p | trigger_;
  p | filter_;
}

PUP::able::PUP_ID Filter::my_PUP_ID = 0;  // NOLINT
}  // namespace DenseTriggers
