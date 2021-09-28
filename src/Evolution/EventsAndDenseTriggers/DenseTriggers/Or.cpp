// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/EventsAndDenseTriggers/DenseTriggers/Or.hpp"

#include <pup_stl.h>
#include <utility>

namespace DenseTriggers {
Or::Or(std::vector<std::unique_ptr<DenseTrigger>> triggers)
    : triggers_(std::move(triggers)) {}

void Or::pup(PUP::er& p) {
  DenseTrigger::pup(p);
  p | triggers_;
}

PUP::able::PUP_ID Or::my_PUP_ID = 0;  // NOLINT
}  // namespace DenseTriggers
