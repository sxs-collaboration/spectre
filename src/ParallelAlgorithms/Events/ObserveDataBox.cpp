// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "ParallelAlgorithms/Events/ObserveDataBox.hpp"

#include <pup.h>

namespace Events {
ObserveDataBox::ObserveDataBox(CkMigrateMessage* /*m*/) {}

void ObserveDataBox::pup(PUP::er& p) { Event::pup(p); }

PUP::able::PUP_ID ObserveDataBox::my_PUP_ID = 0;  // NOLINT
}  // namespace Events
