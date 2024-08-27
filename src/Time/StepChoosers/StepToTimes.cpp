// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Time/StepChoosers/StepToTimes.hpp"

namespace StepChoosers {
bool StepToTimes::uses_local_data() const { return false; }
bool StepToTimes::can_be_delayed() const { return false; }

PUP::able::PUP_ID StepToTimes::my_PUP_ID = 0;  // NOLINT
}  // namespace StepChoosers
