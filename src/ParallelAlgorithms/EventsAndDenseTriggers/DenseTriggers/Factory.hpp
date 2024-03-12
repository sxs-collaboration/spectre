// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "ParallelAlgorithms/EventsAndDenseTriggers/DenseTriggers/Filter.hpp"
#include "ParallelAlgorithms/EventsAndDenseTriggers/DenseTriggers/Or.hpp"
#include "ParallelAlgorithms/EventsAndDenseTriggers/DenseTriggers/Times.hpp"

namespace DenseTriggers {
using standard_dense_triggers =
    tmpl::list<DenseTriggers::Filter, DenseTriggers::Or, DenseTriggers::Times>;
}  // namespace DenseTriggers
