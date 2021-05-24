// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "Evolution/EventsAndDenseTriggers/DenseTriggers/Filter.hpp"
#include "Evolution/EventsAndDenseTriggers/DenseTriggers/Or.hpp"
#include "Evolution/EventsAndDenseTriggers/DenseTriggers/Times.hpp"

namespace DenseTriggers {
using standard_dense_triggers =
    tmpl::list<DenseTriggers::Filter, DenseTriggers::Or, DenseTriggers::Times>;
}  // namespace DenseTriggers
