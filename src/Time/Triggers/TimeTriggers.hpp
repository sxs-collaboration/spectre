// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "Time/Triggers/NearTimes.hpp"
#include "Time/Triggers/SlabCompares.hpp"
#include "Time/Triggers/Slabs.hpp"
#include "Time/Triggers/TimeCompares.hpp"
#include "Time/Triggers/Times.hpp"
#include "Utilities/TMPL.hpp"

namespace Triggers {
/// \ingroup EventsAndTriggersGroup
/// Typelist of Time triggers
using time_triggers =
    tmpl::list<NearTimes, SlabCompares, Slabs, TimeCompares, Times>;
}  // namespace Triggers
