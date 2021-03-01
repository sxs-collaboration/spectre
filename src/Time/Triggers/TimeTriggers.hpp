// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "Time/Triggers/NearTimes.hpp"
#include "Time/Triggers/PastTime.hpp"
#include "Time/Triggers/SlabCompares.hpp"
#include "Time/Triggers/Slabs.hpp"
#include "Time/Triggers/TimeCompares.hpp"
#include "Time/Triggers/Times.hpp"
#include "Utilities/TMPL.hpp"

namespace Triggers {
/// \ingroup EventsAndTriggersGroup
/// Typelist of Time triggers
using time_triggers =
    tmpl::list<Registrars::NearTimes, Registrars::PastTime,
               Registrars::SlabCompares, Registrars::Slabs,
               Registrars::TimeCompares, Registrars::Times>;
}  // namespace Triggers
