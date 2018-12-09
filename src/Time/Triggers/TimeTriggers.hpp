// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "Time/Triggers/EveryNSlabs.hpp"
#include "Time/Triggers/PastTime.hpp"
#include "Time/Triggers/SpecifiedSlabs.hpp"
#include "Utilities/TMPL.hpp"

namespace Triggers {
/// \ingroup EventsAndTriggersGroup
/// Typelist of Time triggers
using time_triggers = tmpl::list<Registrars::EveryNSlabs, Registrars::PastTime,
                                 Registrars::SpecifiedSlabs>;
}  // namespace Triggers
