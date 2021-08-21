// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "Elliptic/Triggers/EveryNIterations.hpp"
#include "Elliptic/Triggers/HasConverged.hpp"
#include "ParallelAlgorithms/EventsAndTriggers/LogicalTriggers.hpp"
#include "Utilities/TMPL.hpp"

/// Triggers for elliptic executables
namespace elliptic::Triggers {
template <typename Label>
using all_triggers =
    tmpl::push_front<::Triggers::logical_triggers, EveryNIterations<Label>,
                     HasConverged<Label>>;
}  // namespace elliptic::Triggers
