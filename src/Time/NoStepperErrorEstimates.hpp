// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "DataStructures/DataBox/Protocols/Mutator.hpp"
#include "Time/Tags/StepperErrorEstimatesEnabled.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/ProtocolHelpers.hpp"
#include "Utilities/TMPL.hpp"

/// Initialization mutator disabling error estimates for executables
/// with non-standard time-stepping.
struct NoStepperErrorEstimates : tt::ConformsTo<db::protocols::Mutator> {
  using return_tags = tmpl::list<::Tags::StepperErrorEstimatesEnabled>;
  using argument_tags = tmpl::list<>;
  static void apply(const gsl::not_null<bool*> needed) { *needed = false; }
};
