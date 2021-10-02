// Distributed under the MIT License.
// See LICENSE.txt for details.

/// \file
/// Defines base class StepController

#pragma once

#include <pup.h>

#include "Parallel/CharmPupable.hpp"
#include "Time/Time.hpp"
#include "Utilities/TMPL.hpp"

/// \ingroup TimeSteppersGroup
///
/// Holds all the StepControllers
namespace StepControllers {}

/// \ingroup TimeSteppersGroup
///
/// StepControllers take desired step sizes (generally determined by
/// StepChoosers) and convert them into TimeDeltas compatible with the
/// slab requirements.
class StepController : public PUP::able {
 public:
  /// \cond HIDDEN_SYMBOLS
  WRAPPED_PUPable_abstract(StepController);  // NOLINT
  /// \endcond

  virtual TimeDelta choose_step(const Time& time,
                                double desired_step) const = 0;
};
