// Distributed under the MIT License.
// See LICENSE.txt for details.

/// \file
/// Defines base class StepController

#pragma once

#include "Time/Time.hpp"
#include "Utilities/TMPL.hpp"

/// \ingroup TimeSteppersGroup
///
/// Holds all the StepControllers
namespace StepControllers {
class BinaryFraction;
class FullSlab;
class SimpleTimes;
class SplitRemaining;
}  // namespace StepControllers

/// \ingroup TimeSteppersGroup
///
/// StepControllers take desired step sizes (generally determined by
/// StepChoosers) and convert them into TimeDeltas compatible with the
/// slab requirements.
class StepController {
 public:
  using creatable_classes = tmpl::list<
    StepControllers::BinaryFraction,
    StepControllers::FullSlab,
    StepControllers::SimpleTimes,
    StepControllers::SplitRemaining>;

  /// \cond HIDDEN_SYMBOLS
  StepController() = default;
  StepController(const StepController&) = default;
  StepController(StepController&&) = default;
  StepController& operator=(const StepController&) = default;
  StepController& operator=(StepController&&) = default;
  virtual ~StepController() = default;
  /// \endcond

  virtual TimeDelta choose_step(const Time& time,
                                double desired_step) const noexcept = 0;
};

#include "Time/StepControllers/BinaryFraction.hpp"
#include "Time/StepControllers/FullSlab.hpp"
#include "Time/StepControllers/SimpleTimes.hpp"
#include "Time/StepControllers/SplitRemaining.hpp"
