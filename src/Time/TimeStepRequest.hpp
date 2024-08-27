// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <optional>

/// \cond
namespace PUP {
class er;
}  // namespace PUP
/// \endcond

/// Information on a requested time-step size returned by `StepChooser`s.
///
/// Requests made using this struct control most of the changing of
/// slab and local-time-stepping step sizes.  Slab-size changing is
/// more complex, as it is controlled using the `ChangeSlabSize`
/// event, which may run zero or more times per slab, and may have the
/// application of its result delayed.  The step size for LTS is
/// adjusted using a constant list of `StepChoosers` on every step.
///
/// For slab-size adjustment (including the global time step when not
/// using LTS), a "goal" step size is maintained from step to step.
/// It is initially the initial slab size from the input file, and can
/// be adjusted by any request setting the `size_goal` field of this
/// struct.  If multiple requests set new goals at the same time, the
/// smallest value is used.  At the start of each slab, the size is
/// chosen to be the smallest of the current goal and the temporary
/// limits imposed by any requests processed at that time.  For the
/// LTS step size, the current step size is used as the goal if no
/// request sets a new one.
///
/// Any request with either the `size` or `end` fields set introduces
/// a temporary step size limit.  If only one of the two is set, the
/// chosen step size will be limited to be not larger than `*size`, or
/// not go beyond the time `*end`.  If both are set, the tighter limit
/// may be ignored if the looser limit is chosen as the actual step,
/// but the step will not be chosen to be intermediate between two
/// temporary limits from the same request object.
///
/// The hard limits are not used in setting the step size, but if the
/// actual step does not satisfy all hard limits from all requests the
/// evolution will be terminated with an error.
///
/// \note All step sizes are subject to roundoff error in either
/// direction except for two cases:
/// * Slab sizes imposed by the `end` limit of a request will always
///   end precisely at the requested time.
/// * LTS step sizes within a single slab will always be considered to
///   have size ratios that are exactly powers of two for the purposes
///   of the `size_goal` and `size` limits.
struct TimeStepRequest {
  std::optional<double> size_goal{};

  std::optional<double> size{};
  std::optional<double> end{};

  std::optional<double> size_hard_limit{};
  std::optional<double> end_hard_limit{};

  void pup(PUP::er& p);
};

bool operator==(const TimeStepRequest& a, const TimeStepRequest& b);
bool operator!=(const TimeStepRequest& a, const TimeStepRequest& b);
