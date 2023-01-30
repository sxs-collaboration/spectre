// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

/// \cond
class Time;
class TimeDelta;
/// \endcond

/// \ingroup TimeGroup
/// Convert an arbitrary desired step to a valid LTS step size.
TimeDelta choose_lts_step_size(const Time& time, const double desired_step);
