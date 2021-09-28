// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

/// \cond
class Time;
/// \endcond

/// \ingroup TimeGroup
/// Scale of the roundoff error incurred from inexact slab operations
/// near the given time.
double slab_rounding_error(const Time& time);
