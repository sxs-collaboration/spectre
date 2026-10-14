// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <string>

namespace sys {
/// @{
/// \ingroup UtilitiesGroup
/// \brief Format the wall time in DD-HH:MM:SS format.
///
/// If the walltime is shorter than a day, omit the `DD-` part.
std::string pretty_wall_time(double total_seconds);

std::string pretty_wall_time();
/// @}
}  // namespace sys
