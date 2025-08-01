// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "ParallelAlgorithms/Events/ObserveTimeStep.hpp"

#include <cstddef>
#include <sstream>
#include <string>

#include "Utilities/System/ParallelInfo.hpp"

namespace PUP {
class er;
}  // namespace PUP

namespace Events::detail {
std::string FormatTimeOutput::operator()(
    const double time, const size_t /* num_points */,
    const double /* slab_size */, const double /* min_time_step */,
    const double /* max_time_step */, const double /* effective_time_step */,
    const double min_wall_time, const double max_wall_time) const {
  std::stringstream ss;
  ss << "Simulation time: " << std::to_string(time)
     << "\n  Wall time: " << sys::pretty_wall_time(min_wall_time) << " (min) - "
     << sys::pretty_wall_time(max_wall_time) << " (max)";
  return ss.str();
}

void FormatTimeOutput::pup(PUP::er& /*p*/) {}
}  // namespace Events::detail
