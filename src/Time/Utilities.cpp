// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Time/Utilities.hpp"

#include <cmath>
#include <limits>

#include "Time/Slab.hpp"
#include "Time/Time.hpp"

double slab_rounding_error(const Time& time) {
  return 4.0 * std::numeric_limits<double>::epsilon() *
         (std::abs(time.value()) + std::abs(time.slab().duration().value()));
}
